"""Inference script for CAD-MLLM."""

import argparse
import torch
from PIL import Image
import numpy as np

from cad_mllm.model import CADMLLMModel, CADMLLMConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with CAD-MLLM")

    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model checkpoint (if None, uses base model)")
    parser.add_argument("--llm_model_name", type=str, default="Qwen/Qwen3-4B", help="Name of the base LLM model")
    parser.add_argument("--image_encoder", type=str, default="facebook/dinov2-large")
    parser.add_argument("--image_path", type=str, default="")
    parser.add_argument("--pc_path", type=str, default="")
    parser.add_argument("--miche_encoder_cfg_path", type=str, default="configs/michelangelo_point_encoder_cfg.yaml")
    parser.add_argument("--miche_encoder_sd_path", type=str, default="checkpoints/michelangelo_point_encoder_state_dict.pt")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt describing the CAD model to generate")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling parameter")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type (float32/float16/bfloat16)")

    return parser.parse_args()


def main():
    """Main inference function."""
    args = parse_args()

    print("=" * 60)
    print("CAD-MLLM Inference")
    print("=" * 60)

    # Load model
    if args.model_path is not None:
        print(f"\nLoading trained model from {args.model_path}...")
        model = CADMLLMModel.from_pretrained(args.model_path)
    else:
        print(f"\nInitializing base model: {args.llm_model_name}...")
        config = CADMLLMConfig(
            llm_model_name=args.llm_model_name,
            image_encoder_name=args.image_encoder,
            use_lora=False,  # No LoRA for inference with base model
            device=args.device,
            dtype=args.dtype,
        )
        model = CADMLLMModel(config)

        pixel_values = None
        if args.image_path != "":
            image = Image.open(args.image_path)
            model.enable_image_encoder()
            model.enable_image_projector()
            pixel_values = model.image_encoder.preprocess(image)
            pixel_values = pixel_values.to(model.config.device)
        
        point_clouds = None
        if args.pc_path != "":
            points = np.load(args.pc_path)["points"]
            point_clouds = torch.from_numpy(points).unsqueeze(0)
            model.enable_point_encoder()
            model.enable_point_projector()
        

    model.eval()
    print("Model loaded successfully!")

    # Print prompt
    print("\n" + "=" * 60)
    print("Input Prompt:")
    print("=" * 60)
    print(args.prompt)

    # Generate
    print("\n" + "=" * 60)
    print("Generating CAD sequence...")
    print("=" * 60)

    with torch.no_grad():
        output = model.generate(
            text_prompt=args.prompt,
            pixel_values=pixel_values,
            point_clouds=point_clouds,
            max_length=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    # Print output
    print("\n" + "=" * 60)
    print("Generated CAD Sequence:")
    print("=" * 60)
    print(output)
    print("\n" + "=" * 60)


def interactive_mode():
    """Run in interactive mode for multiple generations."""
    parser = argparse.ArgumentParser(description="Run CAD-MLLM in interactive mode")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--llm_model_name", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    args = parser.parse_args()

    # Load model
    if args.model_path is not None:
        print(f"Loading trained model from {args.model_path}...")
        model = CADMLLMModel.from_pretrained(args.model_path)
    else:
        print(f"Initializing base model: {args.llm_model_name}...")
        config = CADMLLMConfig(
            llm_model_name=args.llm_model_name,
            use_lora=False,
            device=args.device,
            dtype=args.dtype,
        )
        model = CADMLLMModel(config)

    model.eval()
    print("Model loaded! Enter 'quit' or 'exit' to stop.\n")

    # Interactive loop
    while True:
        try:
            prompt = input("\nEnter CAD description: ").strip()

            if prompt.lower() in ["quit", "exit", "q"]:
                print("Exiting...")
                break

            if not prompt:
                continue

            print("\nGenerating...")
            with torch.no_grad():
                output = model.generate(
                    text_prompt=prompt,
                    max_length=512,
                    temperature=0.7,
                    top_p=0.9,
                )

            print("\nGenerated CAD sequence:")
            print("-" * 60)
            print(output)
            print("-" * 60)

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


if __name__ == "__main__":
    import sys

    # Check if running in interactive mode
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        main()
