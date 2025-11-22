"""Main CAD-MLLM model implementation."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

from .config import CADMLLMConfig
from .encoders import TextEncoder, ImageEncoder, PointCloudEncoder, MichelangeloPointEncoder
from .projectors import MLPProjector, IdentityProjector


class CADMLLMModel(nn.Module):
    """CAD-MLLM: Multimodal Large Language Model for CAD Generation.

    This model can process text, images, and point clouds to generate
    parametric CAD command sequences. The architecture consists of:
    1. Modality-specific encoders (frozen for images/point clouds)
    2. Projection layers to align features with LLM space
    3. LLM with LoRA for efficient fine-tuning

    Args:
        config: Model configuration
    """

    def __init__(self, config: CADMLLMConfig):
        super().__init__()
        self.config = config

        dtype_mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.torch_dtype = dtype_mapping.get(self.config.dtype, torch.bfloat16)

        # Load LLM and tokenizer
        self._load_llm()

        # Initialize encoders and projectors
        self._init_encoders()
        self._init_projectors()

        # Set up LoRA if enabled
        if config.use_lora:
            self._setup_lora()

    def _load_llm(self):
        """Load the base LLM and tokenizer."""
        print(f"Loading LLM: {self.config.llm_model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.llm_model_name,
            trust_remote_code=True,
            padding_side="right",
        )

        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load model
        # dtype_mapping = {
        #     "float32": torch.float32,
        #     "float16": torch.float16,
        #     "bfloat16": torch.bfloat16,
        # }
        # torch_dtype = dtype_mapping.get(self.config.dtype, torch.bfloat16)

        self.llm = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model_name,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            device_map=self.config.device if self.config.device != "cpu" else None,
        )

        # Update config with actual vocab size and pad token
        self.config.vocab_size = len(self.tokenizer)
        self.config.pad_token_id = self.tokenizer.pad_token_id

        print(f"LLM loaded. Vocab size: {self.config.vocab_size}")

    def _init_encoders(self):
        """Initialize modality-specific encoders."""
        # Text encoder (uses LLM's embedding layer)
        self.text_encoder = TextEncoder(
            self.llm.get_input_embeddings(),
            freeze=self.config.freeze_text_encoder,
        )

        # Image encoder (extensible, initialized but not required for text-only)
        self.image_encoder = None
        self.has_image_encoder = False

        # Point cloud encoder (extensible, initialized but not required for text-only)
        self.point_encoder = None
        self.has_point_encoder = False

    def enable_image_encoder(self):
        """Enable image encoder for multimodal training."""
        # Load model

        if self.image_encoder is None:
            print(f"Initializing image encoder: {self.config.image_encoder_name}")
            self.image_encoder = ImageEncoder(
                model_name=self.config.image_encoder_name,
                torch_dtype=self.torch_dtype,
                freeze=self.config.freeze_image_encoder,
            )
            self.image_encoder = self.image_encoder.to(self.config.device)
            self.has_image_encoder = True

    def enable_point_encoder(self):
        """Enable point cloud encoder for multimodal training."""
        if self.point_encoder is None:
            print("Initializing point cloud encoder")

            self.point_encoder = MichelangeloPointEncoder(
                encoder_cfg_path=self.config.miche_encoder_cfg_path,
                encoder_sd_path=self.config.miche_encoder_sd_path,
                num_points=self.config.num_points,
                dtype=self.torch_dtype,
                freeze=self.config.freeze_miche_encoder,
                device=self.config.device
            )
            self.point_encoder = self.point_encoder.to(self.config.device)
            self.point_encoder = self.point_encoder.to(self.torch_dtype)
            self.has_point_encoder = True

    def _init_projectors(self):
        """Initialize projection layers for feature alignment."""
        hidden_size = self.llm.config.hidden_size

        # Text doesn't need projection (already in LLM space)
        self.text_projector = IdentityProjector()

        # Image projector (will be initialized when image encoder is enabled)
        self.image_projector = None

        # Point cloud projector (will be initialized when point encoder is enabled)
        self.point_projector = None

    def enable_image_projector(self):
        """Enable image projector when image encoder is active."""
        if self.has_image_encoder and self.image_projector is None:
            hidden_size = self.llm.config.hidden_size
            image_hidden_size = self.image_encoder.hidden_size

            print(f"Initializing image projector: {image_hidden_size} -> {hidden_size}")
            self.image_projector = MLPProjector(
                input_dim=image_hidden_size,
                output_dim=hidden_size,
                hidden_dim=self.config.projector_hidden_dim,
                num_layers=self.config.projector_num_layers,
            )
            self.image_projector = self.image_projector.to(self.config.device)
            self.image_projector = self.image_projector.to(self.torch_dtype)

    def enable_point_projector(self):
        """Enable point cloud projector when point encoder is active."""
        if self.has_point_encoder and self.point_projector is None:
            hidden_size = self.llm.config.hidden_size

            # Prefer the encoder's true output dim if it exists (Michelangelo case)
            point_hidden_size = getattr(self.point_encoder, "output_dim", -1)

            print(f"Initializing point projector: {point_hidden_size} -> {hidden_size}")
            self.point_projector = MLPProjector(
                input_dim=point_hidden_size,
                output_dim=hidden_size,
                hidden_dim=self.config.projector_hidden_dim,
                num_layers=self.config.projector_num_layers,
            )
            self.point_projector = self.point_projector.to(self.config.device)
            self.point_projector = self.point_projector.to(self.torch_dtype)


    def _setup_lora(self):
        """Setup LoRA for efficient fine-tuning."""
        print(f"Setting up LoRA with r={self.config.lora_r}, alpha={self.config.lora_alpha}")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
        )

        self.llm = get_peft_model(self.llm, lora_config)
        self.llm.print_trainable_parameters()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        point_clouds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, Any]:
        """Forward pass of CAD-MLLM.

        Args:
            input_ids: Text input token IDs (batch_size, seq_len)
            attention_mask: Attention mask for text (batch_size, seq_len)
            pixel_values: Image inputs (batch_size, channels, height, width)
            point_clouds: Point cloud inputs (batch_size, num_points, 3)
            labels: Target token IDs for training (batch_size, seq_len)
            return_dict: Whether to return a dictionary

        Returns:
            Dictionary containing loss and logits
        """
        # Encode and project different modalities
        embeddings_list = []
        attention_masks = []

        # Process text input (always present)
        if input_ids is not None:
            text_embeds = self.text_encoder(input_ids)
            text_embeds = self.text_projector(text_embeds)
            embeddings_list.append(text_embeds)
            if attention_mask is not None:
                attention_masks.append(attention_mask)

        # Process image input (if present and encoder is enabled)
        if pixel_values is not None and self.has_image_encoder:
            pixel_values = pixel_values.to(self.torch_dtype)
            image_features = self.image_encoder(pixel_values)
            if self.image_projector is not None:
                image_embeds = self.image_projector(image_features)
                embeddings_list.append(image_embeds)
                # Create attention mask for image features
                batch_size, seq_len = image_embeds.shape[:2]
                image_mask = torch.ones(batch_size, seq_len, device=image_embeds.device)
                attention_masks.append(image_mask)

        # Process point cloud input (if present and encoder is enabled)
        if point_clouds is not None and self.has_point_encoder:
            point_clouds = point_clouds.to(self.torch_dtype)
            point_features = self.point_encoder(point_clouds)
            if self.point_projector is not None:
                point_embeds = self.point_projector(point_features)
                embeddings_list.append(point_embeds)
                # Create attention mask for point features
                batch_size, seq_len = point_embeds.shape[:2]
                point_mask = torch.ones(batch_size, seq_len, device=point_embeds.device)
                attention_masks.append(point_mask)

        # Concatenate all embeddings
        if len(embeddings_list) > 1:
            inputs_embeds = torch.cat(embeddings_list, dim=1)
            if attention_masks:
                attention_mask = torch.cat(attention_masks, dim=1)
        else:
            inputs_embeds = embeddings_list[0]

        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        text_prompt: str,
        pixel_values: Optional[torch.Tensor] = None,
        point_clouds: Optional[torch.Tensor] = None,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """Generate CAD sequence from multimodal inputs.

        Args:
            text_prompt: Text description of the CAD model
            pixel_values: Optional image inputs
            point_clouds: Optional point cloud inputs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated CAD sequence as a string
        """
        self.eval()

        # Tokenize text input
        inputs = self.tokenizer(
            text_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
        )

        input_ids = inputs["input_ids"].to(self.config.device)
        attention_mask = inputs["attention_mask"].to(self.config.device)

        # Move other inputs to device if present
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.config.device)
        if point_clouds is not None:
            point_clouds = point_clouds.to(self.config.device)

        # Encode inputs
        embeddings_list = []
        attention_masks = []

        # Text
        text_embeds = self.text_encoder(input_ids)
        text_embeds = self.text_projector(text_embeds)
        embeddings_list.append(text_embeds)
        attention_masks.append(attention_mask)

        # Image (if provided)
        if pixel_values is not None and self.has_image_encoder:
            # print(pixel_values.type)
            # print(pixel_values.shape)
            pixel_values = pixel_values.to(self.torch_dtype)
            image_features = self.image_encoder(pixel_values)
            if self.image_projector is not None:
                image_embeds = self.image_projector(image_features)
                embeddings_list.append(image_embeds)
                batch_size, seq_len = image_embeds.shape[:2]
                image_mask = torch.ones(batch_size, seq_len, device=image_embeds.device)
                attention_masks.append(image_mask)

        # Point cloud (if provided)
        if point_clouds is not None and self.has_point_encoder:
            point_clouds = point_clouds.to(self.torch_dtype)
            point_features = self.point_encoder(point_clouds)
            if self.point_projector is not None:
                point_embeds = self.point_projector(point_features)
                embeddings_list.append(point_embeds)
                batch_size, seq_len = point_embeds.shape[:2]
                point_mask = torch.ones(batch_size, seq_len, device=point_embeds.device)
                attention_masks.append(point_mask)

        # Concatenate embeddings
        # print(len(embeddings_list))
        inputs_embeds = torch.cat(embeddings_list, dim=1)
        attention_mask = torch.cat(attention_masks, dim=1)

        # Generate
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )

        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text

    def save_pretrained(self, save_directory: str):
        """Save model checkpoint."""
        import os

        os.makedirs(save_directory, exist_ok=True)

        # Save LLM (with LoRA if enabled)
        self.llm.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

        # Save projectors if they exist
        if self.image_projector is not None:
            torch.save(self.image_projector.state_dict(), os.path.join(save_directory, "image_projector.pt"))
        if self.point_projector is not None:
            torch.save(self.point_projector.state_dict(), os.path.join(save_directory, "point_projector.pt"))

        # Save config
        torch.save(self.config, os.path.join(save_directory, "config.pt"))

        print(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(cls, load_directory: str):
        """Load model checkpoint."""
        import os

        # Load config
        config = torch.load(os.path.join(load_directory, "config.pt"))

        # Create model
        model = cls(config)

        # Load projectors if they exist
        image_projector_path = os.path.join(load_directory, "image_projector.pt")
        if os.path.exists(image_projector_path):
            model.enable_image_encoder()
            model.enable_image_projector()
            model.image_projector.load_state_dict(torch.load(image_projector_path))

        point_projector_path = os.path.join(load_directory, "point_projector.pt")
        if os.path.exists(point_projector_path):
            model.enable_point_encoder()
            model.enable_point_projector()
            model.point_projector.load_state_dict(torch.load(point_projector_path))

        print(f"Model loaded from {load_directory}")
        return model
