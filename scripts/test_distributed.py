#!/usr/bin/env python3
"""
Simple script to test distributed setup before running full training.
Usage: deepspeed --num_gpus=4 scripts/test_distributed.py
"""

import os
import torch
import torch.distributed as dist

def test_distributed():
    """Test basic distributed setup."""
    print("\n" + "="*60)
    print("Testing Distributed Setup")
    print("="*60)

    # Check environment variables
    print("\n1. Environment Variables:")
    for var in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT']:
        print(f"   {var}: {os.environ.get(var, 'NOT SET')}")

    # Check CUDA
    print("\n2. CUDA Information:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    print(f"   CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

    # Setup distributed
    print("\n3. Initializing Distributed Backend:")

    if 'RANK' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = local_rank
        world_size = torch.cuda.device_count()
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    print(f"   Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        print(f"   Set CUDA device to: {local_rank}")
        print(f"   Current device: {torch.cuda.current_device()}")

    if world_size > 1:
        print(f"   Initializing process group...")
        try:
            if not dist.is_initialized():
                dist.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    world_size=world_size,
                    rank=rank
                )
                print(f"   ✓ Process group initialized successfully")
            else:
                print(f"   ⚠ Process group already initialized")

            # Test barrier
            print(f"   Testing barrier synchronization...")
            if dist.is_initialized():
                dist.barrier()
                print(f"   ✓ Barrier successful")

            # Test all_reduce
            print(f"   Testing all_reduce operation...")
            tensor = torch.tensor([rank], dtype=torch.float32).cuda()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected = sum(range(world_size))
            assert tensor.item() == expected, f"Expected {expected}, got {tensor.item()}"
            print(f"   ✓ All_reduce successful (sum of ranks: {tensor.item()})")

        except Exception as e:
            print(f"   ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print(f"   Single GPU mode, skipping distributed init")

    print("\n" + "="*60)
    print("✓ All tests passed!" if world_size > 1 else "Single GPU test complete")
    print("="*60 + "\n")

    return True

if __name__ == "__main__":
    success = test_distributed()
    if not success:
        exit(1)
