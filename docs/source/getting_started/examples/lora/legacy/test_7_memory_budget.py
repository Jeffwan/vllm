"""Test 7: Memory budget verification for LoRA."""

import pytest
import torch
import gc
from vllm import LLM
from vllm.lora.request import LoRARequest
from test_config import BASE_MODEL, get_lora_path


class TestMemoryBudget:
    """Test suite for memory budget verification."""

    def get_gpu_memory_usage(self):
        """Get current GPU memory usage in bytes."""
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated()

    def test_lora_buffer_preallocation(self):
        """Test that LoRA buffers are pre-allocated at init."""
        gc.collect()
        torch.cuda.empty_cache()

        initial_memory = self.get_gpu_memory_usage()

        # Create LLM with LoRA enabled
        llm = LLM(
            model=BASE_MODEL,
            enable_lora=True,
            max_loras=4,
            max_cpu_loras=4,
            max_lora_rank=32,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            enforce_eager=True,
        )

        after_init_memory = self.get_gpu_memory_usage()
        init_overhead = after_init_memory - initial_memory

        # Load 4 LoRAs
        for i in range(4):
            lora_req = LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            llm.llm_engine.add_lora(lora_req)

        after_load_memory = self.get_gpu_memory_usage()
        load_overhead = after_load_memory - after_init_memory

        print(f"\n{'='*60}")
        print("MEMORY BUDGET ANALYSIS")
        print(f"{'='*60}")
        print(f"Initial GPU memory: {initial_memory / 1e9:.2f} GB")
        print(f"After LLM init: {after_init_memory / 1e9:.2f} GB")
        print(f"After loading 4 LoRAs: {after_load_memory / 1e9:.2f} GB")
        print(f"\nInit overhead (includes LoRA buffers): {init_overhead / 1e6:.2f} MB")
        print(f"Load overhead (should be minimal): {load_overhead / 1e6:.2f} MB")
        print(f"{'='*60}")

        # LoRA buffers should be pre-allocated, so loading should add minimal memory
        # (only CPU-side objects and metadata)
        # Note: Some memory may be allocated for activation, so allow some tolerance
        assert load_overhead < 500 * 1e6, \
            f"Loading LoRAs added too much GPU memory: {load_overhead / 1e6:.2f} MB"

        del llm
        gc.collect()
        torch.cuda.empty_cache()

        print("PASS: LoRA buffers are pre-allocated at init")

    def test_different_max_loras_memory(self):
        """Test memory impact of different max_loras settings."""
        results = []

        for max_loras in [1, 2, 4]:
            gc.collect()
            torch.cuda.empty_cache()

            initial = self.get_gpu_memory_usage()

            llm = LLM(
                model=BASE_MODEL,
                enable_lora=True,
                max_loras=max_loras,
                max_cpu_loras=max_loras,
                max_lora_rank=16,
                gpu_memory_utilization=0.7,
                trust_remote_code=True,
                enforce_eager=True,
            )

            after = self.get_gpu_memory_usage()
            overhead = after - initial
            results.append((max_loras, overhead))

            del llm
            gc.collect()
            torch.cuda.empty_cache()

        print(f"\n{'='*60}")
        print("MEMORY vs MAX_LORAS")
        print(f"{'='*60}")
        for max_loras, overhead in results:
            print(f"max_loras={max_loras}: {overhead / 1e9:.3f} GB")
        print(f"{'='*60}")

        # Memory should increase with max_loras
        for i in range(1, len(results)):
            assert results[i][1] >= results[i-1][1], \
                "Memory should increase with max_loras"

        print("PASS: Memory scales with max_loras")


# TODO: add a case that I specify --max_lora_rank but I use a large rank size.. what will happen?

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
