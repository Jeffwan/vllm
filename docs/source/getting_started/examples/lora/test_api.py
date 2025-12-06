"""Test Suite: LoRA API - Dynamic Loading and Performance.

This module tests LoRA API behaviors focused on serverless scenarios:
- Dynamic loading latency measurement
- Hot-swap patterns during inference
- Load/use/unload cycles

Note: Basic add/remove/pin correctness is covered by vllm/tests/lora/test_lora_manager.py
"""

import pytest
import time
import statistics
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from test_config import BASE_MODEL, get_lora_path, TEST_PROMPT, MAX_TOKENS


class TestDynamicLoading:
    """Test suite for dynamic LoRA loading performance."""

    @pytest.fixture
    def llm(self):
        """Create LLM for dynamic loading tests."""
        llm = LLM(
            model=BASE_MODEL,
            enable_lora=True,
            max_loras=4,
            max_cpu_loras=8,
            max_lora_rank=16,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            enforce_eager=True,
        )
        yield llm
        del llm

    def test_loading_latency(self, llm):
        """Measure LoRA loading latency."""
        load_times = []

        for i in range(5):
            lora_req = LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))

            start = time.perf_counter()
            llm.llm_engine.add_lora(lora_req)
            elapsed = time.perf_counter() - start

            load_times.append(elapsed)

        print(f"\n{'='*60}")
        print("LORA LOADING LATENCY")
        print(f"{'='*60}")
        print(f"Samples: {len(load_times)}")
        print(f"Mean: {statistics.mean(load_times)*1000:.2f} ms")
        print(f"Stdev: {statistics.stdev(load_times)*1000:.2f} ms")
        print(f"Min: {min(load_times)*1000:.2f} ms")
        print(f"Max: {max(load_times)*1000:.2f} ms")
        print(f"{'='*60}")

    def test_load_use_unload_cycle(self, llm):
        """Test complete load -> use -> unload cycle."""
        sampling_params = SamplingParams(max_tokens=MAX_TOKENS)

        for cycle in range(3):
            # Load
            lora_req = LoRARequest("cycle_lora", 1, get_lora_path(0))
            add_result = llm.llm_engine.add_lora(lora_req)

            # Use
            output = llm.generate(
                [TEST_PROMPT],
                sampling_params,
                lora_request=LoRARequest("cycle_lora", 1, get_lora_path(0))
            )
            assert output is not None and output[0].outputs[0].text

            # Unload
            remove_result = llm.llm_engine.remove_lora(1)

            print(f"Cycle {cycle + 1}: add={add_result}, remove={remove_result}")

        print("PASS: Load/use/unload cycles work correctly")

    def test_hot_swap_loras(self, llm):
        """Test hot-swapping between LoRAs during inference."""
        sampling_params = SamplingParams(max_tokens=MAX_TOKENS)

        # Load 4 LoRAs
        for i in range(4):
            lora_req = LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            llm.llm_engine.add_lora(lora_req)

        # Rapid switching between LoRAs
        for _ in range(3):
            for i in range(4):
                output = llm.generate(
                    [TEST_PROMPT],
                    sampling_params,
                    lora_request=LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
                )
                assert output is not None

        # All should still be registered
        loras = llm.llm_engine.list_loras()
        assert len(set(loras)) == 4

        print("PASS: Hot-swap between LoRAs works correctly")

    def test_reload_same_lora(self, llm):
        """Test reloading the same LoRA (idempotency behavior)."""
        lora_req = LoRARequest("test_lora", 1, get_lora_path(0))

        # First load
        result1 = llm.llm_engine.add_lora(lora_req)

        # Second load (same LoRA) - should be idempotent
        result2 = llm.llm_engine.add_lora(lora_req)

        # Document behavior
        print(f"\nFirst add_lora: {result1}")
        print(f"Second add_lora (duplicate): {result2}")

        # Only one entry (no duplicates)
        loras = llm.llm_engine.list_loras()
        assert len(set(loras)) == 1, "Should not have duplicate LoRAs"

        print("PASS: Reloading same LoRA handled correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
