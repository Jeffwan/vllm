"""Test 9: Dynamic LoRA loading and unloading."""

import pytest
import time
import statistics
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from test_config import BASE_MODEL, get_lora_path, TEST_PROMPT, MAX_TOKENS


class TestDynamicLoading:
    """Test suite for dynamic LoRA loading/unloading."""

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

    def test_add_lora_api(self, llm):
        """Test add_lora API."""
        # Add LoRA
        lora_req = LoRARequest("test_lora", 1, get_lora_path(0))
        result = llm.llm_engine.add_lora(lora_req)
        assert result, "add_lora should return True"

        # Verify it's registered
        loras = llm.llm_engine.list_loras()
        assert 1 in loras

        # Use it
        sampling_params = SamplingParams(max_tokens=MAX_TOKENS)
        output = llm.generate(
            [TEST_PROMPT],
            sampling_params,
            lora_request=LoRARequest("test_lora", 1, get_lora_path(0))
        )
        assert output is not None

        print("PASS: add_lora API works correctly")

    def test_remove_lora_api(self, llm):
        """Test remove_lora API."""
        # Add LoRA
        lora_req = LoRARequest("test_lora", 1, get_lora_path(0))
        llm.llm_engine.add_lora(lora_req)

        # Remove it
        result = llm.llm_engine.remove_lora(1)
        assert result, "remove_lora should return True"

        # Verify it's gone
        loras = llm.llm_engine.list_loras()
        assert 1 not in loras

        print("PASS: remove_lora API works correctly")

    def test_list_loras_api(self, llm):
        """Test list_loras API."""
        # Initially empty
        loras = llm.llm_engine.list_loras()
        assert len(loras) == 0

        # Add some
        for i in range(3):
            lora_req = LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            llm.llm_engine.add_lora(lora_req)

        loras = llm.llm_engine.list_loras()
        # list_loras returns a list or set depending on vllm version
        assert set(loras) == {1, 2, 3}

        print("PASS: list_loras API works correctly")

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
        print(f"Mean: {statistics.mean(load_times)*1000:.2f} ms")
        print(f"Stdev: {statistics.stdev(load_times)*1000:.2f} ms")
        print(f"Min: {min(load_times)*1000:.2f} ms")
        print(f"Max: {max(load_times)*1000:.2f} ms")
        print(f"{'='*60}")

    def test_reload_same_lora(self, llm):
        """Test reloading the same LoRA."""
        lora_req = LoRARequest("test_lora", 1, get_lora_path(0))

        # First load
        result1 = llm.llm_engine.add_lora(lora_req)
        assert result1

        # Second load (same LoRA) - may return True (idempotent) or False
        # Just verify it doesn't throw an error
        result2 = llm.llm_engine.add_lora(lora_req)
        # Both True and False are valid - it's implementation dependent

        # Only one entry (no duplicates)
        loras = llm.llm_engine.list_loras()
        assert len(set(loras)) == 1, "Should not have duplicate LoRAs"

        print("PASS: Reloading same LoRA handled correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
