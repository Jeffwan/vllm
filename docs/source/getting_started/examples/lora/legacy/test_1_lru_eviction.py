"""Test 1: LRU eviction behavior for LoRA adapters."""

import pytest
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from test_config import BASE_MODEL, get_lora_path, TEST_PROMPT, MAX_TOKENS


class TestLRUEviction:
    """Test suite for LRU eviction behavior."""

    @pytest.fixture
    def llm_with_lora(self):
        """Create LLM with limited LoRA slots."""
        llm = LLM(
            model=BASE_MODEL,
            enable_lora=True,
            max_loras=2,          # Only 2 GPU slots
            max_cpu_loras=4,      # 4 CPU slots
            max_lora_rank=16,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            enforce_eager=True,   # Avoid CUDA graph issues with LoRA
        )
        yield llm
        del llm

    def test_basic_lru_eviction(self, llm_with_lora):
        """Test that oldest LoRA is evicted when capacity exceeded."""
        llm = llm_with_lora

        # Load 4 LoRAs (fills CPU cache)
        for i in range(4):
            lora_req = LoRARequest(
                lora_name=f"lora_{i}",
                lora_int_id=i + 1,
                lora_path=get_lora_path(i)
            )
            result = llm.llm_engine.add_lora(lora_req)
            assert result, f"Failed to add lora_{i}"

        # Verify all 4 are registered, return value is [1,2,3,4]
        registered = llm.llm_engine.list_loras()
        assert len(registered) == 4, f"Exptected 4 LoRAs, go {len(registered)}"

        # Access order: 1, 2, 3, 4 (4 is most recent)
        sampling_params = SamplingParams(max_tokens=MAX_TOKENS)
        for i in range(4):
            llm.generate(
                [TEST_PROMPT],
                sampling_params,
                lora_request=LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            )

        # Load 5th LoRA - should evict LoRA 1 (oldest)
        lora_req_5 = LoRARequest(
            lora_name="lora_4",
            lora_int_id=5,
            lora_path=get_lora_path(4)
        )
        llm.llm_engine.add_lora(lora_req_5)

        # Verify LoRA 1 was evicted
        registered = llm.llm_engine.list_loras()
        assert 1 not in registered, "LoRA 1 should have been evicted"
        assert 5 in registered, "LoRA 5 should be registered"
        assert len(registered) == 4, f"Expected 4 LoRAs, got {len(registered)}"

        print("PASS: Basic LRU eviction works correctly")

    def test_access_updates_lru_order(self, llm_with_lora):
        """Test that accessing a LoRA updates its LRU position."""
        llm = llm_with_lora
        sampling_params = SamplingParams(max_tokens=MAX_TOKENS)

        # Load 4 LoRAs
        for i in range(4):
            lora_req = LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            llm.llm_engine.add_lora(lora_req)

        # Access LoRA 1 again (makes it most recent)
        llm.generate(
            [TEST_PROMPT],
            sampling_params,
            lora_request=LoRARequest("lora_0", 1, get_lora_path(0))
        )

        # Load 5th LoRA - should evict LoRA 2 (now oldest)
        lora_req_5 = LoRARequest("lora_4", 5, get_lora_path(4))
        llm.llm_engine.add_lora(lora_req_5)

        # return [1,3,4,5], note: list_loras() won't give you the order of access but just the registered LoRAs
        registered = llm.llm_engine.list_loras()
        assert 1 in registered, "LoRA 1 should NOT be evicted (recently accessed)"
        assert 2 not in registered, "LoRA 2 should have been evicted (oldest)"

        print("PASS: Access updates LRU order correctly")

    def test_gpu_slot_eviction(self, llm_with_lora):
        """Test GPU slot eviction when activating new LoRA."""
        llm = llm_with_lora
        sampling_params = SamplingParams(max_tokens=MAX_TOKENS)

        # Load and use 3 LoRAs (only 2 GPU slots)
        for i in range(3):
            lora_req = LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            llm.llm_engine.add_lora(lora_req)
            llm.generate(
                [TEST_PROMPT],
                sampling_params,
                lora_request=LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            )

        # All 3 should be registered (CPU cache)
        registered = llm.llm_engine.list_loras()
        assert len(registered) == 3

        # But only 2 can be in GPU at once
        # This test verifies no errors occur with GPU slot eviction
        print("PASS: GPU slot eviction works without errors")

        # TODO: limitation, there's no way to distinguish the cpu lora or gpu lora.
        # for small rank size, it's not a concern but for large, that could be some issues.


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
