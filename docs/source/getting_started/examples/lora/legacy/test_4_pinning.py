"""Test 4: LoRA pinning functionality."""

import pytest
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from test_config import BASE_MODEL, get_lora_path, TEST_PROMPT, MAX_TOKENS


class TestPinning:
    """Test suite for LoRA pinning functionality."""

    @pytest.fixture
    def llm_pinning_test(self):
        """Create LLM for pinning tests."""
        llm = LLM(
            model=BASE_MODEL,
            enable_lora=True,
            max_loras=2,
            max_cpu_loras=3,  # Small cache to test eviction
            max_lora_rank=16,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            enforce_eager=True,
        )
        yield llm
        del llm

    def test_pinned_lora_not_evicted(self, llm_pinning_test):
        """Test that pinned LoRA survives eviction."""
        llm = llm_pinning_test

        # Load 3 LoRAs (fills CPU cache)
        # TODO: double check initial load, load to cpu, only request time it swapp to GPU?
        # TODO: double check overal volume is not cpu+gpu slots but max(gpu, cpu) slots?
        # what if i use --max-lora=4 --max-cpu-lora=2
        for i in range(3):
            lora_req = LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            llm.llm_engine.add_lora(lora_req)

        # Pin LoRA 1 (oldest)
        llm.llm_engine.pin_lora(1)

        # Load 4th LoRA - should evict LoRA 2 (oldest unpinned)
        lora_req_4 = LoRARequest("lora_3", 4, get_lora_path(3))
        llm.llm_engine.add_lora(lora_req_4)

        registered = llm.llm_engine.list_loras()
        assert 1 in registered, "Pinned LoRA 1 should NOT be evicted"
        assert 2 not in registered, "LoRA 2 (oldest unpinned) should be evicted"
        assert 4 in registered, "New LoRA 4 should be registered"

        print("PASS: Pinned LoRA survives eviction")

    def test_pin_activates_to_gpu(self, llm_pinning_test):
        """Test that pinning a CPU-only LoRA activates it to GPU."""
        llm = llm_pinning_test
        sampling_params = SamplingParams(max_tokens=MAX_TOKENS)

        # Load 3 LoRAs
        for i in range(3):
            lora_req = LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            llm.llm_engine.add_lora(lora_req)

        # Use LoRA 2 and 3 to fill GPU slots
        for i in [1, 2]:
            llm.generate(
                [TEST_PROMPT],
                sampling_params,
                lora_request=LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            )

        # LoRA 1 should be CPU-only now
        # Pin LoRA 1 - should activate it to GPU
        llm.llm_engine.pin_lora(1)

        # Generate with LoRA 1 should work without swap
        # TODO: edge case, what if lora 2 and 3 are pinned at this moment?
        output = llm.generate(
            [TEST_PROMPT],
            sampling_params,
            lora_request=LoRARequest("lora_0", 1, get_lora_path(0))
        )
        assert output is not None

        print("PASS: Pinning activates LoRA to GPU")

    def test_all_pinned_error(self, llm_pinning_test):
        """Test error when trying to evict but all are pinned."""
        llm = llm_pinning_test

        # Load 2 LoRAs first (GPU slots = 2)
        for i in range(2):
            lora_req = LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            llm.llm_engine.add_lora(lora_req)

        # Pin both GPU slots
        llm.llm_engine.pin_lora(1)
        llm.llm_engine.pin_lora(2)

        # Try to load 3rd LoRA - should fail because GPU slots are full and all are pinned
        # TODO: if can not register to CPU? why it failed?
        lora_req_3 = LoRARequest("lora_2", 3, get_lora_path(2))

        with pytest.raises(Exception, match="pinned"):
            llm.llm_engine.add_lora(lora_req_3)

        print("PASS: Error when all LoRAs are pinned and eviction needed")

# TODO:
# case 1: max loras=2, max-cpu-lora=3 and pin 2.  request cpu lora now. is it possible to swap cpu → gpu ? 
# case 2: make sure lora swapped to CPU. then is it possible to pin cpu one?  what would happen?  two cases. GPU slot is still available (exclude pins). GPU slot is not available.


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
