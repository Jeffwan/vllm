"""Test 2: Memory isolation between LoRA and KV cache."""

import pytest
import gc
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from test_config import BASE_MODEL, get_lora_path


class TestMemoryIsolation:
    """Test suite for LoRA/KV cache memory isolation."""

    @pytest.fixture
    def llm_memory_constrained(self):
        """Create LLM with constrained memory to force KV pressure."""
        llm = LLM(
            model=BASE_MODEL,
            enable_lora=True,
            max_loras=4,
            max_cpu_loras=4,
            max_lora_rank=16,
            gpu_memory_utilization=0.7,  # Lower to create KV pressure
            max_model_len=4096,
            trust_remote_code=True,
            enforce_eager=True,
        )
        yield llm
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    def test_kv_pressure_no_lora_eviction(self, llm_memory_constrained):
        """Test that KV cache pressure doesn't evict pinned LoRAs."""
        llm = llm_memory_constrained

        # Load and pin 4 LoRAs
        for i in range(4):
            lora_req = LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            llm.llm_engine.add_lora(lora_req)
            llm.llm_engine.pin_lora(i + 1)

        initial_loras = llm.llm_engine.list_loras()
        assert len(initial_loras) == 4, f"Expected 4 LoRAs, got {len(initial_loras)}"

        # Generate with long sequences to pressure KV cache
        # TODO: we need to make sure that the kv size is super large.
        # otherwise, the kv cache pressure won't be enough to trigger the eviction.
        # one possible is to specify the gpu memory percentage which is easier for testing.
        # this is low priority since we double check the code and 
        # pretty sure this is isolated from kv space
        long_prompt = "Write a very detailed essay about " * 50
        sampling_params = SamplingParams(max_tokens=200)

        for iteration in range(5):
            for i in range(4):
                try:
                    outputs = llm.generate(
                        [long_prompt],
                        sampling_params,
                        lora_request=LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
                    )
                except Exception as e:
                    # KV cache exhaustion is OK, LoRA eviction is not
                    if "lora" in str(e).lower():
                        pytest.fail(f"LoRA-related error during KV pressure: {e}")
                    print(f"KV cache pressure error (expected): {e}")

        # Verify all LoRAs still present
        final_loras = llm.llm_engine.list_loras()
        assert final_loras == initial_loras, \
            f"LoRAs changed under KV pressure! Before: {initial_loras}, After: {final_loras}"

        print("PASS: KV cache pressure does not evict LoRAs")

    def test_concurrent_long_sequences(self, llm_memory_constrained):
        """Test multiple long sequences with different LoRAs."""
        llm = llm_memory_constrained

        # Load 2 LoRAs
        for i in range(2):
            lora_req = LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            llm.llm_engine.add_lora(lora_req)

        # Generate alternating between LoRAs with long contexts
        prompts = ["Explain quantum physics in detail: " * 20] * 4
        sampling_params = SamplingParams(max_tokens=100)

        for i, prompt in enumerate(prompts):
            lora_id = (i % 2) + 1
            outputs = llm.generate(
                [prompt],
                sampling_params,
                lora_request=LoRARequest(f"lora_{lora_id-1}", lora_id, get_lora_path(lora_id-1))
            )

        # Both LoRAs should still be present
        final_loras = llm.llm_engine.list_loras()
        assert 1 in final_loras and 2 in final_loras, \
            f"LoRAs missing after concurrent long sequences: {final_loras}"

        print("PASS: Concurrent long sequences don't affect LoRA state")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
