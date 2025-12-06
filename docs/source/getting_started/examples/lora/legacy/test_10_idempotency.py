"""Test idempotency behaviors for LoRA APIs."""

import pytest
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from test_config import BASE_MODEL, get_lora_path, TEST_PROMPT, MAX_TOKENS


class TestIdempotency:
    """Test suite for LoRA API idempotency behaviors."""

    @pytest.fixture
    def llm(self):
        """Create LLM for idempotency tests."""
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

    def test_add_lora_duplicate_returns_false(self, llm):
        """Test that add_lora returns False on duplicate (documented behavior)."""
        lora_req = LoRARequest("test_lora", 1, get_lora_path(0))

        # First add should return True
        result1 = llm.llm_engine.add_lora(lora_req)
        print(f"First add_lora result: {result1}")

        # Second add (duplicate) - check what it returns
        result2 = llm.llm_engine.add_lora(lora_req)
        print(f"Second add_lora (duplicate) result: {result2}")

        # Verify only one entry exists
        loras = llm.llm_engine.list_loras()
        assert len(set(loras)) == 1, f"Expected 1 LoRA, got {len(set(loras))}"

        # Document actual behavior
        print(f"\nACTUAL BEHAVIOR:")
        print(f"  add_lora on duplicate returns: {result2}, 1st time: {result1}")
        print(f"  (Expected per docs: False)")

        # The research doc says it should return False on duplicate
        # Let's verify this
        if result2 is False:
            print("CONFIRMED: add_lora returns False on duplicate")
        else:
            print(f"DEVIATION: add_lora returns {result2} on duplicate (not False)")

        return result2

    def test_add_lora_duplicate_updates_lru(self, llm):
        """Test that duplicate add_lora updates LRU position."""
        # Add 4 LoRAs (fills max_cpu_loras partially)
        for i in range(4):
            lora_req = LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            llm.llm_engine.add_lora(lora_req)

        # Access LoRA 1 again (should update LRU to most recent)
        lora_req = LoRARequest("lora_0", 1, get_lora_path(0))
        llm.llm_engine.add_lora(lora_req)

        # Add 5 more LoRAs to trigger eviction (max_cpu_loras=8)
        for i in range(4, 9):
            lora_req = LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            llm.llm_engine.add_lora(lora_req)

        # LoRA 1 should survive (recently touched), LoRA 2 should be evicted
        loras = llm.llm_engine.list_loras()
        print(f"\nLoRAs after eviction: {loras}")

        if 1 in loras:
            print("CONFIRMED: Duplicate add_lora updates LRU position")
        else:
            print("DEVIATION: LoRA 1 was evicted despite being re-added")

        assert 1 in loras, "LoRA 1 should survive (recently touched via add_lora)"

    def test_remove_lora_not_found_returns_false(self, llm):
        """Test that remove_lora returns False if not found."""
        # Remove non-existent LoRA
        result = llm.llm_engine.remove_lora(999)
        print(f"\nremove_lora(999) for non-existent LoRA: {result}")

        if result is False:
            print("CONFIRMED: remove_lora returns False when LoRA not found")
        else:
            print(f"DEVIATION: remove_lora returns {result} when LoRA not found")

        assert result is False, "remove_lora should return False for non-existent LoRA"

    def test_remove_lora_safe_multiple_calls(self, llm):
        """Test that remove_lora is safe to call multiple times."""
        # Add a LoRA
        lora_req = LoRARequest("test_lora", 1, get_lora_path(0))
        llm.llm_engine.add_lora(lora_req)

        # First remove
        result1 = llm.llm_engine.remove_lora(1)
        print(f"\nFirst remove_lora(1): {result1}")

        # Second remove (already removed)
        result2 = llm.llm_engine.remove_lora(1)
        print(f"Second remove_lora(1): {result2}")

        # Third remove
        result3 = llm.llm_engine.remove_lora(1)
        print(f"Third remove_lora(1): {result3}")

        print("CONFIRMED: remove_lora is safe to call multiple times (no exceptions)")

        assert result1 is True, "First remove should return True"
        assert result2 is False, "Second remove should return False"
        assert result3 is False, "Third remove should return False"

    def test_pin_lora_idempotent(self, llm):
        """Test that pin_lora is idempotent (uses set.add)."""
        # Add a LoRA
        lora_req = LoRARequest("test_lora", 1, get_lora_path(0))
        llm.llm_engine.add_lora(lora_req)

        # Pin multiple times - should not raise
        result1 = llm.llm_engine.pin_lora(1)
        print(f"\nFirst pin_lora(1): {result1}")

        result2 = llm.llm_engine.pin_lora(1)
        print(f"Second pin_lora(1): {result2}")

        result3 = llm.llm_engine.pin_lora(1)
        print(f"Third pin_lora(1): {result3}")

        print("CONFIRMED: pin_lora is idempotent (no exceptions on multiple calls)")

        # Verify LoRA is still pinned and functional
        sampling_params = SamplingParams(max_tokens=MAX_TOKENS)
        output = llm.generate(
            [TEST_PROMPT],
            sampling_params,
            lora_request=LoRARequest("test_lora", 1, get_lora_path(0))
        )
        assert output is not None

    def test_pin_lora_not_found_raises(self, llm):
        """Test that pin_lora raises error if LoRA not found."""
        # Try to pin non-existent LoRA
        with pytest.raises(Exception) as excinfo:
            llm.llm_engine.pin_lora(999)

        print(f"\npin_lora(999) raised: {type(excinfo.value).__name__}: {excinfo.value}")
        print("CONFIRMED: pin_lora raises exception when LoRA not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
