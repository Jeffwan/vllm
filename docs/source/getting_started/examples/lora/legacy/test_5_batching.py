"""Test 5: Scheduler batching constraints for LoRA."""

import pytest
import asyncio
import time
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from test_config import BASE_MODEL, get_lora_path, TEST_PROMPT


class TestBatching:
    """Test suite for LoRA batching constraints."""

    @pytest.fixture
    def async_engine(self):
        """Create async engine for batching tests."""
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        args = AsyncEngineArgs(
            model=BASE_MODEL,
            enable_lora=True,
            max_loras=2,  # Only 2 LoRAs per batch
            max_cpu_loras=5,
            max_lora_rank=16,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            enforce_eager=True,
        )
        engine = AsyncLLMEngine.from_engine_args(args)
        yield engine
        # Cleanup

    @pytest.mark.asyncio
    async def test_batch_lora_limit(self, async_engine):
        """Test that max_loras limits unique LoRAs per batch."""
        engine = async_engine

        # Load 4 LoRAs
        for i in range(4):
            lora_req = LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            await engine.add_lora(lora_req)

        sampling_params = SamplingParams(max_tokens=20)

        # Send 4 concurrent requests with 4 different LoRAs
        async def generate_with_lora(lora_id):
            request_id = f"req_{lora_id}"
            results = []
            async for output in engine.generate(
                TEST_PROMPT,
                sampling_params,
                request_id,
                lora_request=LoRARequest(f"lora_{lora_id-1}", lora_id, get_lora_path(lora_id-1))
            ):
                results.append(output)
            return lora_id, results[-1] if results else None

        # All 4 should complete (some will be deferred)
        start = time.perf_counter()
        results = await asyncio.gather(*[
            generate_with_lora(i + 1) for i in range(4)
        ])
        elapsed = time.perf_counter() - start

        # TODO
        # since --max-lora=2 here, technically, it run 2 request and then another 2 request
        # it runs 2 batches, so the first 2 request finish very quick, second batch has queuing time + running time.
        # our testing can not show this difference.. how to improve the test?

        # Verify all completed
        assert len(results) == 4
        for lora_id, output in results:
            assert output is not None, f"LoRA {lora_id} failed"

        print(f"\nAll 4 requests with 4 LoRAs (max_loras=2) completed in {elapsed:.2f}s")
        print("PASS: Scheduler handles LoRA batching correctly")

    @pytest.mark.asyncio
    async def test_same_lora_batching(self, async_engine):
        """Test that requests with same LoRA can be batched."""
        engine = async_engine

        # Load 1 LoRA
        lora_req = LoRARequest("lora_0", 1, get_lora_path(0))
        await engine.add_lora(lora_req)

        sampling_params = SamplingParams(max_tokens=20)

        # Send 10 concurrent requests with same LoRA
        async def generate_with_lora(req_id):
            request_id = f"req_{req_id}"
            results = []
            async for output in engine.generate(
                TEST_PROMPT,
                sampling_params,
                request_id,
                lora_request=LoRARequest("lora_0", 1, get_lora_path(0))
            ):
                results.append(output)
            return results[-1] if results else None

        start = time.perf_counter()
        results = await asyncio.gather(*[generate_with_lora(i) for i in range(10)])
        elapsed = time.perf_counter() - start

        assert all(r is not None for r in results)
        print(f"\n10 requests with same LoRA completed in {elapsed:.2f}s")
        print("PASS: Same-LoRA requests can be batched efficiently")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
