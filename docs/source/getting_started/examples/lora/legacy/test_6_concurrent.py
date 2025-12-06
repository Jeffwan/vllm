"""Test 6: Concurrent request handling with LoRA."""

import pytest
import asyncio
import time
import random
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from test_config import BASE_MODEL, get_lora_path


class TestConcurrent:
    """Test suite for concurrent LoRA request handling."""

    @pytest.fixture
    def async_engine(self):
        """Create async engine for concurrent tests."""
        from vllm import AsyncLLMEngine, AsyncEngineArgs
        args = AsyncEngineArgs(
            model=BASE_MODEL,
            enable_lora=True,
            max_loras=4,
            max_cpu_loras=8,
            max_lora_rank=16,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            enforce_eager=True,
        )
        engine = AsyncLLMEngine.from_engine_args(args)
        yield engine

    @pytest.mark.asyncio
    async def test_high_concurrency(self, async_engine):
        """Test high concurrency with multiple LoRAs."""
        engine = async_engine

        # Load 8 LoRAs
        for i in range(8):
            lora_req = LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            await engine.add_lora(lora_req)

        sampling_params = SamplingParams(max_tokens=30)
        num_requests = 20

        async def random_lora_request(req_id):
            lora_id = random.randint(1, 8)
            results = []
            try:
                async for output in engine.generate(
                    f"Request {req_id}: Tell me about ",
                    sampling_params,
                    f"req_{req_id}",
                    lora_request=LoRARequest(f"lora_{lora_id-1}", lora_id, get_lora_path(lora_id-1))
                ):
                    results.append(output)
                return req_id, lora_id, results[-1] if results else None
            except Exception as e:
                print(f"Request {req_id} failed: {e}")
                return req_id, lora_id, None

        start = time.perf_counter()
        results = await asyncio.gather(*[
            random_lora_request(i) for i in range(num_requests)
        ])
        elapsed = time.perf_counter() - start

        # Verify all completed
        successful = sum(1 for r in results if r[2] is not None)

        print(f"\n{'='*60}")
        print(f"HIGH CONCURRENCY RESULTS")
        print(f"{'='*60}")
        print(f"Total requests: {num_requests}")
        print(f"Successful: {successful}")
        print(f"Failed: {num_requests - successful}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Throughput: {num_requests/elapsed:.2f} req/s")
        print(f"{'='*60}")

        assert successful == num_requests, \
            f"Some requests failed: {num_requests - successful}/{num_requests}"

    @pytest.mark.asyncio
    async def test_burst_traffic(self, async_engine):
        """Test burst traffic patterns."""
        engine = async_engine

        # Load 4 LoRAs
        for i in range(4):
            lora_req = LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            await engine.add_lora(lora_req)

        sampling_params = SamplingParams(max_tokens=20)

        async def single_request(burst_id, req_id, lora_id):
            """Send a single request for a specific LoRA."""
            results = []
            async for output in engine.generate(
                f"Burst {burst_id} req {req_id}: ",
                sampling_params,
                f"burst_{burst_id}_req_{req_id}",
                lora_request=LoRARequest(f"lora_{lora_id-1}", lora_id, get_lora_path(lora_id-1))
            ):
                results.append(output)
            return results[-1] if results else None

        async def burst_requests(burst_id, lora_id, count):
            """Send a burst of requests for a specific LoRA."""
            tasks = [single_request(burst_id, i, lora_id) for i in range(count)]
            return await asyncio.gather(*tasks)

        # Send bursts for different LoRAs
        start = time.perf_counter()
        all_results = await asyncio.gather(
            burst_requests(1, 1, 5),
            burst_requests(2, 2, 5),
            burst_requests(3, 3, 5),
            burst_requests(4, 4, 5),
        )
        elapsed = time.perf_counter() - start

        total = sum(len(r) for r in all_results)
        successful = sum(1 for r in all_results for x in r if x is not None)

        print(f"\nBurst traffic: 4 bursts x 5 requests = 20 total")
        print(f"Completed in {elapsed:.2f}s")
        print(f"Success rate: {successful}/{total}")

        assert successful == total


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
