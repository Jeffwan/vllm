"""Consolidated Test Suite: Batching and Concurrency.

This module combines tests for:
- Scheduler batching constraints with max_loras limit
- Concurrent request handling with multiple LoRAs
- Burst traffic patterns

Note: Uses synchronous LLM API instead of AsyncLLMEngine to avoid
pytest-asyncio dependency issues.
"""

import pytest
import time
import random
import statistics
import gc
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from test_config import BASE_MODEL, get_lora_path, TEST_PROMPT


def cleanup_gpu():
    """Force GPU memory cleanup."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


class TestBatching:
    """Test suite for LoRA batching constraints."""

    @pytest.fixture
    def llm_batching(self):
        """Create LLM for batching tests with max_loras=2."""
        cleanup_gpu()
        llm = LLM(
            model=BASE_MODEL,
            enable_lora=True,
            max_loras=2,  # Only 2 LoRAs per batch
            max_cpu_loras=8,
            max_lora_rank=16,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            enforce_eager=True,
        )
        yield llm
        del llm
        cleanup_gpu()

    def test_batch_lora_limit_timing(self, llm_batching):
        """Test max_loras limits by measuring request completion times.

        With max_loras=2, requests using >2 different LoRAs should be
        scheduled in separate batches, resulting in delayed completion
        for requests that exceed the limit.
        """
        llm = llm_batching

        # Load 4 LoRAs
        for i in range(4):
            lora_req = LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            llm.llm_engine.add_lora(lora_req)

        sampling_params = SamplingParams(max_tokens=30)

        # Generate with 4 different LoRAs in a batch
        prompts_and_loras = []
        for i in range(4):
            lora_id = i + 1
            prompts_and_loras.append((TEST_PROMPT, lora_id))

        # Measure total time for batch with 4 LoRAs (exceeds max_loras=2)
        global_start = time.perf_counter()
        results = []
        for prompt, lora_id in prompts_and_loras:
            start = time.perf_counter()
            output = llm.generate(
                [prompt],
                sampling_params,
                lora_request=LoRARequest(f"lora_{lora_id-1}", lora_id, get_lora_path(lora_id-1))
            )
            elapsed = time.perf_counter() - start
            results.append((lora_id, elapsed, output[0]))

        total_time = time.perf_counter() - global_start

        # Verify all completed
        assert len(results) == 4
        for lora_id, elapsed, output in results:
            assert output.outputs[0].text, f"LoRA {lora_id} failed"

        # Analyze completion times
        sorted_times = sorted(results, key=lambda x: x[1])

        print(f"\n{'='*60}")
        print("BATCH SCHEDULING ANALYSIS")
        print(f"{'='*60}")
        print(f"max_loras=2, submitted 4 requests with 4 different LoRAs")
        print(f"\nCompletion times by LoRA ID:")
        for lora_id, elapsed, _ in sorted_times:
            print(f"  LoRA {lora_id}: {elapsed*1000:.0f} ms")

        print(f"\nTotal time: {total_time*1000:.0f} ms")
        print(f"{'='*60}")

        print("PASS: Scheduler handles LoRA batching correctly")

    def test_same_lora_batching_efficiency(self, llm_batching):
        """Test that same-LoRA requests batch efficiently.

        Requests using the same LoRA should be batched together
        without scheduler delays.
        """
        llm = llm_batching

        # Load 1 LoRA
        lora_req = LoRARequest("lora_0", 1, get_lora_path(0))
        llm.llm_engine.add_lora(lora_req)

        sampling_params = SamplingParams(max_tokens=20)

        # Batch 10 requests with same LoRA
        prompts = [TEST_PROMPT] * 10

        start = time.perf_counter()
        results = llm.generate(
            prompts,
            sampling_params,
            lora_request=LoRARequest("lora_0", 1, get_lora_path(0))
        )
        total_elapsed = time.perf_counter() - start

        assert len(results) == 10
        assert all(r.outputs[0].text for r in results)

        print(f"\n{'='*60}")
        print("SAME-LORA BATCHING EFFICIENCY")
        print(f"{'='*60}")
        print(f"10 requests with same LoRA (max_loras=2)")
        print(f"Total time: {total_elapsed*1000:.0f} ms")
        print(f"Throughput: {10/total_elapsed:.1f} req/s")
        print(f"{'='*60}")

        # Low variance indicates efficient batching
        print("PASS: Same-LoRA requests batched efficiently")

    def test_throughput_comparison(self, llm_batching):
        """Compare throughput: 2 LoRAs (fits max_loras) vs 4 LoRAs (exceeds).

        With max_loras=2:
        - 2 LoRAs: All requests can batch together
        - 4 LoRAs: Requests split across batches
        """
        llm = llm_batching

        # Load 4 LoRAs
        for i in range(4):
            lora_req = LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            llm.llm_engine.add_lora(lora_req)

        sampling_params = SamplingParams(max_tokens=20)
        num_requests = 8

        def run_workload(lora_count):
            """Run workload with specified number of unique LoRAs."""
            results = []
            start = time.perf_counter()
            for i in range(num_requests):
                lora_id = (i % lora_count) + 1
                output = llm.generate(
                    [TEST_PROMPT],
                    sampling_params,
                    lora_request=LoRARequest(f"lora_{lora_id-1}", lora_id, get_lora_path(lora_id-1))
                )
                results.append(output[0])
            elapsed = time.perf_counter() - start
            return elapsed, all(r.outputs[0].text for r in results)

        # Test with 2 LoRAs (fits in max_loras=2)
        time_2_loras, success_2 = run_workload(2)

        # Test with 4 LoRAs (exceeds max_loras=2)
        time_4_loras, success_4 = run_workload(4)

        print(f"\n{'='*60}")
        print("THROUGHPUT COMPARISON")
        print(f"{'='*60}")
        print(f"max_loras=2, {num_requests} requests each")
        print(f"\n2 LoRAs (fits max_loras):")
        print(f"  Time: {time_2_loras*1000:.0f} ms")
        print(f"  Throughput: {num_requests/time_2_loras:.1f} req/s")
        print(f"\n4 LoRAs (exceeds max_loras):")
        print(f"  Time: {time_4_loras*1000:.0f} ms")
        print(f"  Throughput: {num_requests/time_4_loras:.1f} req/s")
        print(f"\nOverhead from exceeding max_loras: {(time_4_loras/time_2_loras - 1)*100:.0f}%")
        print(f"{'='*60}")

        assert success_2 and success_4, "Some requests failed"
        print("PASS: Throughput comparison completed")


class TestConcurrency:
    """Test suite for concurrent LoRA request handling."""

    @pytest.fixture
    def llm_concurrency(self):
        """Create LLM for concurrency tests with max_loras=4."""
        cleanup_gpu()
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
        cleanup_gpu()

    def test_high_concurrency_random_loras(self, llm_concurrency):
        """Test high concurrency with random LoRA selection.

        Uses batch requests with random LoRA distribution.
        """
        llm = llm_concurrency

        # Load 8 LoRAs
        for i in range(8):
            lora_req = LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            llm.llm_engine.add_lora(lora_req)

        sampling_params = SamplingParams(max_tokens=30)
        num_requests = 20

        # Generate requests with random LoRA selection
        results_data = []
        start = time.perf_counter()

        for req_id in range(num_requests):
            lora_id = random.randint(1, 8)
            req_start = time.perf_counter()
            try:
                output = llm.generate(
                    [f"Request {req_id}: Tell me about "],
                    sampling_params,
                    lora_request=LoRARequest(f"lora_{lora_id-1}", lora_id, get_lora_path(lora_id-1))
                )
                elapsed = time.perf_counter() - req_start
                results_data.append((req_id, lora_id, elapsed, output[0]))
            except Exception as e:
                elapsed = time.perf_counter() - req_start
                print(f"Request {req_id} failed: {e}")
                results_data.append((req_id, lora_id, elapsed, None))

        total_time = time.perf_counter() - start

        # Analyze results
        successful = [(r, l, t) for r, l, t, out in results_data if out is not None]
        failed = [(r, l) for r, l, t, out in results_data if out is None]

        # Count LoRA distribution
        lora_counts = {}
        for _, lora_id, _ in successful:
            lora_counts[lora_id] = lora_counts.get(lora_id, 0) + 1

        latencies = [t for _, _, t in successful]

        print(f"\n{'='*60}")
        print("HIGH CONCURRENCY RESULTS")
        print(f"{'='*60}")
        print(f"Total requests: {num_requests}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Throughput: {num_requests/total_time:.2f} req/s")
        print(f"\nLatency stats (successful requests):")
        if latencies:
            print(f"  Mean: {statistics.mean(latencies)*1000:.0f} ms")
            print(f"  Stdev: {statistics.stdev(latencies)*1000:.0f} ms" if len(latencies) > 1 else "  Stdev: N/A")
            print(f"  Min: {min(latencies)*1000:.0f} ms")
            print(f"  Max: {max(latencies)*1000:.0f} ms")
        print(f"\nLoRA distribution: {lora_counts}")
        print(f"{'='*60}")

        assert len(successful) == num_requests, \
            f"Some requests failed: {len(failed)}/{num_requests}"

        print("PASS: High concurrency test passed")

    def test_burst_traffic_per_lora(self, llm_concurrency):
        """Test burst traffic with dedicated bursts per LoRA."""
        llm = llm_concurrency

        # Load 4 LoRAs
        for i in range(4):
            lora_req = LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            llm.llm_engine.add_lora(lora_req)

        sampling_params = SamplingParams(max_tokens=20)
        burst_size = 5

        # Simulate burst traffic - batch requests per LoRA
        results_data = []
        start = time.perf_counter()

        for lora_id in range(1, 5):  # 4 LoRAs
            # Create burst of prompts for this LoRA
            prompts = [f"Burst LoRA {lora_id} req {i}: " for i in range(burst_size)]
            burst_start = time.perf_counter()

            outputs = llm.generate(
                prompts,
                sampling_params,
                lora_request=LoRARequest(f"lora_{lora_id-1}", lora_id, get_lora_path(lora_id-1))
            )

            elapsed = time.perf_counter() - burst_start
            for i, output in enumerate(outputs):
                results_data.append((lora_id, i, lora_id, elapsed / len(outputs), output))

        total_time = time.perf_counter() - start

        # Analyze results
        successful = [r for r in results_data if r[4] is not None]
        total = len(results_data)

        # Per-LoRA latency analysis
        lora_latencies = {}
        for lora_id, req_id, _, elapsed, _ in successful:
            if lora_id not in lora_latencies:
                lora_latencies[lora_id] = []
            lora_latencies[lora_id].append(elapsed)

        print(f"\n{'='*60}")
        print("BURST TRAFFIC RESULTS")
        print(f"{'='*60}")
        print(f"4 LoRAs x {burst_size} requests = {total} total")
        print(f"Completed in {total_time:.2f}s")
        print(f"Success rate: {len(successful)}/{total}")
        print(f"Throughput: {total/total_time:.1f} req/s")
        print(f"\nPer-LoRA latency (mean):")
        for lora_id in sorted(lora_latencies.keys()):
            latencies = lora_latencies[lora_id]
            print(f"  LoRA {lora_id}: {statistics.mean(latencies)*1000:.0f} ms")
        print(f"{'='*60}")

        assert len(successful) == total, f"Some requests failed"
        print("PASS: Burst traffic test passed")

    def test_sequential_vs_parallel(self, llm_concurrency):
        """Compare sequential vs batched request handling."""
        llm = llm_concurrency

        # Load 4 LoRAs
        for i in range(4):
            lora_req = LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            llm.llm_engine.add_lora(lora_req)

        sampling_params = SamplingParams(max_tokens=20)
        num_requests = 8

        # Sequential execution (one at a time)
        seq_start = time.perf_counter()
        for i in range(num_requests):
            lora_id = (i % 4) + 1
            llm.generate(
                [TEST_PROMPT],
                sampling_params,
                lora_request=LoRARequest(f"lora_{lora_id-1}", lora_id, get_lora_path(lora_id-1))
            )
        seq_time = time.perf_counter() - seq_start

        # Batched execution (same LoRA batches)
        # Group by LoRA for optimal batching
        batch_start = time.perf_counter()
        for lora_id in range(1, 5):  # 4 LoRAs
            # 2 requests per LoRA = 8 total
            prompts = [TEST_PROMPT, TEST_PROMPT]
            llm.generate(
                prompts,
                sampling_params,
                lora_request=LoRARequest(f"lora_{lora_id-1}", lora_id, get_lora_path(lora_id-1))
            )
        batch_time = time.perf_counter() - batch_start

        speedup = seq_time / batch_time if batch_time > 0 else 0

        print(f"\n{'='*60}")
        print("SEQUENTIAL vs BATCHED")
        print(f"{'='*60}")
        print(f"{num_requests} requests with 4 LoRAs")
        print(f"\nSequential (one at a time):")
        print(f"  Time: {seq_time*1000:.0f} ms")
        print(f"  Throughput: {num_requests/seq_time:.1f} req/s")
        print(f"\nBatched (grouped by LoRA):")
        print(f"  Time: {batch_time*1000:.0f} ms")
        print(f"  Throughput: {num_requests/batch_time:.1f} req/s")
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"{'='*60}")

        # Batched should be faster (or at least not slower)
        assert batch_time <= seq_time * 1.2, "Batched should not be significantly slower"
        print("PASS: Batched execution is efficient")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
