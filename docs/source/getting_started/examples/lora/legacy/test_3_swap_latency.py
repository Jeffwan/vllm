"""Test 3: CPU-to-GPU swap latency for LoRA adapters."""

import time
import statistics
import pytest
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from test_config import BASE_MODEL, get_lora_path, TEST_PROMPT, MAX_TOKENS


class TestSwapLatency:
    """Test suite for measuring LoRA swap latency."""

    @pytest.fixture
    def llm_swap_test(self):
        """Create LLM for swap testing."""
        llm = LLM(
            model=BASE_MODEL,
            enable_lora=True,
            max_loras=2,          # Small GPU cache
            max_cpu_loras=10,     # Large CPU cache
            max_lora_rank=16,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            enforce_eager=True,
        )
        yield llm
        del llm

    def measure_generation_time(self, llm, lora_request, num_runs=5):
        """Measure average generation time."""
        sampling_params = SamplingParams(max_tokens=MAX_TOKENS)
        times = []

        for _ in range(num_runs):
            start = time.perf_counter()
            llm.generate([TEST_PROMPT], sampling_params, lora_request=lora_request)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return {
            "mean": statistics.mean(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times),
            "times": times
        }

    def test_hot_vs_cold_latency(self, llm_swap_test):
        """Compare latency of GPU-resident vs CPU-cached LoRA."""
        llm = llm_swap_test

        # Load 4 LoRAs (2 will be GPU, 2 CPU-only)
        for i in range(4):
            lora_req = LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            llm.llm_engine.add_lora(lora_req)

        # Use LoRA 3 and 4 to ensure they're in GPU
        sampling_params = SamplingParams(max_tokens=MAX_TOKENS)
        for i in [2, 3]:  # lora_2, lora_3 (ids 3, 4)
            llm.generate(
                [TEST_PROMPT],
                sampling_params,
                lora_request=LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            )

        # Measure HOT path (LoRA 4 in GPU)
        hot_lora = LoRARequest("lora_3", 4, get_lora_path(3))
        hot_times = self.measure_generation_time(llm, hot_lora)

        # TODO: single request is not enough. we also need to measure performance using higher rank size
        # TODO in this test, we should test different lora ranks.
        # Measure COLD path (LoRA 1 in CPU, needs swap)
        cold_lora = LoRARequest("lora_0", 1, get_lora_path(0))
        cold_times = self.measure_generation_time(llm, cold_lora)

        # Calculate overhead
        swap_overhead = cold_times["mean"] - hot_times["mean"]

        print("\n" + "="*60)
        print("SWAP LATENCY RESULTS")
        print("="*60)
        print(f"HOT path (GPU-resident):")
        print(f"  Mean: {hot_times['mean']*1000:.2f} ms")
        print(f"  Stdev: {hot_times['stdev']*1000:.2f} ms")
        print(f"  Range: [{hot_times['min']*1000:.2f}, {hot_times['max']*1000:.2f}] ms")
        print(f"\nCOLD path (CPU->GPU swap):")
        print(f"  Mean: {cold_times['mean']*1000:.2f} ms")
        print(f"  Stdev: {cold_times['stdev']*1000:.2f} ms")
        print(f"  Range: [{cold_times['min']*1000:.2f}, {cold_times['max']*1000:.2f}] ms")
        print(f"\nSWAP OVERHEAD: {swap_overhead*1000:.2f} ms")
        if hot_times['mean'] > 0:
            print(f"Overhead %: {(swap_overhead/hot_times['mean'])*100:.1f}%")
        print("="*60)

        # Store results for reporting
        return {
            "hot": hot_times,
            "cold": cold_times,
            "swap_overhead_ms": swap_overhead * 1000
        }

    def test_repeated_swaps(self, llm_swap_test):
        """Test latency consistency across repeated swaps."""
        llm = llm_swap_test
        sampling_params = SamplingParams(max_tokens=MAX_TOKENS)

        # Load 3 LoRAs
        for i in range(3):
            lora_req = LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
            llm.llm_engine.add_lora(lora_req)

        # Alternate between LoRAs to force swaps
        # TODO: is this one reasonable? should we compare with 2 hot lora (always hit?)
        swap_times = []
        for iteration in range(50):
            for i in range(3):
                start = time.perf_counter()
                llm.generate(
                    [TEST_PROMPT],
                    sampling_params,
                    lora_request=LoRARequest(f"lora_{i}", i + 1, get_lora_path(i))
                )
                elapsed = time.perf_counter() - start
                # TODO: this is running + swap time, right? should we run same test against 2 hot loras and then use the avg time (swapped) - avg(hot) to see any differences?
                swap_times.append(elapsed)

        print("\n" + "="*60)
        print("REPEATED SWAP LATENCY")
        print("="*60)
        print(f"Total swaps: {len(swap_times)}")
        print(f"Mean: {statistics.mean(swap_times)*1000:.2f} ms")
        print(f"Stdev: {statistics.stdev(swap_times)*1000:.2f} ms")
        print(f"Min: {min(swap_times)*1000:.2f} ms")
        print(f"Max: {max(swap_times)*1000:.2f} ms")
        print("="*60)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
