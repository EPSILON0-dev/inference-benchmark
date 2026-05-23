#!/usr/bin/env python3
"""
LLM Inference Speed Test Tool

A tool for testing LLM inference speed with parallel requests.
Supports OpenAI-compatible APIs (e.g., Ollama).
"""

import argparse
import json
import os
import random
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, List

import requests


@dataclass
class RequestResult:
    """Result of a single request."""
    token_count: int
    duration: float
    timestamp: float


@dataclass
class ThreadStats:
    """Statistics for a single worker thread."""
    results: List[RequestResult] = field(default_factory=list)
    first_start_time: Optional[float] = None
    last_end_time: Optional[float] = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def add_result(self, start_time: float, result: RequestResult):
        """Record a request result."""
        with self.lock:
            self.results.append(result)
            if self.first_start_time is None:
                self.first_start_time = start_time
            self.last_end_time = result.timestamp
    
    def get_stats(self) -> dict:
        """Calculate stats for this thread."""
        with self.lock:
            if not self.results:
                return {
                    "count": 0,
                    "total_tokens": 0,
                    "total_time": 0,
                    "tps": 0,
                    "tps_min": 0,
                    "tps_avg": 0,
                    "tps_max": 0
                }
            
            # Per-request TPS values
            request_tps_values = [
                r.token_count / r.duration if r.duration > 0 else 0
                for r in self.results
            ]
            
            total_tokens = sum(r.token_count for r in self.results)
            
            # Thread total time: from first request start to last request end
            thread_total_time = 0
            if self.first_start_time and self.last_end_time:
                thread_total_time = self.last_end_time - self.first_start_time
            
            # Thread TPS: total tokens / thread total time
            thread_tps = total_tokens / thread_total_time if thread_total_time > 0 else 0
            
            return {
                "count": len(self.results),
                "total_tokens": total_tokens,
                "total_time": thread_total_time,
                "tps": thread_tps,
                "tps_min": min(request_tps_values),
                "tps_avg": sum(request_tps_values) / len(request_tps_values),
                "tps_max": max(request_tps_values)
            }


@dataclass
class GlobalStats:
    """Global statistics across all threads."""
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    def mark_start(self):
        """Mark global start time."""
        with self.lock:
            if self.start_time is None:
                self.start_time = time.time()
    
    def mark_end(self):
        """Mark global end time."""
        with self.lock:
            self.end_time = time.time()
    
    def get_elapsed(self) -> float:
        """Get elapsed time."""
        with self.lock:
            if self.start_time is None:
                return 0
            end = self.end_time if self.end_time else time.time()
            return end - self.start_time


class InferenceSpeedTest:
    """Main class for running inference speed tests."""
    
    DEFAULT_PROMPT = """Write a comprehensive essay about the history and impact of artificial intelligence. 
Cover the early beginnings from the 1950s, through the expert systems of the 1980s, 
the machine learning revolution of the 2000s, to the modern era of large language models.
Discuss key figures like Turing, McCarthy, Hinton, and Bengio. 
Explain the technological breakthroughs, the winters and springs of AI development, 
and the societal impacts including both benefits and concerns. 
Be thorough and detailed in your response, providing specific examples and dates."""

    def __init__(self, base_url: str, endpoint: str, parallel: int, 
                 prompt: Optional[str], duration: int, model: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.endpoint = endpoint
        self.parallel = parallel
        self.duration = duration
        
        # Load prompts: use single prompt if provided, otherwise load and shuffle from file
        if prompt is not None:
            self.prompts = [prompt]
            self.use_random_prompts = False
        else:
            self.prompts = self._load_prompts()
            self.use_random_prompts = True
            # Shuffle once at startup
            random.shuffle(self.prompts)
        
        self.current_prompt_index = 0
        self.prompt_lock = threading.Lock()
        
        self.api_url = f"{self.base_url}/v1/{'chat/completions' if endpoint == 'chat' else 'completions'}"
        self.model = model if model else self._detect_model()
        
        self.global_stats = GlobalStats()
        self.thread_stats = [ThreadStats() for _ in range(parallel)]
        self.stop_event = threading.Event()
        self.soft_stop_event = threading.Event()
        self.threads = []
        self.stats_thread = None
        
        self._setup_signal_handlers()
    
    def _load_prompts(self) -> List[str]:
        """Load prompts from prompts.json file."""
        # Determine the path to prompts.json (same directory as this file)
        module_dir = os.path.dirname(os.path.abspath(__file__))
        prompts_path = os.path.join(module_dir, 'prompts.json')
        
        try:
            with open(prompts_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                prompts = data.get('prompts', [])
                if prompts:
                    return prompts
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load prompts.json ({e}), using default prompt.")
        
        # Fallback to default prompt
        return [self.DEFAULT_PROMPT]
    
    def _get_next_prompt(self) -> str:
        """Get the next prompt in the cycle (thread-safe)."""
        with self.prompt_lock:
            prompt = self.prompts[self.current_prompt_index]
            self.current_prompt_index = (self.current_prompt_index + 1) % len(self.prompts)
            return prompt
    
    def _detect_model(self) -> str:
        """Try to detect the model name from Ollama API."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("models", [])
                if models:
                    return models[0].get("name", "llama3.2")
        except Exception:
            pass
        return "llama3.2"  # Default fallback
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        self.interrupt_count = 0
        
        def signal_handler(signum, frame):
            self.interrupt_count += 1
            if self.interrupt_count == 1:
                print("\n[Soft stop requested] Finishing current requests (timeout: 30s)...")
                self.soft_stop_event.set()
            else:
                print("\n[Hard stop] Terminating immediately...")
                self.stop_event.set()
                sys.exit(1)
        
        signal.signal(signal.SIGINT, signal_handler)
    
    def _make_request(self, thread_id: int) -> Optional[RequestResult]:
        """Make a single API request and return result."""
        # Get the prompt to use for this request
        prompt = self._get_next_prompt()
        
        try:
            if self.endpoint == 'chat':
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False
                }
            else:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            
            start_time = time.time()
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=300
            )
            end_time = time.time()
            
            if response.status_code != 200:
                print(f"Thread {thread_id}: Error {response.status_code} - {response.text}")
                return None
            
            data = response.json()
            
            # Extract token count
            if 'usage' in data:
                token_count = data['usage'].get('completion_tokens', 0)
            else:
                token_count = 0
            
            # Fallback: estimate tokens from response content
            if token_count == 0:
                if 'choices' in data and len(data['choices']) > 0:
                    if self.endpoint == 'chat':
                        content = data['choices'][0].get('message', {}).get('content', '')
                    else:
                        content = data['choices'][0].get('text', '')
                    # Rough estimate: ~4 chars per token
                    token_count = max(1, len(content) // 4)
            
            duration = end_time - start_time
            
            return RequestResult(
                token_count=token_count,
                duration=duration,
                timestamp=end_time
            )
            
        except requests.Timeout:
            print(f"Thread {thread_id}: Request timeout")
            return None
        except Exception as e:
            print(f"Thread {thread_id}: Error - {e}")
            return None
    
    def _worker_thread(self, thread_id: int):
        """Worker thread that continuously makes requests."""
        stats = self.thread_stats[thread_id]
        
        while not self.stop_event.is_set() and not self.soft_stop_event.is_set():
            if self.global_stats.start_time and (time.time() - self.global_stats.start_time) >= self.duration:
                break
            
            start_time = time.time()
            result = self._make_request(thread_id)
            if result:
                stats.add_result(start_time, result)
    
    def _calculate_stats(self) -> dict:
        """Calculate all statistics."""
        all_thread_stats = [ts.get_stats() for ts in self.thread_stats]
        
        total_tokens = sum(s["total_tokens"] for s in all_thread_stats)
        total_requests = sum(s["count"] for s in all_thread_stats)
        
        # Calculate total TPS: sum of per-thread TPS (tokens/time for each thread)
        total_tps = sum(s["tps"] for s in all_thread_stats)
        
        # Global stats: min of mins, max of maxs, sum of avgs
        tps_mins = [s["tps_min"] for s in all_thread_stats if s["count"] > 0]
        tps_maxs = [s["tps_max"] for s in all_thread_stats if s["count"] > 0]
        tps_avgs = [s["tps_avg"] for s in all_thread_stats if s["count"] > 0]
        
        return {
            "elapsed": self.global_stats.get_elapsed(),
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "total_tps": total_tps,
            "global_tps_min": min(tps_mins) if tps_mins else 0,
            "global_tps_max": max(tps_maxs) if tps_maxs else 0,
            "global_tps_sum_avg": sum(tps_avgs) if tps_avgs else 0,
            "thread_stats": all_thread_stats
        }
    
    def _stats_printer(self):
        """Thread that prints stats every 10 seconds."""
        last_print = time.time()
        
        while not self.stop_event.is_set():
            time.sleep(0.1)
            
            if time.time() - last_print >= 10:
                stats = self._calculate_stats()
                print(f"[T+{stats['elapsed']:.1f}s] Completed: {stats['total_requests']} | "
                      f"TPS - min: {stats['global_tps_min']:.2f}, avg: {stats['global_tps_sum_avg']:.2f}, "
                      f"max: {stats['global_tps_max']:.2f} | Total: {stats['total_tps']:.2f}")
                last_print = time.time()
    
    def run(self):
        """Run the speed test."""
        print(f"Starting inference speed test")
        print(f"  Base URL: {self.base_url}")
        print(f"  Endpoint: {self.api_url}")
        print(f"  Model: {self.model}")
        print(f"  Parallel workers: {self.parallel}")
        print(f"  Duration: {self.duration}s")
        if self.use_random_prompts:
            print(f"  Prompts: {len(self.prompts)} randomized (shuffled once, cycling)")
        else:
            print(f"  Prompts: single custom prompt")
        print(f"  Press Ctrl+C once for soft stop (30s timeout), twice for hard stop\n")
        
        # Start global timer
        self.global_stats.mark_start()
        
        # Start worker threads
        for i in range(self.parallel):
            t = threading.Thread(target=self._worker_thread, args=(i,), daemon=True)
            t.start()
            self.threads.append(t)
        
        # Start stats printer
        self.stats_thread = threading.Thread(target=self._stats_printer, daemon=True)
        self.stats_thread.start()
        
        # Wait for duration or stop signal
        try:
            while not self.stop_event.is_set():
                elapsed = time.time() - self.global_stats.start_time
                if elapsed >= self.duration and not self.soft_stop_event.is_set():
                    print(f"\n[Duration reached] Stopping test...")
                    break
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        
        # Soft stop - wait for threads with timeout
        print("\nWaiting for threads to finish...")
        self.soft_stop_event.set()
        
        join_timeout = 30 if not self.stop_event.is_set() else 0
        for t in self.threads:
            t.join(timeout=join_timeout)
        
        # Mark end time
        self.global_stats.mark_end()
        
        # Final stats
        self._print_final_stats()
    
    def _print_final_stats(self):
        """Print final statistics."""
        stats = self._calculate_stats()
        
        print("\n" + "=" * 80)
        print("FINAL RESULTS")
        print("=" * 80)
        print(f"Total time: {stats['elapsed']:.2f}s")
        print(f"Total requests: {stats['total_requests']}")
        print(f"Total tokens: {stats['total_tokens']}")
        print(f"\nTotal TPS (sum of per-thread TPS): {stats['total_tps']:.2f}")
        print(f"\nGlobal TPS stats:")
        print(f"  Min of thread mins: {stats['global_tps_min']:.2f}")
        print(f"  Sum of thread avgs: {stats['global_tps_sum_avg']:.2f}")
        print(f"  Max of thread maxs: {stats['global_tps_max']:.2f}")
        
        # Per-thread stats
        print("\n" + "-" * 80)
        print("Per-thread statistics:")
        print("-" * 80)
        print(f"{'Thread':<8} {'Reqs':<8} {'Tokens':<10} {'Time(s)':<10} {'TPS':<10} "
              f"{'Min':<10} {'Avg':<10} {'Max':<10}")
        print("-" * 80)
        
        for i, s in enumerate(stats['thread_stats']):
            print(f"{i:<8} {s['count']:<8} {s['total_tokens']:<10} {s['total_time']:<10.2f} "
                  f"{s['tps']:<10.2f} {s['tps_min']:<10.2f} {s['tps_avg']:<10.2f} {s['tps_max']:<10.2f}")
        
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="LLM Inference Speed Test Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic test with default settings (1 thread, 60s)
  python -m src.inference_speed_test
  
  # Test with 4 parallel threads for 2 minutes
  python -m src.inference_speed_test --parallel 4 --duration 120
  
  # Test with custom prompt
  python -m src.inference_speed_test --prompt "Explain quantum computing"
  
  # Test specific model
  python -m src.inference_speed_test --model llama3.3 --parallel 4
        """
    )
    
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Base URL of the API server (default: http://localhost:11434)"
    )
    
    parser.add_argument(
        "--endpoint",
        choices=["response", "chat"],
        default="chat",
        help="API endpoint to use (default: chat)"
    )
    
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel generation threads (default: 1)"
    )
    
    parser.add_argument(
        "--prompt",
        default=None,
        help="Use a single custom prompt instead of randomized prompts from prompts.json"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Test duration in seconds (default: 60)"
    )
    
    parser.add_argument(
        "--model",
        default=None,
        help="Model name to use (default: auto-detect from Ollama)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.parallel < 1:
        parser.error("--parallel must be at least 1")
    if args.duration < 1:
        parser.error("--duration must be at least 1 second")
    
    # Run test
    test = InferenceSpeedTest(
        base_url=args.base_url,
        endpoint=args.endpoint,
        parallel=args.parallel,
        prompt=args.prompt,
        duration=args.duration,
        model=args.model
    )
    
    test.run()


if __name__ == "__main__":
    main()
