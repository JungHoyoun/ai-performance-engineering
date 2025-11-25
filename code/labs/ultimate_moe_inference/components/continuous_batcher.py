"""Continuous Batching for dynamic request handling (Ch17).

Implements iteration-level batching where requests can join
and leave the batch at any decode step.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import time

import torch


class RequestState(Enum):
    """State of a request in the batcher."""
    PENDING = "pending"
    PREFILLING = "prefilling"
    DECODING = "decoding"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class Request:
    """A single inference request."""
    
    request_id: str
    prompt: str
    max_tokens: int
    
    # State
    state: RequestState = RequestState.PENDING
    
    # Tokens
    input_ids: Optional[torch.Tensor] = None
    generated_ids: List[int] = field(default_factory=list)
    
    # Timing
    arrival_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    end_time: Optional[float] = None
    
    @property
    def output_tokens(self) -> int:
        """Number of generated tokens."""
        return len(self.generated_ids)
    
    @property
    def is_done(self) -> bool:
        """Check if request is complete."""
        return (
            self.state == RequestState.COMPLETED or
            self.state == RequestState.CANCELLED or
            self.output_tokens >= self.max_tokens
        )
    
    @property
    def ttft_ms(self) -> Optional[float]:
        """Time to first token in milliseconds."""
        if self.start_time and self.first_token_time:
            return (self.first_token_time - self.start_time) * 1000
        return None
    
    @property
    def latency_ms(self) -> Optional[float]:
        """Total latency in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return None


@dataclass
class BatchState:
    """Current state of the continuous batch."""
    
    active_requests: List[Request]
    total_tokens: int
    max_batch_size: int
    
    @property
    def batch_size(self) -> int:
        """Current batch size."""
        return len(self.active_requests)
    
    @property
    def is_full(self) -> bool:
        """Check if batch is at capacity."""
        return self.batch_size >= self.max_batch_size


class ContinuousBatcher:
    """Continuous batching engine for dynamic request handling.
    
    Unlike static batching, continuous batching allows requests to:
    - Join the batch at any decode step
    - Leave when complete (free up capacity for new requests)
    - Run with different sequence lengths
    
    This maximizes GPU utilization by keeping the batch full.
    
    Example:
        batcher = ContinuousBatcher(model, tokenizer, max_batch_size=32)
        
        # Add requests (can be done asynchronously)
        await batcher.add_request("Hello, how are you?", max_tokens=100)
        await batcher.add_request("What is AI?", max_tokens=50)
        
        # Run continuous batching
        async for batch_result in batcher.run():
            print(f"Completed: {batch_result.completed_requests}")
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        max_batch_size: int = 32,
        max_seq_len: int = 4096,
    ):
        """Initialize continuous batcher.
        
        Args:
            model: Language model
            tokenizer: Tokenizer
            max_batch_size: Maximum concurrent requests
            max_seq_len: Maximum sequence length
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        
        # Request queues
        self.pending_queue: asyncio.Queue[Request] = asyncio.Queue()
        self.active_requests: List[Request] = []
        self.completed_requests: List[Request] = []
        
        # Statistics
        self.total_requests = 0
        self.total_tokens_generated = 0
        
        # Running state
        self._running = False
    
    async def add_request(
        self,
        prompt: str,
        max_tokens: int = 256,
        request_id: Optional[str] = None,
    ) -> str:
        """Add a request to the pending queue.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            request_id: Optional request ID
            
        Returns:
            Request ID
        """
        if request_id is None:
            request_id = f"req_{self.total_requests}"
        
        request = Request(
            request_id=request_id,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        
        await self.pending_queue.put(request)
        self.total_requests += 1
        
        return request_id
    
    def _admit_requests(self) -> int:
        """Admit pending requests up to batch capacity.
        
        Returns:
            Number of requests admitted
        """
        admitted = 0
        
        while len(self.active_requests) < self.max_batch_size:
            try:
                request = self.pending_queue.get_nowait()
                
                # Tokenize prompt
                encoding = self.tokenizer(
                    request.prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_seq_len,
                )
                request.input_ids = encoding["input_ids"].to(self.model.device)
                request.state = RequestState.PREFILLING
                request.start_time = time.time()
                
                self.active_requests.append(request)
                admitted += 1
                
            except asyncio.QueueEmpty:
                break
        
        return admitted
    
    def _run_prefill(self) -> None:
        """Run prefill for newly admitted requests."""
        prefilling = [r for r in self.active_requests if r.state == RequestState.PREFILLING]
        
        if not prefilling:
            return
        
        # Batch prefill
        for request in prefilling:
            with torch.no_grad():
                # Run prefill (just forward pass, no generation)
                _ = self.model(request.input_ids, use_cache=True)
            
            request.state = RequestState.DECODING
    
    def _run_decode_step(self) -> List[Request]:
        """Run one decode step for all active requests.
        
        Returns:
            List of newly completed requests
        """
        decoding = [r for r in self.active_requests if r.state == RequestState.DECODING]
        
        if not decoding:
            return []
        
        completed = []
        
        for request in decoding:
            # Get last token for each request
            if request.generated_ids:
                last_token = torch.tensor([[request.generated_ids[-1]]], device=self.model.device)
            else:
                last_token = request.input_ids[:, -1:]
            
            # Generate next token
            with torch.no_grad():
                outputs = self.model(last_token, use_cache=True)
                next_token = outputs.logits[:, -1].argmax(dim=-1).item()
            
            # Record first token time
            if not request.generated_ids:
                request.first_token_time = time.time()
            
            request.generated_ids.append(next_token)
            self.total_tokens_generated += 1
            
            # Check for completion
            if request.is_done or next_token == self.tokenizer.eos_token_id:
                request.state = RequestState.COMPLETED
                request.end_time = time.time()
                completed.append(request)
        
        return completed
    
    def _evict_completed(self) -> List[Request]:
        """Remove completed requests from active batch.
        
        Returns:
            List of evicted requests
        """
        completed = [r for r in self.active_requests if r.state == RequestState.COMPLETED]
        
        self.active_requests = [r for r in self.active_requests if r.state != RequestState.COMPLETED]
        self.completed_requests.extend(completed)
        
        return completed
    
    async def run_iteration(self) -> Dict[str, Any]:
        """Run one iteration of continuous batching.
        
        Returns:
            Dictionary with iteration results
        """
        # Admit new requests
        admitted = self._admit_requests()
        
        # Prefill new requests
        self._run_prefill()
        
        # Decode step
        completed = self._run_decode_step()
        
        # Evict completed
        evicted = self._evict_completed()
        
        return {
            "admitted": admitted,
            "active": len(self.active_requests),
            "completed": len(completed),
            "evicted": len(evicted),
        }
    
    async def run(self, max_iterations: Optional[int] = None):
        """Run continuous batching loop.
        
        Args:
            max_iterations: Maximum iterations (None = run until queue empty)
            
        Yields:
            Iteration results
        """
        self._running = True
        iteration = 0
        
        while self._running:
            result = await self.run_iteration()
            yield result
            
            iteration += 1
            if max_iterations and iteration >= max_iterations:
                break
            
            # Stop if nothing to do
            if not self.active_requests and self.pending_queue.empty():
                break
            
            # Small delay to allow new requests
            await asyncio.sleep(0.001)
        
        self._running = False
    
    def stop(self) -> None:
        """Stop the batching loop."""
        self._running = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics.
        
        Returns:
            Dictionary with statistics
        """
        completed = self.completed_requests
        
        latencies = [r.latency_ms for r in completed if r.latency_ms]
        ttfts = [r.ttft_ms for r in completed if r.ttft_ms]
        
        return {
            "total_requests": self.total_requests,
            "completed_requests": len(completed),
            "total_tokens": self.total_tokens_generated,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
            "avg_ttft_ms": sum(ttfts) / len(ttfts) if ttfts else 0,
            "throughput_tok_s": self.total_tokens_generated / (
                max(r.end_time for r in completed) - min(r.start_time for r in completed)
            ) if completed else 0,
        }


