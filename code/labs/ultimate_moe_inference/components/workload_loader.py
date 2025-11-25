"""Workload loader for ShareGPT and synthetic prompts.

Provides realistic prompt distributions for benchmarking:
- ShareGPT: Real conversation prompts with variable lengths
- Synthetic: Fixed-length prompts for controlled testing
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False


class WorkloadType(Enum):
    """Type of workload."""
    SYNTHETIC = "synthetic"
    SHAREGPT = "sharegpt"


@dataclass
class Prompt:
    """A single prompt for inference."""
    
    text: str
    tokens: Optional[List[int]] = None
    expected_output_tokens: int = 256
    
    # Metadata
    source: str = "synthetic"
    conversation_id: Optional[str] = None
    turn_index: int = 0
    
    def __len__(self) -> int:
        """Return number of tokens."""
        if self.tokens:
            return len(self.tokens)
        # Rough estimate: 4 chars per token
        return len(self.text) // 4


@dataclass
class WorkloadConfig:
    """Configuration for workload generation."""
    
    workload_type: WorkloadType = WorkloadType.SYNTHETIC
    num_samples: int = 100
    
    # Prompt parameters
    min_prompt_tokens: int = 128
    max_prompt_tokens: int = 2048
    mean_prompt_tokens: int = 512
    
    # Output parameters
    min_output_tokens: int = 32
    max_output_tokens: int = 512
    mean_output_tokens: int = 128
    
    # ShareGPT specific
    sharegpt_dataset: str = "anon8231489123/ShareGPT_Vicuna_unfiltered"
    filter_long_prompts: bool = True
    
    # System prompt for testing prefix caching
    system_prompt: Optional[str] = None
    system_prompt_ratio: float = 0.5  # Fraction of prompts with system prompt


class WorkloadLoader:
    """Load and generate inference workloads.
    
    Supports:
    - ShareGPT: Real conversation prompts
    - Synthetic: Generated prompts with controlled lengths
    
    Example:
        loader = WorkloadLoader(WorkloadConfig(workload_type=WorkloadType.SHAREGPT))
        for prompt in loader.get_prompts(100):
            outputs = model.generate(prompt.text)
    """
    
    # Default system prompt for testing prefix caching
    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful, harmless, and honest AI assistant. "
        "You provide clear, accurate, and thoughtful responses. "
        "If you're unsure about something, you say so."
    )
    
    def __init__(
        self,
        config: Optional[WorkloadConfig] = None,
        tokenizer: Optional[Any] = None,
    ):
        """Initialize workload loader.
        
        Args:
            config: Workload configuration
            tokenizer: Tokenizer for token counting (optional)
        """
        self.config = config or WorkloadConfig()
        self.tokenizer = tokenizer
        self._prompts: List[Prompt] = []
        self._loaded = False
    
    def load(self) -> None:
        """Load prompts from source."""
        if self.config.workload_type == WorkloadType.SHAREGPT:
            self._load_sharegpt()
        else:
            self._load_synthetic()
        
        self._loaded = True
    
    def _load_sharegpt(self) -> None:
        """Load prompts from ShareGPT dataset."""
        if not DATASETS_AVAILABLE:
            print("Warning: datasets library not available, falling back to synthetic")
            self._load_synthetic()
            return
        
        print(f"Loading ShareGPT dataset: {self.config.sharegpt_dataset}")
        
        try:
            dataset = load_dataset(
                self.config.sharegpt_dataset,
                split="train",
            )
        except Exception as e:
            print(f"Warning: Failed to load ShareGPT: {e}, falling back to synthetic")
            self._load_synthetic()
            return
        
        # Extract prompts from conversations
        prompts = []
        for idx, item in enumerate(dataset):
            if len(prompts) >= self.config.num_samples:
                break
            
            conversations = item.get("conversations", [])
            if not conversations:
                continue
            
            # Get human messages
            for turn_idx, turn in enumerate(conversations):
                if turn.get("from") == "human":
                    text = turn.get("value", "")
                    
                    # Filter by length if configured
                    if self.config.filter_long_prompts:
                        token_count = self._estimate_tokens(text)
                        if token_count > self.config.max_prompt_tokens:
                            continue
                    
                    # Optionally add system prompt
                    if (self.config.system_prompt or 
                        random.random() < self.config.system_prompt_ratio):
                        sys_prompt = self.config.system_prompt or self.DEFAULT_SYSTEM_PROMPT
                        text = f"{sys_prompt}\n\nUser: {text}"
                    
                    prompts.append(Prompt(
                        text=text,
                        expected_output_tokens=self.config.mean_output_tokens,
                        source="sharegpt",
                        conversation_id=str(idx),
                        turn_index=turn_idx,
                    ))
                    
                    if len(prompts) >= self.config.num_samples:
                        break
        
        self._prompts = prompts
        print(f"Loaded {len(self._prompts)} prompts from ShareGPT")
    
    def _load_synthetic(self) -> None:
        """Generate synthetic prompts."""
        print(f"Generating {self.config.num_samples} synthetic prompts")
        
        # Sample prompt templates
        templates = [
            "Explain the concept of {} in simple terms.",
            "Write a detailed analysis of {}.",
            "Compare and contrast {} and {}.",
            "What are the main advantages and disadvantages of {}?",
            "Describe the history and evolution of {}.",
            "How does {} work? Provide a technical explanation.",
            "What are the best practices for {}?",
            "Summarize the key points of {}.",
            "What is the future outlook for {}?",
            "Provide a step-by-step guide to {}.",
        ]
        
        topics = [
            "artificial intelligence",
            "machine learning",
            "neural networks",
            "deep learning",
            "natural language processing",
            "computer vision",
            "reinforcement learning",
            "transformer models",
            "GPU computing",
            "distributed systems",
            "cloud computing",
            "quantum computing",
            "blockchain technology",
            "cybersecurity",
            "software engineering",
        ]
        
        prompts = []
        for i in range(self.config.num_samples):
            template = random.choice(templates)
            topic = random.choice(topics)
            
            # Generate prompt
            if "{}" in template and template.count("{}") == 2:
                topic2 = random.choice([t for t in topics if t != topic])
                text = template.format(topic, topic2)
            else:
                text = template.format(topic)
            
            # Optionally add system prompt
            if (self.config.system_prompt or 
                random.random() < self.config.system_prompt_ratio):
                sys_prompt = self.config.system_prompt or self.DEFAULT_SYSTEM_PROMPT
                text = f"{sys_prompt}\n\nUser: {text}"
            
            # Pad to desired length if needed
            target_tokens = random.randint(
                self.config.min_prompt_tokens,
                self.config.max_prompt_tokens,
            )
            text = self._pad_to_length(text, target_tokens)
            
            prompts.append(Prompt(
                text=text,
                expected_output_tokens=random.randint(
                    self.config.min_output_tokens,
                    self.config.max_output_tokens,
                ),
                source="synthetic",
            ))
        
        self._prompts = prompts
        print(f"Generated {len(self._prompts)} synthetic prompts")
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate number of tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        # Rough estimate: ~4 characters per token
        return len(text) // 4
    
    def _pad_to_length(self, text: str, target_tokens: int) -> str:
        """Pad text to reach target token count."""
        current_tokens = self._estimate_tokens(text)
        
        if current_tokens >= target_tokens:
            return text
        
        # Add filler content
        filler = " Additionally, please provide more context and detail."
        while self._estimate_tokens(text) < target_tokens:
            text += filler
        
        return text
    
    def get_prompts(self, n: Optional[int] = None) -> List[Prompt]:
        """Get list of prompts.
        
        Args:
            n: Number of prompts to return (None = all)
            
        Returns:
            List of Prompt objects
        """
        if not self._loaded:
            self.load()
        
        if n is None:
            return self._prompts
        
        return self._prompts[:n]
    
    def __iter__(self) -> Iterator[Prompt]:
        """Iterate over prompts."""
        if not self._loaded:
            self.load()
        return iter(self._prompts)
    
    def __len__(self) -> int:
        """Return number of prompts."""
        if not self._loaded:
            self.load()
        return len(self._prompts)
    
    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> Iterator[List[Prompt]]:
        """Get prompts in batches.
        
        Args:
            batch_size: Size of each batch
            shuffle: Whether to shuffle prompts
            
        Yields:
            Lists of Prompt objects
        """
        if not self._loaded:
            self.load()
        
        prompts = list(self._prompts)
        if shuffle:
            random.shuffle(prompts)
        
        for i in range(0, len(prompts), batch_size):
            yield prompts[i:i + batch_size]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the workload.
        
        Returns:
            Dictionary with workload statistics
        """
        if not self._loaded:
            self.load()
        
        if not self._prompts:
            return {}
        
        token_counts = [len(p) for p in self._prompts]
        
        return {
            "num_prompts": len(self._prompts),
            "workload_type": self.config.workload_type.value,
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "mean_tokens": sum(token_counts) / len(token_counts),
            "total_tokens": sum(token_counts),
        }


def load_sharegpt_workload(
    num_samples: int = 1000,
    tokenizer: Optional[Any] = None,
) -> List[Prompt]:
    """Convenience function to load ShareGPT workload.
    
    Args:
        num_samples: Number of prompts to load
        tokenizer: Tokenizer for token counting
        
    Returns:
        List of Prompt objects
    """
    config = WorkloadConfig(
        workload_type=WorkloadType.SHAREGPT,
        num_samples=num_samples,
    )
    loader = WorkloadLoader(config, tokenizer)
    return loader.get_prompts()


def load_synthetic_workload(
    num_samples: int = 100,
    min_tokens: int = 128,
    max_tokens: int = 2048,
) -> List[Prompt]:
    """Convenience function to generate synthetic workload.
    
    Args:
        num_samples: Number of prompts to generate
        min_tokens: Minimum prompt tokens
        max_tokens: Maximum prompt tokens
        
    Returns:
        List of Prompt objects
    """
    config = WorkloadConfig(
        workload_type=WorkloadType.SYNTHETIC,
        num_samples=num_samples,
        min_prompt_tokens=min_tokens,
        max_prompt_tokens=max_tokens,
    )
    loader = WorkloadLoader(config)
    return loader.get_prompts()


