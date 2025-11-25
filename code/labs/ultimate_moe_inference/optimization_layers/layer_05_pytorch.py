"""Layer 05: PyTorch Optimizations (Ch13-14).

PyTorch-level optimizations:
- FP8 via Transformer Engine
- torch.compile with TorchInductor
- Triton custom kernels
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn


# Try to import Transformer Engine
TE_AVAILABLE = False
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe as te_recipe
    TE_AVAILABLE = True
except ImportError:
    te = None
    te_recipe = None


class Layer05PyTorch:
    """Layer 5: PyTorch optimizations from Chapters 13-14.
    
    These optimizations use PyTorch's advanced features for
    reduced precision and kernel optimization.
    
    Techniques:
    - Ch13: FP8 via Transformer Engine, mixed precision
    - Ch14: torch.compile, TorchInductor, Triton kernels
    """
    
    name = "Layer 05: PyTorch (Ch13-14)"
    chapters = [13, 14]
    
    def __init__(self):
        """Initialize layer."""
        self._fp8_enabled = False
        self._compiled = False
        self._fp8_recipe = None
        self._compile_mode = "max-autotune"
    
    def apply(self, benchmark: Any) -> None:
        """Apply PyTorch optimizations.
        
        Args:
            benchmark: Benchmark instance to configure
        """
        config = getattr(benchmark, 'config', None)
        model = getattr(benchmark, 'model', None)
        
        if config:
            if getattr(config, 'use_fp8_kv_cache', False):
                self.setup_fp8()
            
            if getattr(config, 'use_torch_compile', True):
                self._compile_mode = getattr(config, 'compile_mode', 'max-autotune')
    
    def setup_fp8(self) -> None:
        """Setup FP8 recipe for Transformer Engine (Ch13).
        
        Creates an optimized FP8 recipe with:
        - DelayedScaling for stable training/inference
        - Hysteresis to prevent scale oscillation
        - Long amax history for better estimation
        """
        if not TE_AVAILABLE:
            print("  [Layer 05] Transformer Engine not available, skipping FP8")
            return
        
        # Create optimized FP8 recipe
        self._fp8_recipe = te_recipe.DelayedScaling(
            margin=8,  # Increased margin for stability
            interval=1,  # Update amax every iteration
            amax_history_len=1024,  # Long history for better estimation
            fp8_format=te_recipe.Format.HYBRID,  # E4M3 for forward, E5M2 for backward
        )
        
        self._fp8_enabled = True
        print("  [Layer 05] FP8 recipe configured (DelayedScaling)")
    
    def compile_model(
        self,
        model: nn.Module,
        mode: Optional[str] = None,
        fullgraph: bool = True,
        dynamic: bool = False,
    ) -> nn.Module:
        """Compile model with torch.compile (Ch14).
        
        torch.compile uses TorchInductor to optimize the model:
        - Kernel fusion reduces memory bandwidth
        - Triton codegen for custom kernels
        - Automatic optimization based on mode
        
        Args:
            model: Model to compile
            mode: Compile mode ("default", "reduce-overhead", "max-autotune")
            fullgraph: Require full graph compilation
            dynamic: Enable dynamic shapes
            
        Returns:
            Compiled model
        """
        mode = mode or self._compile_mode
        
        print(f"  [Layer 05] Compiling model with mode='{mode}'")
        
        compiled = torch.compile(
            model,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        
        self._compiled = True
        return compiled
    
    def fp8_autocast_context(self):
        """Get FP8 autocast context manager (Ch13).
        
        Returns:
            Context manager for FP8 execution
        """
        if not self._fp8_enabled or te is None:
            from contextlib import nullcontext
            return nullcontext()
        
        return te.fp8_autocast(enabled=True, fp8_recipe=self._fp8_recipe)
    
    def wrap_linear_fp8(self, linear: nn.Linear) -> nn.Module:
        """Wrap a linear layer with FP8 (Ch13).
        
        Args:
            linear: Linear layer to wrap
            
        Returns:
            FP8-enabled linear layer
        """
        if not TE_AVAILABLE:
            return linear
        
        # Create TE linear with same config
        te_linear = te.Linear(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
        )
        
        # Copy weights
        te_linear.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            te_linear.bias.data.copy_(linear.bias.data)
        
        return te_linear
    
    @staticmethod
    def configure_inductor() -> None:
        """Configure TorchInductor settings (Ch14).
        
        Sets optimal inductor configuration for inference.
        """
        try:
            import torch._inductor.config as inductor_config
            
            # Enable CUDA graphs in inductor
            inductor_config.triton.cudagraphs = True
            
            # Enable max autotune for best kernels
            inductor_config.max_autotune = True
            
            # Coordinate descent tuning for better results
            inductor_config.coordinate_descent_tuning = True
            
            print("  [Layer 05] Configured TorchInductor for inference")
        except ImportError:
            pass
    
    def get_compile_stats(self) -> Dict[str, Any]:
        """Get compilation statistics (Ch14).
        
        Returns:
            Dictionary with compile statistics
        """
        stats = {
            "compiled": self._compiled,
            "mode": self._compile_mode,
        }
        
        try:
            # Get inductor metrics if available
            from torch._dynamo import utils as dynamo_utils
            stats["cache_size"] = dynamo_utils.counters.get("graph_breaks", 0)
        except ImportError:
            pass
        
        return stats
    
    def get_status(self) -> dict:
        """Get status of applied optimizations."""
        return {
            "fp8_enabled": self._fp8_enabled,
            "compiled": self._compiled,
            "compile_mode": self._compile_mode,
            "te_available": TE_AVAILABLE,
        }
    
    def __str__(self) -> str:
        """Return string representation."""
        return f"{self.name}: FP8={self._fp8_enabled}, compiled={self._compiled}"


