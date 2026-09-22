"""
Qwen2.5-VL wrapper and abstract VLM backend.

This module provides:
  - VLMBackend: Abstract base class for any VLM backend.
  - VLMResult: Dataclass holding a single inference result.
  - QwenVLMWrapper: Concrete implementation for Qwen2.5-VL-3B-Instruct.

Design note
-----------
VLMBackend is intentionally designed so that a Florence-2 wrapper can be
added later without changing the calling notebooks. Replace the backend
by subclassing VLMBackend and passing the new class to experiments.

NOTE on Florence-2 vs Qwen2.5-VL
----------------------------------
These are different model families with different tokenizers, visual encoders,
and output formats. Do NOT assume Qwen results translate directly to Florence-2.
This module only implements Qwen. Florence-2 support is a future TODO.
"""

from __future__ import annotations

import gc
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import torch
from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================
# Data types
# ============================================================

@dataclass
class VLMResult:
    """Container for a single VLM inference result."""

    generated_text: str
    input_token_count: int
    output_token_count: int
    latency_seconds: float
    tokens_per_second: float
    peak_vram_gb: float
    status: str = "ok"          # "ok" | "failed" | "oom"
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to plain dict (JSON-safe)."""
        return {
            "generated_text": self.generated_text,
            "input_token_count": self.input_token_count,
            "output_token_count": self.output_token_count,
            "latency_seconds": self.latency_seconds,
            "tokens_per_second": self.tokens_per_second,
            "peak_vram_gb": self.peak_vram_gb,
            "status": self.status,
            "error": self.error,
            "metadata": self.metadata,
        }


# ============================================================
# Abstract backend
# ============================================================

class VLMBackend(ABC):
    """
    Abstract base class for VLM backends.

    Subclass this to add support for a new model family.
    Currently implemented: QwenVLMWrapper (Qwen2.5-VL).
    Future: Florence2VLMWrapper (Florence-2).
    """

    @abstractmethod
    def load(self) -> None:
        """Load model and processor into memory."""
        ...

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 256,
        kv_cache_config: Optional[Dict[str, Any]] = None,
    ) -> VLMResult:
        """
        Run inference on a single image + text prompt.

        Parameters
        ----------
        image:
            PIL Image (RGB).
        prompt:
            Text instruction/question.
        max_new_tokens:
            Maximum output tokens.
        kv_cache_config:
            Optional dict controlling KV cache backend.
            Example: {"backend": "HQQ", "nbits": 4}

        Returns
        -------
        VLMResult
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """Unload model and free GPU memory."""
        ...


# ============================================================
# Qwen2.5-VL implementation
# ============================================================

class QwenVLMWrapper(VLMBackend):
    """
    Wrapper for Qwen/Qwen2.5-VL-3B-Instruct.

    Supports:
    - CUDA (float16 / bfloat16) and CPU (float32) backends
    - HF DynamicCache (FP16 baseline)
    - HQQ quantized KV cache (INT4, INT2)
    - Quanto quantized KV cache (INT4)
    - Deterministic generation via fixed seed

    Usage
    -----
        wrapper = QwenVLMWrapper(model_name="Qwen/Qwen2.5-VL-3B-Instruct")
        wrapper.load()
        result = wrapper.generate(image, "Describe this scene.")
        wrapper.unload()
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        dtype: str = "float16",
        seed: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        model_name:
            HuggingFace model ID.
        device:
            "auto" | "cuda" | "cpu". "auto" selects CUDA if available.
        dtype:
            "float16" | "bfloat16" | "float32".
            float32 is used automatically on CPU regardless of this setting.
        seed:
            Random seed for reproducibility.
        """
        self.model_name = model_name
        self.seed = seed

        # Resolve device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Resolve dtype
        if self.device == "cpu":
            # CPU does not support float16 reliably in all torch builds
            self.torch_dtype = torch.float32
            logger.warning("CPU mode: forcing dtype=float32 regardless of config.")
        elif dtype == "bfloat16":
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float16

        self.model = None
        self.processor = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Environment metadata
    # ------------------------------------------------------------------

    def get_environment_info(self) -> Dict[str, Any]:
        """Return environment metadata for reproducibility logging."""
        import platform
        info: Dict[str, Any] = {
            "model_name": self.model_name,
            "device": self.device,
            "dtype": str(self.torch_dtype),
            "seed": self.seed,
            "python_version": platform.python_version(),
        }
        try:
            import torch
            info["torch_version"] = torch.__version__
        except ImportError:
            info["torch_version"] = "not_installed"

        try:
            import transformers
            info["transformers_version"] = transformers.__version__
        except ImportError:
            info["transformers_version"] = "not_installed"

        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
            )
        else:
            info["cuda_version"] = "N/A"
            info["gpu_name"] = "CPU only"
            info["gpu_memory_gb"] = 0.0

        try:
            import hqq
            info["hqq_version"] = getattr(hqq, "__version__", "unknown")
        except ImportError:
            info["hqq_version"] = "not_installed"

        try:
            import optimum.quanto as quanto
            info["quanto_version"] = getattr(quanto, "__version__", "unknown")
        except ImportError:
            info["quanto_version"] = "not_installed"

        return info

    # ------------------------------------------------------------------
    # Load / unload
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model and processor from HuggingFace Hub."""
        if self._loaded:
            logger.debug("Model already loaded; skipping.")
            return

        logger.info(f"Loading {self.model_name} on {self.device} ({self.torch_dtype})")

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device if self.device != "cpu" else None,
            trust_remote_code=True,
        )

        if self.device == "cpu":
            self.model = self.model.to("cpu")

        self.model.eval()
        self._loaded = True
        logger.info("Model loaded successfully.")

    def unload(self) -> None:
        """Unload model from memory and free GPU cache."""
        if not self._loaded:
            return
        del self.model
        del self.processor
        self.model = None
        self.processor = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded and GPU cache cleared.")

    # ------------------------------------------------------------------
    # KV cache helpers
    # ------------------------------------------------------------------

    def _build_kv_cache(self, kv_cache_config: Optional[Dict[str, Any]]):
        """
        Construct a KV cache object from config dict.

        Parameters
        ----------
        kv_cache_config:
            None → use HF default (DynamicCache / FP16)
            {"backend": "HQQ", "nbits": 4} → HQQ quantized cache
            {"backend": "HQQ", "nbits": 2} → HQQ INT2 cache
            {"backend": "quanto", "nbits": 4} → Quanto INT4 cache

        Returns
        -------
        A cache object compatible with `model.generate(past_key_values=...)`,
        or None to use the default.

        Raises
        ------
        ImportError if the requested backend library is not installed.
        RuntimeError if the backend is unknown.
        """
        if kv_cache_config is None:
            return None  # Use HF default (DynamicCache)

        backend = kv_cache_config.get("backend", "dynamic").lower()
        nbits = kv_cache_config.get("nbits", 4)

        if backend in ("dynamic", "fp16", "none"):
            return None  # HF DynamicCache default

        elif backend == "hqq":
            try:
                from transformers import HQQConfig, QuantizedCacheConfig
                # transformers >= 4.43 API
                quantization_config = HQQConfig(nbits=nbits, axis=1)
                return QuantizedCacheConfig(
                    backend="HQQ",
                    nbits=nbits,
                    q_group_size=64,
                    residual_length=128,
                )
            except ImportError:
                # Older API fallback
                try:
                    from hqq.core.quantize import HQQBackend, HQQLinear
                    logger.warning(
                        "transformers HQQConfig not available; "
                        "attempting direct hqq construction."
                    )
                    # Return None — caller must handle
                    raise ImportError(
                        "transformers >= 4.43 required for HQQ KV cache support."
                    )
                except ImportError:
                    raise ImportError(
                        "HQQ not installed. Run: pip install hqq>=0.1.7"
                    )

        elif backend in ("quanto", "optimum-quanto"):
            try:
                from transformers import QuantizedCacheConfig
                return QuantizedCacheConfig(
                    backend="quanto",
                    nbits=nbits,
                    residual_length=128,
                )
            except ImportError:
                raise ImportError(
                    "optimum-quanto not installed. Run: pip install optimum-quanto"
                )

        else:
            raise RuntimeError(
                f"Unknown KV cache backend: '{backend}'. "
                "Choose from: dynamic, HQQ, quanto"
            )

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 256,
        kv_cache_config: Optional[Dict[str, Any]] = None,
    ) -> VLMResult:
        """
        Run inference on one image + prompt.

        Measures:
        - Input/output token counts
        - Latency (wall-clock, seconds)
        - Tokens/sec
        - Peak GPU VRAM allocated (GB)

        Parameters
        ----------
        image:
            PIL Image (RGB). Will be resized by processor automatically.
        prompt:
            Text instruction.
        max_new_tokens:
            Max output tokens.
        kv_cache_config:
            KV cache backend config. None = FP16 DynamicCache.

        Returns
        -------
        VLMResult
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call wrapper.load() first.")

        # Set reproducibility seed
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Build chat-style messages for Qwen2.5-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        input_token_count = inputs["input_ids"].shape[-1]

        # Reset VRAM peak counter
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Build KV cache config (may be None for default)
        cache_implementation = None
        generation_config_kwargs: Dict[str, Any] = {}
        try:
            cache_config = self._build_kv_cache(kv_cache_config)
            if cache_config is not None:
                generation_config_kwargs["cache_config"] = cache_config
                generation_config_kwargs["cache_implementation"] = "quantized"
        except (ImportError, RuntimeError) as e:
            # Propagate as error to caller
            return VLMResult(
                generated_text="",
                input_token_count=input_token_count,
                output_token_count=0,
                latency_seconds=0.0,
                tokens_per_second=0.0,
                peak_vram_gb=0.0,
                status="failed",
                error=str(e),
            )

        # Generate
        t_start = time.perf_counter()
        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    **generation_config_kwargs,
                )
        except torch.cuda.OutOfMemoryError as oom_err:
            gc.collect()
            torch.cuda.empty_cache()
            return VLMResult(
                generated_text="",
                input_token_count=input_token_count,
                output_token_count=0,
                latency_seconds=0.0,
                tokens_per_second=0.0,
                peak_vram_gb=self._peak_vram_gb(),
                status="oom",
                error=str(oom_err),
            )
        except Exception as e:
            return VLMResult(
                generated_text="",
                input_token_count=input_token_count,
                output_token_count=0,
                latency_seconds=0.0,
                tokens_per_second=0.0,
                peak_vram_gb=self._peak_vram_gb(),
                status="failed",
                error=str(e),
            )
        t_end = time.perf_counter()
        latency = t_end - t_start

        # Decode — only the newly generated tokens
        generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        output_token_count = generated_ids.shape[-1]
        tps = output_token_count / latency if latency > 0 else 0.0
        peak_vram = self._peak_vram_gb()

        return VLMResult(
            generated_text=generated_text,
            input_token_count=input_token_count,
            output_token_count=output_token_count,
            latency_seconds=latency,
            tokens_per_second=tps,
            peak_vram_gb=peak_vram,
            status="ok",
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _peak_vram_gb() -> float:
        """Return peak GPU memory allocated in GB. Returns 0.0 on CPU."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 ** 3)

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return (
            f"QwenVLMWrapper(model={self.model_name}, "
            f"device={self.device}, dtype={self.torch_dtype}, "
            f"status={status})"
        )
