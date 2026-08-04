"""open protein language model."""

from __future__ import annotations

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from .model import (
    PRESETS,
    AblmConfig,
    AblmForMaskedLM,
    AblmForSequenceClassification,
    AblmForTokenClassification,
    AblmModel,
    AblmTokenizerFast,
    from_preset,
)

__version__ = "0.0.1"

# In-process registration so `import ablm` plus AutoModel*.from_pretrained
# resolves the ABLM classes (BALM-style). Loading a checkpoint therefore
# requires ablm-forge installed — there is deliberately no `register_for_auto_class`
# / `auto_map` / trust_remote_code path, which would force a flat model package.
# HF's `register` raises on a duplicate model_type, so guard each call to keep
# re-imports idempotent.
AutoConfig.register("ablm", AblmConfig, exist_ok=True)
AutoModel.register(AblmConfig, AblmModel, exist_ok=True)
AutoModelForMaskedLM.register(AblmConfig, AblmForMaskedLM, exist_ok=True)
AutoModelForSequenceClassification.register(
    AblmConfig, AblmForSequenceClassification, exist_ok=True
)
AutoModelForTokenClassification.register(AblmConfig, AblmForTokenClassification, exist_ok=True)
AutoTokenizer.register(AblmConfig, fast_tokenizer_class=AblmTokenizerFast, exist_ok=True)

__all__ = [
    "PRESETS",
    "AblmConfig",
    "AblmForMaskedLM",
    "AblmForSequenceClassification",
    "AblmForTokenClassification",
    "AblmModel",
    "AblmTokenizerFast",
    "__version__",
    "from_preset",
]
