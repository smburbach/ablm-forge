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
    AblmConfig,
    AblmForMaskedLM,
    AblmForSequenceClassification,
    AblmForTokenClassification,
    AblmModel,
    AblmTokenizerFast,
    LogitsConfig,
    LogitsOutput,
)

__version__ = "0.0.1"

# (1) In-process registration so `import ablm` plus AutoModel*.from_pretrained
# works without trust_remote_code. HF's `register` raises on a duplicate
# model_type, so guard each call to keep re-imports idempotent.
AutoConfig.register("ablm", AblmConfig, exist_ok=True)
AutoModel.register(AblmConfig, AblmModel, exist_ok=True)
AutoModelForMaskedLM.register(AblmConfig, AblmForMaskedLM, exist_ok=True)
AutoModelForSequenceClassification.register(
    AblmConfig, AblmForSequenceClassification, exist_ok=True
)
AutoModelForTokenClassification.register(AblmConfig, AblmForTokenClassification, exist_ok=True)
AutoTokenizer.register(AblmConfig, fast_tokenizer_class=AblmTokenizerFast, exist_ok=True)

# (2) Tell HF to copy the custom-code .py files when push_to_hub is called and
# to write the matching auto_map entries into config.json / tokenizer_config.json.
# Setting auto_map manually is NOT sufficient — register_for_auto_class is the
# documented hook for the file-copy step. These set a class attribute, so repeat
# calls are no-ops.
AblmConfig.register_for_auto_class("AutoConfig")
AblmModel.register_for_auto_class("AutoModel")
AblmForMaskedLM.register_for_auto_class("AutoModelForMaskedLM")
AblmForSequenceClassification.register_for_auto_class("AutoModelForSequenceClassification")
AblmForTokenClassification.register_for_auto_class("AutoModelForTokenClassification")
AblmTokenizerFast.register_for_auto_class("AutoTokenizer")

__all__ = [
    "LogitsConfig",
    "LogitsOutput",
    "AblmConfig",
    "AblmForMaskedLM",
    "AblmForSequenceClassification",
    "AblmForTokenClassification",
    "AblmModel",
    "AblmTokenizerFast",
    "__version__",
]
