"""All public Ablm* model classes.

Every public model class lives in one module (standard HF convention);
the architecture building blocks live in `.layers` and are imported here.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from .configuration_ablm import AblmConfig
from .layers.embedding import cls_pool, mean_pool
from .layers.norm import AblmLayerNorm, AblmRMSNorm, make_norm
from .layers.transformer import AblmBlock, AblmStack

__all__ = [
    "AblmPreTrainedModel",
    "AblmModel",
    "AblmForMaskedLM",
    "AblmForSequenceClassification",
    "AblmForTokenClassification",
    "AblmMLMHead",
]


class AblmPreTrainedModel(PreTrainedModel):
    """Abstract base for every ABLM model: weight init and gradient checkpointing."""

    config_class = AblmConfig
    base_model_prefix = "ablm"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _no_split_modules = ["AblmBlock"]
    _supports_sdpa = True

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights (truncated normal; residual-writer scaling).

        Residual-stream-writing projections (attention `o_proj`, FFN
        `down_proj`) are marked `_is_residual_writer = True` in their parent
        modules; when `config.init_scale_output_projections` is set, their std
        is shrunk by `1/sqrt(2 * num_hidden_layers)` to keep the residual
        stream variance stable with depth.
        """
        std = self.config.initializer_range

        if isinstance(module, nn.Linear):
            module_std = std
            if self.config.mup_enabled:
                module_std = std * math.sqrt(self.config.mup_base_hidden_size / module.in_features)
            if getattr(module, "_is_residual_writer", False) and (
                self.config.init_scale_output_projections
            ):
                module_std = module_std / math.sqrt(2 * self.config.num_hidden_layers)
            nn.init.trunc_normal_(
                module.weight, mean=0.0, std=module_std, a=-2 * module_std, b=2 * module_std
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
            # HF convention: do NOT zero the <pad> row.
        elif isinstance(module, (AblmLayerNorm, AblmRMSNorm)):
            if getattr(module, "weight", None) is not None:
                nn.init.ones_(module.weight)
            if getattr(module, "bias", None) is not None:
                # nn.Module.__getattr__ types `.bias` as Tensor | Module; it is a Tensor here.
                nn.init.zeros_(module.bias)  # ty: ignore[invalid-argument-type]

    # ------------------------------------------------------------------
    # Gradient checkpointing — propagate the toggle to every AblmBlock.
    # ------------------------------------------------------------------

    def _set_gradient_checkpointing(self, value: bool) -> None:
        self.gradient_checkpointing = value
        for module in self.modules():
            if isinstance(module, AblmBlock):
                module.gradient_checkpointing = value
            if isinstance(module, AblmStack):
                module.gradient_checkpointing = value

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None) -> None:
        """Enable activation checkpointing on every transformer block."""
        self._set_gradient_checkpointing(True)

    def gradient_checkpointing_disable(self) -> None:
        """Disable activation checkpointing on every transformer block."""
        self._set_gradient_checkpointing(False)


class AblmModel(AblmPreTrainedModel):
    """The ABLM encoder backbone — token embedding, N × AblmBlock, final norm.

    Returns a `BaseModelOutput`. Has no task head.
    """

    def __init__(self, config: AblmConfig) -> None:
        super().__init__(config)
        self.backbone = AblmStack(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.embed_tokens.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.backbone.embed_tokens.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> BaseModelOutput | tuple:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        last_hidden, hidden_states, attentions = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        if not return_dict:
            return tuple(v for v in (last_hidden, hidden_states, attentions) if v is not None)
        return BaseModelOutput(
            last_hidden_state=last_hidden,
            hidden_states=hidden_states,
            attentions=attentions,
        )


class AblmMLMHead(nn.Module):
    """Masked-language-model head: dense -> activation -> norm -> decoder."""

    def __init__(self, config: AblmConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.act = ACT2FN[config.mlm_head_activation]
        self.norm = make_norm(
            config.norm_type,
            config.hidden_size,
            eps=config.norm_eps,
            bias=getattr(config, "norm_bias", True),
        )
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        # The decoder writes to vocab space, NOT the residual stream: no 1/sqrt(2L) scaling.
        self.decoder._is_residual_writer = False  # ty: ignore[unresolved-attribute]  # nn.Module setattr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.norm(self.act(self.dense(x))))


class AblmForMaskedLM(AblmPreTrainedModel):
    """ABLM with a masked-language-model head."""

    _tied_weights_keys = {
        "lm_head.decoder.weight": "ablm.backbone.embed_tokens.embed_tokens.weight"
    }

    def __init__(self, config: AblmConfig) -> None:
        super().__init__(config)
        self.ablm = AblmModel(config)
        self.lm_head = AblmMLMHead(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.ablm.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.ablm.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> MaskedLMOutput | tuple:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.ablm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        logits = self.lm_head(outputs.last_hidden_state).float()
        if self.config.mup_output_mult != 1.0:
            logits = logits * self.config.mup_output_mult

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits, outputs.hidden_states, outputs.attentions)
            output = tuple(v for v in output if v is not None)
            return ((loss,) + output) if loss is not None else output
        return MaskedLMOutput(
            # transformers stubs annotate loss as FloatTensor; torch ops return Tensor.
            loss=loss,  # ty: ignore[invalid-argument-type]
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class AblmForSequenceClassification(AblmPreTrainedModel):
    """ABLM with a pooled sequence-classification head."""

    def __init__(self, config: AblmConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.ablm = AblmModel(config)
        if config.pre_head_norm:
            self.pre_head_norm: nn.Module = make_norm(
                config.norm_type,
                config.hidden_size,
                eps=config.norm_eps,
                bias=getattr(config, "norm_bias", True),
            )
        else:
            self.pre_head_norm = nn.Identity()
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=True)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.ablm.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.ablm.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> SequenceClassifierOutput | tuple:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.ablm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state

        if self.config.classifier_pool == "cls":
            pooled = cls_pool(last_hidden)
        elif self.config.classifier_pool == "mean":
            pooled_mask = self._pooling_mask(attention_mask, last_hidden)
            pooled = mean_pool(last_hidden, pooled_mask)
        else:
            raise ValueError(
                f"Unknown classifier_pool {self.config.classifier_pool!r}; "
                "expected 'mean' or 'cls'."
            )

        pooled = self.pre_head_norm(pooled)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled).float()

        loss = self._compute_loss(logits, labels)

        if not return_dict:
            output = (logits, outputs.hidden_states, outputs.attentions)
            output = tuple(v for v in output if v is not None)
            return ((loss,) + output) if loss is not None else output
        return SequenceClassifierOutput(
            # transformers stubs annotate loss as FloatTensor; torch ops return Tensor.
            loss=loss,  # ty: ignore[invalid-argument-type]
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def _pooling_mask(attention_mask: torch.Tensor | None, hidden: torch.Tensor) -> torch.Tensor:
        if attention_mask is not None:
            return attention_mask
        return hidden.new_ones(hidden.shape[:2], dtype=torch.long)

    def _compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Loss with HF-style `problem_type` inference (regression / classification)."""
        if labels is None:
            return None

        problem_type = self.config.problem_type
        if problem_type is None:
            if self.num_labels == 1:
                problem_type = "regression"
            elif labels.dtype in (torch.long, torch.int):
                problem_type = "single_label_classification"
            else:
                problem_type = "multi_label_classification"
            self.config.problem_type = problem_type

        if problem_type == "regression":
            if self.num_labels == 1:
                return F.mse_loss(logits.squeeze(-1), labels.squeeze(-1))
            return F.mse_loss(logits, labels)
        if problem_type == "single_label_classification":
            return F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        return F.binary_cross_entropy_with_logits(logits, labels.float())


class AblmForTokenClassification(AblmPreTrainedModel):
    """ABLM with a per-token classification head."""

    def __init__(self, config: AblmConfig) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        self.ablm = AblmModel(config)
        if config.pre_head_norm:
            self.pre_head_norm: nn.Module = make_norm(
                config.norm_type,
                config.hidden_size,
                eps=config.norm_eps,
                bias=getattr(config, "norm_bias", True),
            )
        else:
            self.pre_head_norm = nn.Identity()
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=True)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.ablm.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.ablm.set_input_embeddings(value)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> TokenClassifierOutput | tuple:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.ablm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden = self.pre_head_norm(outputs.last_hidden_state)
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden).float()

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits, outputs.hidden_states, outputs.attentions)
            output = tuple(v for v in output if v is not None)
            return ((loss,) + output) if loss is not None else output
        return TokenClassifierOutput(
            # transformers stubs annotate loss as FloatTensor; torch ops return Tensor.
            loss=loss,  # ty: ignore[invalid-argument-type]
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
