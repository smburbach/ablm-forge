"""Structured configuration for ABLM.

OmegaConf carries the human-facing YAML (defaults → preset → ``--config`` →
CLI dotlist), which `load_config` resolves into:

* an :class:`ablm.model.AblmConfig` (the HF model config),
* a stock :class:`transformers.TrainingArguments` (no subclass),
* an :class:`ablm.training.optim.OptimizerSettings`, and
* a :class:`DataConfig`.

The training knobs map onto HF's native `TrainingArguments` fields — including
`optim` and `lr_scheduler_type` — so the stock `Trainer` does the work and we
write no Trainer code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from omegaconf import DictConfig, OmegaConf

from ablm.training.optim import OptimizerSettings, available_optimizers, resolve_optimizer

if TYPE_CHECKING:
    from transformers import TrainingArguments


AVAILABLE_PRESETS = (
    "50M",
    "170M",
    "400M",
    "800M",
    "1B",
    "3B",
    "6B",
    "12B",
    # ESM-C (EvolutionaryScale Cambrian) exact sizes; arch matches base.yaml.
    "esmc_300m",
    "esmc_600m",
    "esmc_6b",
)

_VALID_SCHEDULERS = ("warmup_linear", "warmup_cosine", "wsd_linear", "wsd_cosine")
_VALID_MIXED_PRECISION = ("bf16", "fp16", "no")
_VALID_COMPILE_MODES = ("default", "reduce-overhead", "max-autotune")
_VALID_MUON_ADJUST_LR_FNS = ("match_rms_adamw", "original")

# YAML scheduler name -> HF lr_scheduler_type. min_lr / WSD plateau are threaded
# through lr_scheduler_kwargs (see _scheduler_kwargs).
_HF_SCHEDULER = {
    "warmup_linear": "linear",
    "warmup_cosine": "cosine",
    "wsd_linear": "warmup_stable_decay",
    "wsd_cosine": "warmup_stable_decay",
}


@dataclass
class TrainConfig:
    """Training configuration (mirrors configs/train/base.yaml)."""

    # Duration
    max_steps: int = 50_000
    max_epochs: int | None = None

    # Batch
    batch_size: int = 32
    gradient_accumulation_steps: int = 1

    # Optimizer
    optimizer: str = "adamw"
    lr: float = 1e-4
    min_lr: float = 0.0
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-8
    muon_adjust_lr_fn: str = "match_rms_adamw"
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    max_grad_norm: float = 1.0

    # Scheduler
    scheduler: str = "warmup_linear"
    warmup_steps: int = 5_000
    stable_steps: int = 0

    # Logging
    log_every: int = 10
    wandb_project: str = "ablm"
    wandb_run_name: str | None = None
    wandb_enabled: bool = True

    # Checkpointing
    save_every: int = 10_000
    save_total_limit: int = 3
    resume_from: str | None = None

    # Infrastructure
    seed: int = 42
    output_dir: str = "outputs"
    config_path: str | None = None
    mixed_precision: str = "bf16"
    compile: bool = False
    compile_mode: str = "default"

    # Memory / sharding (FSDP2 via torchrun; see configs/train/base.yaml)
    gradient_checkpointing: bool = False
    # HF FSDP flag string, e.g. "" (off) or "full_shard auto_wrap".
    fsdp: str = ""
    # HF fsdp_config mapping (fsdp_version, transformer_layer_cls_to_wrap, ...).
    fsdp_config: Any = None

    def __post_init__(self) -> None:
        """Validate training configuration."""
        if self.optimizer not in available_optimizers():
            raise ValueError(
                f"optimizer must be one of {available_optimizers()}, got {self.optimizer!r}"
            )
        if self.muon_adjust_lr_fn not in _VALID_MUON_ADJUST_LR_FNS:
            raise ValueError(
                f"muon_adjust_lr_fn must be one of {_VALID_MUON_ADJUST_LR_FNS}, "
                f"got {self.muon_adjust_lr_fn!r}"
            )
        if self.scheduler not in _VALID_SCHEDULERS:
            raise ValueError(
                f"scheduler must be one of {_VALID_SCHEDULERS}, got {self.scheduler!r}"
            )
        if self.mixed_precision not in _VALID_MIXED_PRECISION:
            raise ValueError(
                f"mixed_precision must be one of {_VALID_MIXED_PRECISION}, "
                f"got {self.mixed_precision!r}"
            )
        if self.compile_mode not in _VALID_COMPILE_MODES:
            raise ValueError(
                f"compile_mode must be one of {_VALID_COMPILE_MODES}, got {self.compile_mode!r}"
            )
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.min_lr < 0:
            raise ValueError(f"min_lr must be >= 0, got {self.min_lr}")
        if self.min_lr > self.lr:
            raise ValueError(f"min_lr ({self.min_lr}) must be <= lr ({self.lr})")
        if self.muon_momentum < 0:
            raise ValueError(f"muon_momentum must be >= 0, got {self.muon_momentum}")
        if self.muon_ns_steps < 1:
            raise ValueError(f"muon_ns_steps must be >= 1, got {self.muon_ns_steps}")
        if self.stable_steps < 0:
            raise ValueError(f"stable_steps must be >= 0, got {self.stable_steps}")
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}"
            )


@dataclass
class TrainDatasetEntry:
    """Parsed configuration for a single training dataset.

    Populated by :func:`ablm.data.config.parse_train_configs`, not directly from YAML.
    """

    name: str
    path: str
    fraction: float


@dataclass
class DataConfig:
    """Data configuration for training datasets and loading."""

    train: Any = None

    # Masked-LM corruption (DataCollatorForLanguageModeling): mask_prob of tokens
    # are selected; of those, mask_token_prob -> <mask>, random_token_prob -> a
    # random token, the rest are kept (BERT 80/10/10 by default).
    mask_prob: float = 0.15
    mask_token_prob: float = 0.8
    random_token_prob: float = 0.1

    # Streaming shuffle buffer (datasets.IterableDataset.shuffle).
    shuffle_buffer_size: int = 10_000

    # DataLoader settings
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 4

    def __post_init__(self) -> None:
        """Validate the masking-split probabilities."""
        if not 0.0 <= self.mask_token_prob <= 1.0:
            raise ValueError(f"mask_token_prob must be in [0, 1], got {self.mask_token_prob}")
        if not 0.0 <= self.random_token_prob <= 1.0:
            raise ValueError(f"random_token_prob must be in [0, 1], got {self.random_token_prob}")
        if self.mask_token_prob + self.random_token_prob > 1.0 + 1e-9:
            raise ValueError(
                "mask_token_prob + random_token_prob must be <= 1, got "
                f"{self.mask_token_prob} + {self.random_token_prob} = "
                f"{self.mask_token_prob + self.random_token_prob}"
            )


@dataclass
class AblmRunConfig:
    """Root configuration composing model, training, and data configs."""

    model: Any = field(default_factory=dict)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)


# ----------------------------------------------------------------------
# Mapping TrainConfig -> stock transformers.TrainingArguments
# ----------------------------------------------------------------------


def _scheduler_kwargs(train: TrainConfig) -> tuple[str, dict[str, Any]]:
    """Return (lr_scheduler_type, lr_scheduler_kwargs) for HF from a TrainConfig."""
    kwargs: dict[str, Any] = {}
    if train.scheduler == "warmup_cosine" and train.min_lr > 0:
        # HF's cosine_with_min_lr takes a min-LR ratio.
        return "cosine_with_min_lr", {"min_lr_rate": train.min_lr / train.lr}
    hf_type = _HF_SCHEDULER[train.scheduler]
    if hf_type == "warmup_stable_decay":
        decay = max(1, train.max_steps - train.warmup_steps - train.stable_steps)
        kwargs = {"num_stable_steps": train.stable_steps, "num_decay_steps": decay}
    return hf_type, kwargs


def optimizer_settings(train: TrainConfig) -> OptimizerSettings:
    """Extract the registry-facing optimizer hyperparameters from a TrainConfig."""
    return OptimizerSettings(
        lr=train.lr,
        weight_decay=train.weight_decay,
        adam_beta1=train.adam_beta1,
        adam_beta2=train.adam_beta2,
        adam_eps=train.adam_eps,
        muon_momentum=train.muon_momentum,
        muon_nesterov=train.muon_nesterov,
        muon_ns_steps=train.muon_ns_steps,
        muon_adjust_lr_fn=train.muon_adjust_lr_fn,
    )


def build_training_arguments(train: TrainConfig, data: DataConfig) -> TrainingArguments:
    """Map a TrainConfig + DataConfig onto a stock `transformers.TrainingArguments`.

    HF-native optimizers are selected via `optim`; custom optimizers (e.g. Muon)
    keep `optim` at a valid default and are injected by the caller through the
    Trainer's `optimizers=` / `optimizer_cls_and_kwargs` hook.
    """
    from transformers import TrainingArguments

    spec = resolve_optimizer(train.optimizer)
    hf_optim = spec.hf_optim or "adamw_torch"
    lr_scheduler_type, lr_scheduler_kwargs = _scheduler_kwargs(train)

    # FSDP2 auto-wraps the AblmBlock (matching AblmPreTrainedModel._no_split_modules)
    # unless the config provides an explicit fsdp_config.
    fsdp_config = train.fsdp_config
    if train.fsdp and fsdp_config is None:
        fsdp_config = {
            "fsdp_version": 2,
            "transformer_layer_cls_to_wrap": ["AblmBlock"],
        }
    # Under FSDP, activation checkpointing belongs in fsdp_config (the Trainer's
    # gradient_checkpointing introduces a redundant all-gather in the backward
    # pass). Route it there and leave TrainingArguments.gradient_checkpointing off.
    use_grad_ckpt_arg = train.gradient_checkpointing and not train.fsdp
    if train.fsdp and train.gradient_checkpointing:
        fsdp_config = {**fsdp_config, "activation_checkpointing": True}

    return TrainingArguments(
        output_dir=train.output_dir,
        max_steps=train.max_steps,
        per_device_train_batch_size=train.batch_size,
        gradient_accumulation_steps=train.gradient_accumulation_steps,
        learning_rate=train.lr,
        weight_decay=train.weight_decay,
        adam_beta1=train.adam_beta1,
        adam_beta2=train.adam_beta2,
        adam_epsilon=train.adam_eps,
        max_grad_norm=train.max_grad_norm,
        optim=hf_optim,
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        warmup_steps=train.warmup_steps,
        logging_steps=train.log_every,
        save_steps=train.save_every,
        save_total_limit=train.save_total_limit,
        seed=train.seed,
        bf16=(train.mixed_precision == "bf16"),
        fp16=(train.mixed_precision == "fp16"),
        gradient_checkpointing=use_grad_ckpt_arg,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fsdp=train.fsdp,
        fsdp_config=fsdp_config,
        dataloader_num_workers=data.num_workers,
        dataloader_pin_memory=data.pin_memory,
        dataloader_prefetch_factor=data.prefetch_factor if data.num_workers > 0 else None,
        # Each rank streams its own shard (build_train_dataset splits by node), so
        # don't let accelerate re-dispatch batches from the main process.
        accelerator_config={"dispatch_batches": False},
        report_to=(["wandb"] if train.wandb_enabled else ["none"]),
        run_name=train.wandb_run_name,
        remove_unused_columns=False,
        torch_compile=train.compile,
        torch_compile_mode=train.compile_mode if train.compile else None,
    )


# ----------------------------------------------------------------------
# OmegaConf loading
# ----------------------------------------------------------------------


def get_preset_config(preset: str) -> DictConfig:
    """Load a model size preset by name."""
    if preset not in AVAILABLE_PRESETS:
        raise ValueError(
            f"Unknown preset {preset!r}. Available presets: {', '.join(AVAILABLE_PRESETS)}"
        )
    preset_dir = files("ablm.configs.model.presets")
    yaml_text = preset_dir.joinpath(f"{preset}.yaml").read_text()
    return cast("DictConfig", OmegaConf.create(yaml_text))


_BASE_CONFIG_LAYERS = (
    ("ablm.configs.model", "base.yaml"),
    ("ablm.configs.train", "base.yaml"),
    ("ablm.configs.data", "base.yaml"),
)


def _load_packaged_yaml(package: str, filename: str) -> DictConfig:
    """Load a YAML resource shipped inside the package as a DictConfig."""
    text = files(package).joinpath(filename).read_text()
    return cast("DictConfig", OmegaConf.create(text))


def load_config(argv: list[str]) -> AblmRunConfig:
    """Load config from defaults, optional preset, optional YAML file, and CLI overrides.

    Merge order (later overrides earlier): defaults → preset → YAML file → CLI overrides.

    Args:
        argv: Command-line arguments. Supports ``--preset <name>``,
            ``--config <path>``, ``--name <run>``, and dotlist overrides like
            ``model.num_hidden_layers=32``.

    Returns:
        A fully resolved `AblmRunConfig` with `model` instantiated as an
        `ablm.model.AblmConfig`.
    """
    base: DictConfig = OmegaConf.structured(AblmRunConfig)
    OmegaConf.set_struct(base, False)

    for package, filename in _BASE_CONFIG_LAYERS:
        base = cast("DictConfig", OmegaConf.merge(base, _load_packaged_yaml(package, filename)))

    config_path: str | None = None
    preset: str | None = None
    run_name: str | None = None
    remaining: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--config" and i + 1 < len(argv):
            config_path = str(Path(argv[i + 1]).expanduser().resolve())
            i += 2
        elif argv[i] == "--preset" and i + 1 < len(argv):
            preset = argv[i + 1]
            i += 2
        elif argv[i] == "--name" and i + 1 < len(argv):
            run_name = argv[i + 1]
            i += 2
        else:
            remaining.append(argv[i])
            i += 1

    overrides: list[DictConfig] = []
    if preset is not None:
        overrides.append(get_preset_config(preset))
    if config_path is not None:
        overrides.append(cast("DictConfig", OmegaConf.load(config_path)))
    if remaining:
        overrides.append(OmegaConf.from_dotlist(remaining))

    for ov in overrides:
        base = cast("DictConfig", OmegaConf.merge(base, ov))

    cfg: AblmRunConfig = OmegaConf.to_object(base)  # ty: ignore[invalid-assignment]  # OmegaConf union

    from ablm.model import AblmConfig as AblmModelConfig

    model_dict = OmegaConf.to_container(base.model, resolve=True) or {}
    cfg.model = AblmModelConfig(**model_dict)  # ty: ignore[invalid-argument-type]  # OmegaConf union

    cfg.train.config_path = config_path
    if run_name is not None and cfg.train.wandb_run_name is None:
        cfg.train.wandb_run_name = run_name

    return cfg
