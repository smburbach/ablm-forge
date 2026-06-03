"""Command-line interface for ABLM.

`ablm train ...` forwards its arguments to `ablm.train.main`, which uses the same
OmegaConf surface as the module entry point. Under multi-GPU/FSDP, launch the
module form with torchrun:

    torchrun --standalone --nproc_per_node=8 -m ablm.train --config run.yaml
"""

from __future__ import annotations

import typer

from ablm import __version__

# Pass dotlist overrides (e.g. `model.num_hidden_layers=32`) and flags like
# `--config` straight through to the config loader rather than letting Typer
# parse them.
app = typer.Typer(
    add_completion=False,
    help="ABLM: base model-architecture repo for protein/antibody LM experiments.",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def train(ctx: typer.Context) -> None:
    """Train a model. Forwards all arguments to the config loader.

    Examples:
        ablm train --config run.yaml
        ablm train --preset 170M model.num_hidden_layers=24 train.lr=2e-4
    """
    from ablm.train import main

    main(list(ctx.args))


@app.command()
def info() -> None:
    """Print the registered optimizers."""
    from ablm.training.optim import available_optimizers

    typer.echo(f"ablm-forge {__version__}")
    typer.echo(f"optimizers: {', '.join(available_optimizers())}")


if __name__ == "__main__":
    app()
