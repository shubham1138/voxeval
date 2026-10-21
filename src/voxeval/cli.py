"""`voxeval` command line."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

from .types import VoxevalError

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Modular, automated, open-source TTS evaluation.",
)


def _fail(e: Exception) -> None:
    typer.secho(f"error: {e}", fg=typer.colors.RED, err=True)
    raise typer.Exit(1)


@app.command()
def run(
    config: Path = typer.Argument(..., help="Run config YAML."),
    profile: str | None = typer.Option(None, help="Override profile: laptop | gpu."),
    output_dir: str | None = typer.Option(None, help="Override output_dir."),
    limit: int | None = typer.Option(None, help="Use at most N prompts (quick check)."),
    resume: Path | None = typer.Option(None, help="Resume an existing run directory."),
) -> None:
    """Generate audio with the adapter, score it, write a run folder."""
    from .config import load_config
    from .runner import run as run_eval

    try:
        cfg = load_config(config, {"profile": profile, "output_dir": output_dir})
        if limit:
            cfg.suite.limit = limit
        out = run_eval(cfg, run_dir=resume)
    except VoxevalError as e:
        _fail(e)
    typer.echo(
        f"\nDone: {out}\n  samples.csv  summary.csv  summary.json\n  view: voxeval ui {out.parent}"
    )


@app.command()
def aggregate(run_dir: Path) -> None:
    """Recompute samples.csv / summary after editing gates or composite in the run's config."""
    from .aggregate import aggregate as agg

    agg(run_dir)
    typer.echo(f"re-aggregated {run_dir}")


@app.command(name="list")
def list_() -> None:
    """Show registered adapters, metrics, built-in suites and profiles."""
    import voxeval.metrics  # noqa: F401

    from .adapters import ADAPTERS
    from .profiles import PROFILES
    from .registry import METRICS
    from .suite import builtin_suites

    typer.echo("adapters : " + ", ".join(ADAPTERS.names()))
    typer.echo("suites   : " + ", ".join(builtin_suites()))
    typer.echo("profiles : " + ", ".join(PROFILES))
    typer.echo("metrics  :")
    for n in METRICS.names():
        m = METRICS.get(n)
        extra = f"  [pip install 'voxeval[{m.extra}]']" if m.extra else ""
        typer.echo(f"  {n:20s} {m.description}{extra}\n  {'':20s} licence: {m.license}")


@app.command("fetch-voices")
def fetch_voices(
    out: Path = typer.Option(Path("voices"), help="Destination folder."),
    lang: str = typer.Option("en", help="en | es | fr | de"),
    n: int = typer.Option(4, help="Number of reference voices (distinct speakers)."),
) -> None:
    """Download a few openly licensed reference voices (CC-BY) for cloning evaluations."""
    from .voices import fetch

    try:
        paths = fetch(out, lang, n)
    except VoxevalError as e:
        _fail(e)
    typer.echo(f"saved {len(paths)} clips to {out / lang}  (see ATTRIBUTION.md there)")


@app.command()
def ui(runs_dir: Path = typer.Argument(Path("runs")), port: int = 8501) -> None:
    """Open the Streamlit viewer on a folder of runs."""
    app_py = Path(__file__).parent / "ui" / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_py),
        "--server.port",
        str(port),
        "--",
        "--runs-dir",
        str(runs_dir),
    ]
    raise typer.Exit(subprocess.call(cmd))


if __name__ == "__main__":
    app()
