import typer

app = typer.Typer(
    name="nashgate",
    help="LLM/agent gateway that routes under contention by finding a Nash equilibrium.",
    no_args_is_help=False,
)

BANNER = """
╭─────────────────────────────────────────────────────────────────╮
│ ✻ nashgate                                                       │
│                                                                   │
│   An LLM / agent gateway that routes traffic by finding a Nash   │
│   equilibrium, not a fixed weight table.                         │
│                                                                   │
│   nashgate route    start the gateway (OpenAI-compatible /v1)    │
│   nashgate policy   inspect / train the equilibrium router       │
│   nashgate bench    replay traffic vs. static routing            │
╰─────────────────────────────────────────────────────────────────╯
"""


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        typer.echo(BANNER)
        typer.echo(ctx.get_help())


@app.command()
def route() -> None:
    """Start the gateway (OpenAI-compatible /v1 API)."""
    typer.echo("nashgate route: not implemented yet")


@app.command()
def policy() -> None:
    """Inspect or train the equilibrium-seeking router."""
    typer.echo("nashgate policy: not implemented yet")


@app.command()
def bench() -> None:
    """Replay traffic and compare against static routing."""
    typer.echo("nashgate bench: not implemented yet")


if __name__ == "__main__":
    app()
