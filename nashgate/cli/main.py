import typer

app = typer.Typer(
    name="nashgate",
    help="LLM/agent gateway that routes under contention by finding a Nash equilibrium.",
    no_args_is_help=True,
)


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
