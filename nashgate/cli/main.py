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
def route(
    config: str = typer.Option(..., "--config", "-c", help="Path to gateway config YAML (see docs/example.config.yaml)"),
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
) -> None:
    """Start the gateway (OpenAI-compatible /v1 API)."""
    import uvicorn

    from nashgate.gateway.config import app_from_config

    fastapi_app = app_from_config(config)
    uvicorn.run(fastapi_app, host=host, port=port)


@app.command()
def policy() -> None:
    """Inspect or train the equilibrium-seeking router."""
    typer.echo("nashgate policy: not implemented yet")


@app.command()
def bench(
    config: str = typer.Option(
        ..., "--config", "-c", help="Path to config YAML (same format as `route`; connection fields are ignored)"
    ),
    checkpoint: str = typer.Option(None, "--checkpoint", help="Trained policy dir to evaluate; trains a fresh one if omitted"),
    steps: int = typer.Option(5000, help="Steps to evaluate each router over"),
    train_steps: int = typer.Option(20_000, help="Steps to pretrain a fresh policy over, if no --checkpoint given"),
    seed: int = typer.Option(0, help="Random seed — same seed used for every router's env"),
) -> None:
    """Replay traffic and compare against static routing."""
    from nashgate.bench import compare_routers, format_table, train_policy
    from nashgate.env import obs_dim
    from nashgate.gateway.backends import backends_from_dicts
    from nashgate.gateway.callers import callers_from_dicts
    from nashgate.gateway.config import load_config
    from nashgate.policy import NashEquilibriumRouter

    cfg = load_config(config)
    backend_configs = [b.routing_config for b in backends_from_dicts(cfg["backends"])]
    caller_configs = [c.config for c in callers_from_dicts(cfg["callers"])]

    if checkpoint:
        policy = NashEquilibriumRouter(
            n_players=len(caller_configs),
            obs_dim=obs_dim(len(backend_configs)),
            n_backends=len(backend_configs),
        )
        policy.load(checkpoint)
        typer.echo(f"loaded policy from {checkpoint}")
    else:
        typer.echo(f"no --checkpoint given, pretraining a fresh policy for {train_steps} steps...")
        policy = train_policy(backend_configs, caller_configs, steps=train_steps, seed=seed)

    typer.echo(f"evaluating {len(backend_configs)} backends x {len(caller_configs)} callers over {steps} steps...")
    results = compare_routers(backend_configs, caller_configs, policy, n_steps=steps, seed=seed)
    title = f"nashgate bench — {steps} steps, {len(backend_configs)} backends, {len(caller_configs)} callers, seed={seed}"
    typer.echo(format_table(results, title=title))


if __name__ == "__main__":
    app()
