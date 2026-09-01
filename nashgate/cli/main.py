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


policy_app = typer.Typer(help="Train or inspect the equilibrium-seeking router.")
app.add_typer(policy_app, name="policy")


def _load_backend_and_caller_configs(config: str):
    from nashgate.gateway.backends import backends_from_dicts
    from nashgate.gateway.callers import callers_from_dicts
    from nashgate.gateway.config import load_config

    cfg = load_config(config)
    backend_configs = [b.routing_config for b in backends_from_dicts(cfg["backends"])]
    caller_configs = [c.config for c in callers_from_dicts(cfg["callers"])]
    return backend_configs, caller_configs


@policy_app.command("train")
def policy_train(
    config: str = typer.Option(
        ..., "--config", "-c", help="Path to config YAML (same format as `route`; connection fields are ignored)"
    ),
    out: str = typer.Option(..., "--out", "-o", help="Directory to save the trained checkpoint into"),
    steps: int = typer.Option(20_000, help="Training steps"),
    seed: int = typer.Option(0, help="Random seed"),
) -> None:
    """Train a fresh policy against the routing game and save a checkpoint."""
    from nashgate.policy.train import train_policy

    backend_configs, caller_configs = _load_backend_and_caller_configs(config)

    def report(step: int, total: int, stats: dict) -> None:
        typer.echo(
            f"  step {step:>7}/{total}  mean_reward={stats['mean_reward']:+.3f}  mean_alpha={stats['mean_alpha']:.3f}"
        )

    typer.echo(f"training {len(caller_configs)} players x {len(backend_configs)} backends for {steps} steps...")
    policy = train_policy(
        backend_configs, caller_configs, steps=steps, seed=seed,
        on_progress=report, progress_every=max(1, steps // 10),
    )
    policy.save(out)
    typer.echo(f"saved checkpoint to {out}")


@policy_app.command("inspect")
def policy_inspect(
    config: str = typer.Option(..., "--config", "-c", help="Config YAML — used to derive obs_dim/n_backends/n_players"),
    checkpoint: str = typer.Option(..., "--checkpoint", help="Checkpoint directory to inspect"),
) -> None:
    """Print per-player training stats from a saved checkpoint."""
    from nashgate.env import obs_dim
    from nashgate.policy import NashEquilibriumRouter

    backend_configs, caller_configs = _load_backend_and_caller_configs(config)

    policy = NashEquilibriumRouter(
        n_players=len(caller_configs), obs_dim=obs_dim(len(backend_configs)), n_backends=len(backend_configs)
    )
    policy.load(checkpoint)

    typer.echo(f"{'player':<8} {'steps':>10} {'updates':>10} {'alpha':>8}")
    for pid, agent in policy.agents.items():
        typer.echo(f"{pid:<8} {agent.total_steps:>10} {agent.updates:>10} {agent.alpha:>8.4f}")


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
    from nashgate.policy import NashEquilibriumRouter

    backend_configs, caller_configs = _load_backend_and_caller_configs(config)

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
