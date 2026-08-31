import torch

from nashgate.policy.networks import ActorNetwork, CriticNetwork


def test_actor_output_probs_sum_to_one():
    actor = ActorNetwork(obs_dim=10, n_actions=4)
    obs = torch.randn(8, 10)
    probs, log_probs = actor(obs)
    assert probs.shape == (8, 4)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)
    assert torch.allclose(log_probs, torch.log(probs + 1e-8), atol=1e-5)


def test_actor_epsilon_floor_prevents_zero_probability():
    # push logits to an extreme so softmax alone would saturate to ~0/~1
    actor = ActorNetwork(obs_dim=2, n_actions=3, epsilon_floor=0.05)
    with torch.no_grad():
        for p in actor.net.parameters():
            p.zero_()
        actor.net[-1].bias.copy_(torch.tensor([100.0, -100.0, -100.0]))
    obs = torch.zeros(1, 2)
    probs, _ = actor(obs)
    assert (probs > 0.0).all()
    assert (probs < 1.0).all()
    # floor guarantees at least epsilon/n_actions on every action
    assert probs.min().item() >= (0.05 / 3) - 1e-4


def test_actor_sample_returns_valid_action_and_nonzero_entropy():
    actor = ActorNetwork(obs_dim=6, n_actions=5)
    obs = torch.randn(4, 6)
    action, log_prob, entropy = actor.sample(obs)
    assert action.shape == (4,)
    assert ((action >= 0) & (action < 5)).all()
    assert (entropy > 0).all()


def test_critic_output_shape():
    critic = CriticNetwork(obs_dim=10, n_actions=4)
    obs = torch.randn(8, 10)
    q_values = critic(obs)
    assert q_values.shape == (8, 4)
