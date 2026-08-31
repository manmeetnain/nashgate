import numpy as np
import pytest

from nashgate.policy.agent import NashSACAgent


def make_agent(**kwargs):
    defaults = dict(player_id=0, obs_dim=8, n_actions=3, batch_size=4)
    defaults.update(kwargs)
    return NashSACAgent(**defaults)


def test_select_action_returns_valid_index_when_exploring():
    agent = make_agent()
    obs = np.random.randn(8).astype(np.float32)
    for _ in range(10):
        action = agent.select_action(obs, explore=True)
        assert 0 <= action < 3


def test_select_action_greedy_is_deterministic():
    agent = make_agent()
    obs = np.random.randn(8).astype(np.float32)
    a1 = agent.select_action(obs, explore=False)
    a2 = agent.select_action(obs, explore=False)
    assert a1 == a2


def test_update_returns_none_below_batch_size():
    agent = make_agent(batch_size=64)
    obs = np.random.randn(8).astype(np.float32)
    for _ in range(5):
        agent.store(obs, 0, 1.0, obs, False)
    assert agent.update() is None


def test_update_returns_loss_dict_once_buffer_is_full_enough():
    agent = make_agent(batch_size=4)
    obs = np.random.randn(8).astype(np.float32)
    for _ in range(10):
        agent.store(obs, 0, 1.0, obs, False)
    result = agent.update()
    assert result is not None
    for key in ["actor", "critic1", "critic2", "alpha", "entropy", "alpha_val"]:
        assert key in result
    assert agent.updates == 1


def test_alpha_stays_within_configured_bounds_after_many_updates():
    agent = make_agent(batch_size=4, alpha_min=0.05, alpha_max=0.3)
    obs = np.random.randn(8).astype(np.float32)
    for _ in range(200):
        agent.store(obs, np.random.randint(3), np.random.randn(), obs, False)
        agent.update()
    assert agent.alpha_min - 1e-6 <= agent.alpha <= agent.alpha_max + 1e-6


def test_set_epsilon_floor_updates_actor_in_place():
    agent = make_agent()
    agent.set_epsilon_floor(0.1)
    assert agent.epsilon_floor == 0.1
    assert agent.actor.epsilon_floor == 0.1


def test_set_target_entropy_scale_recomputes_target():
    agent = make_agent()
    agent.set_target_entropy_scale(0.5)
    expected = -np.log(1.0 / agent.n_actions) * 0.5
    assert agent.target_entropy == pytest.approx(expected)


def test_save_and_load_round_trip(tmp_path):
    agent = make_agent(batch_size=4)
    obs = np.random.randn(8).astype(np.float32)
    for _ in range(10):
        agent.store(obs, 0, 1.0, obs, False)
    agent.update()
    original_alpha = agent.alpha
    original_steps = agent.total_steps

    agent.save(str(tmp_path))

    fresh = make_agent(batch_size=4)
    fresh.load(str(tmp_path))
    assert fresh.alpha == pytest.approx(original_alpha)
    assert fresh.total_steps == original_steps
    for p1, p2 in zip(agent.actor.parameters(), fresh.actor.parameters()):
        assert (p1 == p2).all()
