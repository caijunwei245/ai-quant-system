import unittest
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from layers.execution import PortfolioEnv


class TestPortfolioEnv(unittest.TestCase):
    def test_gym_environment_interface(self):
        env = PortfolioEnv()
        self.assertIsNotNone(env.observation_space)
        self.assertIsNotNone(env.action_space)
        self.assertEqual(env.action_space.shape, (1,))
        self.assertEqual(env.action_space.dtype.name, "float32")

        obs, info = env.reset(seed=42)
        self.assertEqual(obs.shape[0], env.observation_space.shape[0])
        self.assertIsInstance(info, dict)

        action = env.action_space.sample()
        obs2, reward, terminated, truncated, info2 = env.step(action)
        self.assertEqual(obs2.shape[0], env.observation_space.shape[0])
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info2, dict)

    def test_state_dimension_matches_config(self):
        env = PortfolioEnv(config={"position": {"state_dim": 24}})
        obs, _ = env.reset(seed=1)
        self.assertEqual(obs.shape[0], 24)

    def test_build_ppo_state_matches_state_dim(self):
        exec_env = PortfolioEnv(config={"position": {"state_dim": 20}})
        state = exec_env._get_state()
        self.assertEqual(state.shape[0], 20)

        market_state = {"volatility": 0.05, "momentum": 0.02}
        # Simulate build state from verified stock list
        from layers.execution import ExecutionLayer
        layer = ExecutionLayer({"position": {"state_dim": 20}, "auto_load_model": False})
        ppo_state = layer._build_ppo_state(
            verified_stocks=[{"score": 0.7}, {"score": 0.3}],
            market_state=market_state,
        )
        self.assertEqual(ppo_state.shape, (1, 20))
        self.assertAlmostEqual(float(ppo_state[0, 0]), 2 / layer.max_positions)


if __name__ == "__main__":
    unittest.main()
