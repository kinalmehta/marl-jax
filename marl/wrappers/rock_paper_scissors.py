"""Simple MARL environment to be used as a dm_env environment."""

from typing import Any, List

import dm_env
import numpy as np
from acme import specs, types

EPS = 1e-6


class RockPaperScissors(dm_env.Environment):

  def __init__(self, reward_scale=1.0, max_env_steps=1000):
    self.reward_scale = reward_scale
    self._reset_next_step = True
    self.is_turn_based = False
    self.num_agents = 2
    self.num_actions = 4
    self.agents = list(range(self.num_agents))
    self.max_env_steps = max_env_steps
    self.play_every_n_steps = 100
    # environment_states
    self._reset_global_state()
    # states = [rock_objects, paper_objects, scissors_objects]
    # actions = [rock, paper, scissors, noop]
    self.reward_matrix = 100 * np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]],
                                        dtype=np.float32)

  def _reset_global_state(self):
    self.global_inventory = np.array([1000, 1000, 1000], dtype=np.float32)
    stochastic_rocks = np.random.randint(0, 1000)
    stochastic_papers = np.random.randint(0, 1000 - stochastic_rocks)
    stochastic_scissors = 1000 - stochastic_rocks - stochastic_papers

    self.global_inventory[0] += stochastic_rocks
    self.global_inventory[1] += stochastic_papers
    self.global_inventory[2] += stochastic_scissors

    self._reset_inventories()
    self.env_steps = 0

  def _reset_inventories(self):
    self.agent_observations = [
        np.zeros_like(self.global_inventory, dtype=np.float32)
        for _ in range(self.num_agents)
    ]

  def _process_observation(self) -> List[types.NestedArray]:
    return [{
        "agent_obs":
            np.concatenate(
                (self.agent_observations[i], self.global_inventory)) / 500.
    } for i in range(self.num_agents)]

  def reset(self) -> dm_env.TimeStep:
    self._reset_next_step = False
    self._reset_global_state()

    observation = self._process_observation()
    ts = dm_env.restart(observation)
    rew = [np.array(0., dtype=np.float32) for _ in range(self.num_agents)]
    discount = [np.array(1., dtype=np.float32) for _ in range(self.num_agents)]
    ts = ts._replace(reward=rew)
    ts = ts._replace(discount=discount)
    return ts

  def step(self, actions: types.NestedArray) -> dm_env.TimeStep:
    """Steps the environment."""
    if self._reset_next_step:
      return self.reset()

    self.env_steps += 1
    reward = [0., 0.]

    # compute reward by playing out inventories if its time to play
    # reset the inventories and assign rewards to agents
    if self.env_steps % self.play_every_n_steps == 0:
      ob1 = self.agent_observations[0]
      ob2 = self.agent_observations[1]
      ob1 /= (np.linalg.norm(ob1, ord=2) + EPS)
      ob2 /= (np.linalg.norm(ob2, ord=2) + EPS)
      rw1 = ob1 @ self.reward_matrix @ ob2
      reward = [rw1, -rw1]
      self._reset_inventories()

    reward = [
        np.array(rew, dtype=np.float32) * self.reward_scale for rew in reward
    ]
    # update inventories based on actions
    for agent_idx, action in enumerate(actions):
      if action != 3 and self.global_inventory[action] > 0:
        self.agent_observations[agent_idx][action] += 1
        self.global_inventory[action] -= 1
      # action==3 is a noop action and does not update inventories

    observation = self._process_observation()
    if self.env_steps == self.max_env_steps:
      self._reset_next_step = True
      self._env_done = True
      ts = dm_env.termination(reward=reward, observation=observation)
      ts = ts._replace(discount=[
          np.array(0., dtype=np.float32) for _ in range(self.num_agents)
      ])
      return ts
    return dm_env.transition(
        reward=reward,
        observation=observation,
        discount=[
            np.array(1., dtype=np.float32) for _ in range(self.num_agents)
        ])

  def env_done(self) -> bool:
    """Check if env is done.
    Returns:
        bool: bool indicating if env is done.
    """
    done = not self.agents or self._env_done
    return done

  def observation_spec(self) -> List[specs.Array]:
    obs_spec = [{
        "agent_obs":
            specs.Array(shape=(6,), dtype=np.float32, name='observation')
    }] * self.num_agents
    return obs_spec

  def action_spec(self) -> List[specs.DiscreteArray]:
    act_spec = [specs.DiscreteArray(4, name='action')] * self.num_agents
    return act_spec

  def reward_spec(self) -> List[specs.Array]:
    reward_spec = [specs.Array(shape=(), dtype=np.float32, name='reward')
                  ] * self.num_agents
    return reward_spec

  def discount_spec(self) -> List[specs.Array]:
    disc_spec = [specs.Array(shape=(), dtype=np.float32, name='discount')
                ] * self.num_agents
    return disc_spec

  def extras_spec(self) -> List[Any]:
    return list()
