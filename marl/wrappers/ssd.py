"""Wrapper class for sequential social dilemma environments to be used as a dm_env environment."""

from typing import Any, List

import dm_env
import numpy as np
from acme import specs, types
from natsort import natsorted

from marl.wrappers.ssd_envs.map_env import MapEnv


class SSDWrapper(dm_env.Environment):

  def __init__(self, env: MapEnv, reward_scale=1.0, max_env_steps=1000) -> None:
    self.env = env
    self.reward_scale = reward_scale
    self.max_env_steps = max_env_steps
    self.num_agents = self.env.num_agents
    self.num_actions = self.env.action_space.n
    self.agents = list(range(self.num_agents))
    self.agent_ids = natsorted(self.env.agents.keys())
    self._reset_next_step = True

  def _get_observation(self, orig_observation) -> List[types.NestedArray]:
    # observation = self.env.lossless_state_encoding_mdp(self.env.state)
    return [{
        "agent_obs":
            np.array(orig_observation[agent_id]["curr_obs"], dtype=np.float32)
    } for agent_id in self.agent_ids]

  def reset(self) -> dm_env.TimeStep:
    self.current_step = 0
    orig_observation = self.env.reset()
    observation = self._get_observation(orig_observation)

    ts = dm_env.restart(observation)
    rew = [np.array(0., dtype=np.float32) for _ in range(self.num_agents)]
    discount = [np.array(1., dtype=np.float32) for _ in range(self.num_agents)]
    ts = ts._replace(reward=rew)
    ts = ts._replace(discount=discount)
    return ts

  def step(self, actions: types.NestedArray) -> dm_env.TimeStep:
    processed_actions = {
        agent_id: int(action)
        for agent_id, action in zip(self.agent_ids, actions)
    }
    orig_observation, orig_reward, done, info = self.env.step(processed_actions)
    reward = [
        np.array(orig_reward[agent_id], dtype=np.float32) * self.reward_scale
        for agent_id in self.agent_ids
    ]
    observation = self._get_observation(orig_observation)
    self.current_step += 1
    if self.current_step == self.max_env_steps:
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
    done = not self.agents or self._env_done
    return done

  def observation_spec(self) -> List[specs.Array]:
    obs_spec = [{
        "agent_obs":
            specs.Array(
                shape=self.env.observation_space["curr_obs"].shape,
                dtype=np.float32,
                name='observation')
    }] * self.num_agents
    return obs_spec

  def action_spec(self) -> List[specs.DiscreteArray]:
    act_spec = [specs.DiscreteArray(self.num_actions, name='action')
               ] * self.num_agents
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
