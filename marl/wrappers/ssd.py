"""Wrapper class for sequential social dilemma environments to be used as a dm_env environment."""

import datetime
import os
from typing import Any

from acme import specs
from acme import types
import cv2
import dm_env
from natsort import natsorted
import numpy as np

from marl.wrappers.ssd_envs.map_env import MapEnv


class SSDWrapper(dm_env.Environment):

  def __init__(self,
               env: MapEnv,
               reward_scale=1.0,
               max_env_steps=1000,
               record=False) -> None:
    self.env = env
    self.reward_scale = reward_scale
    self.max_env_steps = max_env_steps
    self.num_agents = self.env.num_agents
    self.num_actions = self.env.action_space.n
    self.agents = list(range(self.num_agents))
    self.agent_ids = natsorted(self.env.agents.keys())
    self._reset_next_step = True
    self._record = record
    self.min_side = 720
    self.file_number = 0
    self.file_path = None
    self.data_dir = "./recordings/ssd/" + str(
        datetime.datetime.now()).split(".")[0] + "/"
    if self._record and not os.path.exists(self.data_dir):
      os.makedirs(self.data_dir)
    self.cap = None

  def _resize(self, frame):
    h, w, _ = frame.shape
    if h < w:
      new_h = self.min_side
      new_w = int(w * new_h / h)
    else:
      new_w = self.min_side
      new_h = int(h * new_w / w)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

  def _record_step(self) -> None:
    frame = self.env.full_map_to_colors().astype(np.uint8)
    frame = self._resize(frame)
    frame = cv2.cvtColor(
        cv2.flip(cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE), 1),
        cv2.COLOR_BGR2RGB)
    if self.cap is None:
      self.cap = cv2.VideoWriter(self.file_path,
                                 cv2.VideoWriter_fourcc(*'mp4v'), 3,
                                 (frame.shape[1], frame.shape[0]))
    self.cap.write(frame)

  def _get_observation(self, orig_observation) -> list[types.NestedArray]:
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
    if self._record:
      self.file_number += 1
      self.file_path = self.data_dir + str(self.file_number) + ".mp4"
      if self.cap:
        self.cap.release()
        self.cap = None
      self._record_step()
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
    if self._record:
      self._record_step()
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

  def observation_spec(self) -> list[specs.Array]:
    obs_spec = [{
        "agent_obs":
            specs.Array(
                shape=self.env.observation_space["curr_obs"].shape,
                dtype=np.float32,
                name='observation')
    }] * self.num_agents
    return obs_spec

  def action_spec(self) -> list[specs.DiscreteArray]:
    act_spec = [specs.DiscreteArray(self.num_actions, name='action')
               ] * self.num_agents
    return act_spec

  def reward_spec(self) -> list[specs.Array]:
    reward_spec = [specs.Array(shape=(), dtype=np.float32, name='reward')
                  ] * self.num_agents
    return reward_spec

  def discount_spec(self) -> list[specs.Array]:
    disc_spec = [specs.Array(shape=(), dtype=np.float32, name='discount')
                ] * self.num_agents
    return disc_spec

  def extras_spec(self) -> list[Any]:
    return list()
