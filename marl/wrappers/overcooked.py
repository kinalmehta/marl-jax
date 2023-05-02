"""Wrapper class for overcooked-ai environment to be used as a dm_env environment."""

import datetime
import os
from typing import Any

from acme import specs
from acme import types
import cv2
import dm_env
import numpy as np
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
import pygame


class OverCooked(dm_env.Environment):

  def __init__(self,
               scenario_name,
               reward_scale=1.0,
               max_env_steps=400,
               record=False) -> None:

    base_mdp = OvercookedGridworld.from_layout_name(scenario_name)
    self.env = OvercookedEnv.from_mdp(base_mdp, horizon=max_env_steps)
    self.reward_scale = reward_scale
    self.num_agents = self.env.mdp.num_players
    self.num_actions = len(Action.ALL_ACTIONS)
    self.agents = list(range(self.num_agents))
    self._reset_next_step = True
    self._record = record
    self.file_path = None
    self.file_number = 0
    self.data_dir = "./recordings/overcooked/" + str(
        datetime.datetime.now()).split(".")[0] + "/"
    if self._record and not os.path.exists(self.data_dir):
      os.makedirs(self.data_dir)
    self.viz = None
    self.cap = None

  def _record_step(self) -> None:
    if self.viz is None:
      self.viz = StateVisualizer()
      self.viz.grid = self.env.mdp.terrain_mtx
    surf = self.viz.render_state(state=self.env.state, grid=None)
    frame = pygame.surfarray.array3d(surf)
    frame = cv2.cvtColor(
        cv2.flip(cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE), 1),
        cv2.COLOR_BGR2RGB)
    if self.cap is None:
      self.cap = cv2.VideoWriter(self.file_path,
                                 cv2.VideoWriter_fourcc(*'mp4v'), 3,
                                 (frame.shape[1], frame.shape[0]))
    self.cap.write(frame)

  def _get_observation(self) -> list[types.NestedArray]:
    # observation = self.env.lossless_state_encoding_mdp(self.env.state)
    observation = self.env.featurize_state_mdp(self.env.state)
    return [{"agent_obs": obs} for obs in observation]

  def reset(self) -> dm_env.TimeStep:
    self.env.reset()
    self._reset_next_step = False
    observation = self._get_observation()
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
    processed_actions = [Action.ALL_ACTIONS[action] for action in actions]
    _, reward, done, info = self.env.step(processed_actions)
    reward = [
        np.array(reward, dtype=np.float32) * self.reward_scale
        for _ in range(self.num_agents)
    ]
    observation = self._get_observation()
    if self._record:
      self._record_step()
    if done:
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
                shape=self._get_observation()[0]["agent_obs"].shape,
                dtype=np.float32,
                name='observation')
    } for _ in range(self.num_agents)]
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
