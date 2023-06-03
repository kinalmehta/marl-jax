import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

import datetime
import functools
from functools import partial

from acme.wrappers import SinglePrecisionWrapper
import cv2
import dm_env
import dmlab2d
from meltingpot.python import scenario
from meltingpot.python import substrate
import numpy as np

from marl.wrappers import all_observations_wrapper
from marl.wrappers import AutoResetWrapper
from marl.wrappers import default_observation_wrapper
from marl.wrappers import HierarchyVecWrapper
from marl.wrappers import MeltingPotWrapper
from marl.wrappers import MergeWrapper
from marl.wrappers import ObservationActionRewardWrapper
from marl.wrappers import OverCooked
from marl.wrappers import SSDWrapper
from marl.wrappers.ssd_envs.env_creator import get_env_creator


def node_allocation(available_gpus, use_inference_server):
  available_gpus = available_gpus.split(",")
  # if len(available_gpus) > num_agents:
  #   pass
  #   resource_dict = {"learner": ",".join(available_gpus[:num_agents])}
  #   possible_gpu_actors = len(available_gpus) - num_agents
  #   gpu_actors = []
  #   for i in range(possible_gpu_actors):
  #     resource_dict[f"gpu_actor_{i}"] = available_gpus[num_agents + i]
  #     gpu_actors.append(f"gpu_actor_{i}")
  #   return resource_dict, gpu_actors
  if len(available_gpus) == 1 or not use_inference_server:
    return {"learner": ",".join(available_gpus)}
  return {
      "learner": ",".join(available_gpus[:-1]),
      "inference_server": available_gpus[-1]
  }  # inference_server


def make_meltingpot_environment(seed: int,
                                substrate_name: str,
                                *,
                                autoreset: bool = False,
                                shared_reward: bool = False,
                                reward_scale: float = 1.0,
                                global_observation_sharing: bool = False,
                                diversity_dim: int = None,
                                record: bool = False) -> dmlab2d.Environment:
  """Returns a MeltingPot environment."""
  env_config = substrate.get_config(substrate_name)
  env = substrate.build(substrate_name, roles=env_config.default_player_roles)
  if record:
    vid_rec = partial(
        my_render_func_efficient,
        recorder=MeltingPotRecorder(f"{substrate_name}"))
    env.observables().timestep.subscribe(vid_rec)
  for missing_obs in ["INVENTORY", "READY_TO_SHOOT"]:
    if missing_obs not in env.observation_spec()[0]:
      env = default_observation_wrapper.Wrapper(
          env, key=missing_obs, default_value=np.zeros([1]))
  if global_observation_sharing:
    env = all_observations_wrapper.Wrapper(
        env,
        observations_to_share=["INVENTORY", "READY_TO_SHOOT", "RGB"],
        share_actions=True,
        share_rewards=True)
  env = MeltingPotWrapper(
      env, shared_reward=shared_reward, reward_scale=reward_scale)
  env = SinglePrecisionWrapper(env)
  env = ObservationActionRewardWrapper(env)
  if diversity_dim is not None:
    env = HierarchyVecWrapper(env, diversity_dim=diversity_dim, seed=seed)
  if autoreset:
    env = AutoResetWrapper(env)
  env = MergeWrapper(env)
  return env


def make_meltingpot_scenario(seed: int,
                             scenario_name: str,
                             *,
                             autoreset: bool = False,
                             shared_reward: bool = False,
                             reward_scale: float = 1.0,
                             global_observation_sharing: bool = False,
                             diversity_dim: int = None,
                             record: bool = False) -> dmlab2d.Environment:
  """Returns a `MeltingPotWrapper` environment."""

  def transform_substrate(env, wrappers):
    for wrapper in wrappers:
      env = wrapper(env)
    return env

  wrappers = []
  temp_env = scenario.build(scenario_name)
  for missing_obs in ["INVENTORY", "READY_TO_SHOOT"]:
    if missing_obs not in temp_env.observation_spec()[0]:
      wrappers.append(
          functools.partial(
              default_observation_wrapper.Wrapper,
              key=missing_obs,
              default_value=np.zeros([1])))
  del temp_env
  if global_observation_sharing:
    wrappers.append(
        functools.partial(
            all_observations_wrapper.Wrapper,
            observations_to_share=["INVENTORY", "READY_TO_SHOOT", "RGB"],
            share_actions=True,
            share_rewards=True))

  substrate_transform = (lambda env: transform_substrate(env, wrappers)
                        ) if len(wrappers) > 0 else None
  env = scenario.build(scenario_name, substrate_transform=substrate_transform)
  if record:
    vid_rec = partial(
        my_render_func_efficient,
        recorder=MeltingPotRecorder(f"{scenario_name}"))
    env.observables().substrate.timestep.subscribe(vid_rec)
  env = MeltingPotWrapper(
      env, shared_reward=shared_reward, reward_scale=reward_scale)
  env = SinglePrecisionWrapper(env)
  env = ObservationActionRewardWrapper(env)
  if diversity_dim is not None:
    env = HierarchyVecWrapper(env, diversity_dim=diversity_dim, seed=seed)
  env = MergeWrapper(env)
  if autoreset:
    env = AutoResetWrapper(env)
  return env


def env_factory(seed: int,
                env_name,
                *,
                autoreset: bool = False,
                shared_reward: bool = False,
                reward_scale: float = 1.0,
                shared_obs: bool = False,
                diversity_dim: int = None,
                record: bool = False):
  if env_name[-1].isdigit():
    final_env = make_meltingpot_scenario(
        seed,
        env_name,
        autoreset=autoreset,
        shared_reward=shared_reward,
        reward_scale=reward_scale,
        global_observation_sharing=shared_obs,
        record=record)
  else:
    final_env = make_meltingpot_environment(
        seed,
        env_name,
        autoreset=autoreset,
        shared_reward=shared_reward,
        reward_scale=reward_scale,
        global_observation_sharing=shared_obs,
        diversity_dim=diversity_dim,
        record=record)
  return final_env


def eval_env_factories(env_name, reward_scale: float = 1.0):
  eval_envs = []
  if env_name[-1].isdigit():
    eval_envs.append(lambda seed: make_meltingpot_scenario(
        seed, env_name, reward_scale=reward_scale))
  else:
    scenarios = scenario.SCENARIOS_BY_SUBSTRATE[env_name]
    for scene in scenarios:
      eval_envs.append(lambda seed: make_meltingpot_scenario(
          seed, scene, reward_scale=reward_scale))
  return eval_envs


def make_overcooked_environment(seed: int,
                                map_name: str,
                                *,
                                autoreset: bool = False,
                                reward_scale: float = 1.0,
                                global_observation_sharing: bool = False,
                                diversity_dim: int = None,
                                record: bool = False) -> dmlab2d.Environment:
  """Returns an Overcooked environment."""
  env = OverCooked(
      map_name, reward_scale=reward_scale, max_env_steps=400, record=record)
  env = SinglePrecisionWrapper(env)
  if global_observation_sharing:
    env = all_observations_wrapper.AllObservationWrapper(
        env,
        observations_to_share=["agent_obs"],
        share_actions=True,
        share_rewards=True)
  env = ObservationActionRewardWrapper(env)
  if diversity_dim is not None:
    env = HierarchyVecWrapper(env, diversity_dim=diversity_dim, seed=seed)
  if autoreset:
    env = AutoResetWrapper(env)
  env = MergeWrapper(env)
  return env


def make_ssd_environment(seed: int,
                         map_name: str,
                         *,
                         autoreset: bool = False,
                         reward_scale: float = 1.0,
                         team_reward: bool = False,
                         num_agents: int = 2,
                         global_observation_sharing: bool = False,
                         diversity_dim: int = None,
                         record: bool = False) -> dmlab2d.Environment:
  """Returns an Overcooked environment."""
  env = get_env_creator(
      map_name, num_agents=num_agents, use_collective_reward=team_reward)(
          seed)
  env = SSDWrapper(env, reward_scale=reward_scale, record=record)
  env = SinglePrecisionWrapper(env)
  if global_observation_sharing:
    env = all_observations_wrapper.AllObservationWrapper(
        env,
        observations_to_share=["agent_obs"],
        share_actions=True,
        share_rewards=True)
  env = ObservationActionRewardWrapper(env)
  if diversity_dim is not None:
    env = HierarchyVecWrapper(env, diversity_dim=diversity_dim, seed=seed)
  if autoreset:
    env = AutoResetWrapper(env)
  env = MergeWrapper(env)
  return env


class MeltingPotRecorder:

  def __init__(self, substrate_name="") -> None:
    self.cap = None
    self.min_side = 720
    self.file_number = 0
    self.file_path = None
    self.data_dir = f"./recordings/meltingpot/{substrate_name}/{str(datetime.datetime.now()).split('.')[0]}/"
    if not os.path.exists(self.data_dir):
      os.makedirs(self.data_dir)

  def resize(self, frame):
    h, w, _ = frame.shape
    if h < w:
      new_h = self.min_side
      new_w = int(w * new_h / h)
    else:
      new_w = self.min_side
      new_h = int(h * new_w / w)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

  def record(self, frame):
    frame = self.resize(frame)
    if self.cap is None:
      self.cap = cv2.VideoWriter(self.file_path,
                                 cv2.VideoWriter_fourcc(*'mp4v'), 3,
                                 (frame.shape[1], frame.shape[0]))
    self.cap.write(frame)

  def reset(self):
    self.file_number += 1
    self.file_path = self.data_dir + str(self.file_number) + ".mp4"
    if self.cap:
      self.cap.release()
      self.cap = None

  def __del__(self):
    if self.cap:
      self.cap.release()


def my_render_func_efficient(timestep: dm_env.TimeStep, recorder):
  if timestep.first():
    recorder.reset()
  obs = timestep.observation[0]
  img = obs['WORLD.RGB']
  recorder.record(img)
