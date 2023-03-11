import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)

import functools

import dmlab2d
import numpy as np
from acme.wrappers import SinglePrecisionWrapper
from meltingpot.python import scenario, substrate

from marl.wrappers import (
    AutoResetWrapper,
    HierarchyVecWrapper,
    MeltingPotWrapper,
    MergeWrapper,
    ObservationActionRewardWrapper,
    OverCooked,
    RockPaperScissors,
    SSDWrapper,
    all_observations_wrapper,
    default_observation_wrapper,
)
from marl.wrappers.ssd_envs.env_creator import get_env_creator


def node_allocation(num_agents, available_gpus):
  available_gpus = available_gpus.split(",")
  if len(available_gpus) > num_agents:
    pass
  #   resource_dict = {"learner": ",".join(available_gpus[:num_agents])}
  #   possible_gpu_actors = len(available_gpus) - num_agents
  #   gpu_actors = []
  #   for i in range(possible_gpu_actors):
  #     resource_dict[f"gpu_actor_{i}"] = available_gpus[num_agents + i]
  #     gpu_actors.append(f"gpu_actor_{i}")
  #   return resource_dict, gpu_actors
  return {"learner": ",".join(available_gpus)}, []


def make_meltingpot_environment(
    seed: int,
    substrate_name: str,
    *,
    autoreset: bool = False,
    shared_reward: bool = False,
    reward_scale: float = 1.0,
    global_observation_sharing: bool = False,
    diversity_dim: int = None) -> dmlab2d.Environment:
  """Returns a MeltingPot environment."""
  env_config = substrate.get_config(substrate_name)
  env = substrate.build(substrate_name, roles=env_config.default_player_roles)
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
                             diversity_dim: int = None) -> dmlab2d.Environment:
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
                diversity_dim: int = None):
  if env_name[-1].isdigit():
    final_env = make_meltingpot_scenario(
        seed,
        env_name,
        autoreset=autoreset,
        shared_reward=shared_reward,
        reward_scale=reward_scale,
        global_observation_sharing=shared_obs)
  else:
    final_env = make_meltingpot_environment(
        seed,
        env_name,
        autoreset=autoreset,
        shared_reward=shared_reward,
        reward_scale=reward_scale,
        global_observation_sharing=shared_obs,
        diversity_dim=diversity_dim)
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


def make_rps_environment(seed: int,
                         *,
                         autoreset: bool = False,
                         reward_scale: float = 1.0,
                         global_observation_sharing: bool = False,
                         diversity_dim: int = None) -> dmlab2d.Environment:
  env = RockPaperScissors(reward_scale=reward_scale)
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
  env = MergeWrapper(env)
  if autoreset:
    env = AutoResetWrapper(env)
  return env


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
                         diversity_dim: int = None) -> dmlab2d.Environment:
  """Returns an Overcooked environment."""
  env = get_env_creator(
      map_name, num_agents=num_agents, use_collective_reward=team_reward)(
          seed)
  env = SSDWrapper(env, reward_scale=reward_scale)
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
