from marl.wrappers.ssd_envs.cleanup import CleanupEnv
from marl.wrappers.ssd_envs.harvest import HarvestEnv
from marl.wrappers.ssd_envs.switch import SwitchEnv


def get_env_creator(
    env,
    num_agents,
    use_collective_reward=False,
    inequity_averse_reward=False,
    alpha=0.0,
    beta=0.0,
    num_switches=6,
):
  if env == "harvest":

    def env_creator(_):
      return HarvestEnv(
          num_agents=num_agents,
          return_agent_actions=True,
          use_collective_reward=use_collective_reward,
          inequity_averse_reward=inequity_averse_reward,
          alpha=alpha,
          beta=beta,
      )

  elif env == "cleanup":

    def env_creator(_):
      return CleanupEnv(
          num_agents=num_agents,
          return_agent_actions=True,
          use_collective_reward=use_collective_reward,
          inequity_averse_reward=inequity_averse_reward,
          alpha=alpha,
          beta=beta,
      )

  elif env == "switch":

    def env_creator(_):
      return SwitchEnv(num_agents=num_agents, num_switches=num_switches)

  else:
    raise ValueError(f"env must be one of harvest, cleanup, switch, not {env}")

  return env_creator
