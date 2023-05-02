from typing import Any, Union

from acme.specs import EnvironmentSpec
import dm_env
from natsort import natsorted


class MAEnvironmentSpec:

  def __init__(
      self,
      environment: dm_env.Environment,
      agent_environment_specs: list[EnvironmentSpec] = None,
      extras_specs: dict[str, Any] = None,
  ):
    """Multi-agent environment spec

        Create a multi-agent environment spec through a dm_env Environment
        or through pre-existing environment specs (specifying an environment
        spec for each agent) and extras specs

        Args:
            environment : dm_env.Environment object
            agent_environment_specs : environment specs for each agent
            extras_specs : extras specs for additional data not contained
            in the acme EnvironmentSpec format, such as global state information
        """
    if not agent_environment_specs:
      agent_environment_specs = self._make_ma_environment_spec(environment)
    else:
      self._extras_specs = extras_specs
    self._keys = list(natsorted(environment.agents))
    self._agent_environment_specs = agent_environment_specs
    self.num_agents = environment.num_agents

  def _make_ma_environment_spec(
      self, environment: dm_env.Environment) -> EnvironmentSpec:
    """Create a multi-agent environment spec from a dm_env environment

        Args:
            environment : dm_env.Environment
        Returns:
            Dictionary with an environment spec for each agent
        """
    self._extras_specs = environment.extras_spec()
    self._single_agent_environment_specs = EnvironmentSpec(
        observations=environment.single_observation_spec(),
        actions=environment.single_action_spec(),
        rewards=environment.single_reward_spec(),
        discounts=environment.single_discount_spec(),
    )
    return EnvironmentSpec(
        observations=environment.observation_spec(),
        actions=environment.action_spec(),
        rewards=environment.reward_spec(),
        discounts=environment.discount_spec(),
    )

  def get_extras_specs(self) -> list[dict[str, Any]]:
    """Get extras specs
        Returns:
            Extras spec that contains any additional information not contained
            within the environment specs
        """
    return self._extras_specs  # type: ignore

  def get_agent_environment_specs(self) -> EnvironmentSpec:
    """Get environment specs for all agents
        Returns:
            Dictionary of environment specs, representing each agent in the environment
        """
    return self._agent_environment_specs

  def set_extras_specs(self, extras_specs: list[dict[str, Any]]) -> None:
    """Set extras specs
        Returns:
            None
        """
    self._extras_specs = extras_specs

  def get_agent_ids(self) -> list[Union[int, str]]:
    return self._keys

  def get_single_agent_environment_specs(self) -> EnvironmentSpec:
    return self._single_agent_environment_specs
