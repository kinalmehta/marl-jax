""" Multi-Agent Importance-weighted actor-learner architecture (IMPALA) agent."""

from marl.agents.impala.builder import IMPALABuilder, PopArtIMPALABuilder
from marl.agents.impala.config import IMPALAConfig
from marl.agents.impala.learning import IMPALALearner, PopArtIMPALALearner
from marl.agents.impala.networks import make_network, make_network_2
