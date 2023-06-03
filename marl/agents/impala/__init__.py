""" Multi-Agent Importance-weighted actor-learner architecture (IMPALA) agent."""

from marl.agents.impala.builder import IMPALABuilder
from marl.agents.impala.builder import PopArtIMPALABuilder
from marl.agents.impala.config import IMPALAConfig
from marl.agents.impala.learning import IMPALALearner
from marl.agents.impala.learning import IMPALALearnerME
from marl.agents.impala.learning import PopArtIMPALALearner
from marl.agents.impala.learning import PopArtIMPALALearnerME
from marl.agents.impala.networks import make_network
from marl.agents.impala.networks import make_network_2
