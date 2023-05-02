from absl import flags
from launchpad.nodes.python.local_multi_processing import PythonProcess

FLAGS = flags.FLAGS


def to_device(program_nodes: list,
              nodes_on_gpu: dict = {"learner": [0]}) -> dict:
  """Specifies which nodes should run on gpu.

    If nodes_on_gpu is an empty list, this returns a cpu only config.

    Args:
      program_nodes (List): nodes in lp program.
      nodes_on_gpu (List, optional): nodes to run on gpu. Defaults to ["trainer"].

    Returns:
      Dict: dict with cpu only lp config.

    Example:
      local_resources = to_device(
          program_nodes=lp_program.groups.keys(),
          nodes_on_gpu=["learner"]
        )
      lp.launch(
          lp_program,
          launch_type = lp.LaunchType.LOCAL_MULTI_PROCESSING,
          local_resources = local_resources
        )

    """
  return {
      node: PythonProcess(env={"CUDA_VISIBLE_DEVICES": str(-1)}) if
      (node not in nodes_on_gpu.keys())
      # use below to allocate specific GPUs/TPUs/Accelerators to specific node groups
      # else PythonProcess(env={"CUDA_VISIBLE_DEVICES": ",".join(list(map(str,nodes_on_gpu[node])))})
      else PythonProcess(env={"CUDA_VISIBLE_DEVICES": nodes_on_gpu[node]})
      for node in program_nodes
  }
