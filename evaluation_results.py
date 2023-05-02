import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd

FLAGS = flags.FLAGS
flags.DEFINE_string('results_dir', "./results/",
                    'Directory containing results.')
flags.DEFINE_string('algos_to_plot', "IMPALA,OPRE",
                    'Comma-separated list of algorithms to plot.')
flags.DEFINE_string(
    'env_name',
    None,
    "Environment to generate plots for ({env_name}_{map_name} for meltingpot and overcooked).",
    required=True)


def process_df(df):
  cols = df.columns
  res = []
  for col in cols:
    if "episode_return" in col:
      res.append(np.mean(df[col]))
  return np.mean(res)
  # return np.array(res)


def get_hypothesis(algo_name, experiments):
  skip_files = {"actor.csv", "learner.csv"}
  runs = {}
  name = "Scenario "
  idx = 0
  experiments = list(filter(lambda x: algo_name in x, experiments))
  for exp in experiments:
    files = set(os.listdir(exp)) - skip_files
    for idx, cur_file in enumerate(sorted(files)):
      file_name = os.path.join(exp, cur_file)
      df = pd.read_csv(file_name)
      env_name = name + str(idx - 1) if idx > 0 else "Substrate"
      if not env_name in runs:
        runs[env_name] = process_df(df)
      else:
        runs[env_name] += process_df(df)
      idx += 1
  if not runs:
    logging.warning(f"No runs found for {algo_name}")
    return None
  for k, v in runs.items():
    runs[k] = v / len(experiments)
  return runs


def main(_):
  start_dir = FLAGS.results_dir
  experiments = [
      os.path.join(start_dir, data, "csv_logs")
      for data in os.listdir(start_dir)
      if os.path.isdir(os.path.join(start_dir, data))
  ]

  algos = FLAGS.algos_to_plot.split(",")
  results = dict()
  for algo in algos[:]:
    h = get_hypothesis(algo, experiments)
    if h:
      results[algo] = h

  final_df = pd.DataFrame(results)
  print(final_df)


if __name__ == "__main__":
  app.run(main)
