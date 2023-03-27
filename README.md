# marl-jax
JAX library for MARL research

[Demo Video](https://youtu.be/WQVQXPIUZxk)
[Paper Link](https://arxiv.org/abs/2303.13808)

## Implemented Algorithms
- [x] Independent-IMPALA for multi-agent environments
- [x] [OPRE](https://www.deepmind.com/publications/options-as-responses-grounding-behavioural-hierarchies-in-multi-agent-rl)


## Environments supported
- [x] [Meltingpot](https://github.com/deepmind/meltingpot/)
- [x] [Overcooked](https://github.com/HumanCompatibleAI/overcooked_ai)
- [x] [Sequential Social Dilemma](https://github.com/eugenevinitsky/sequential_social_dilemma_games)

## Other Features
- [x] Distributed training (IMPALA style architecture)
  - Dynamically distribute load of multiple agents across available GPUs
  - Run multiple environment instances, one per CPU core for experience collection
- [x] Wandb and Tensorboard logging
- [x] PopArt normalization

## Help
- [Installation Instructions](installation.md)
- [Environment Details](environments.md)

## Implementation References
- [Deepmind's Acme](https://github.com/deepmind/acme/)

## Citation

If you use this code in your project, please cite the following paper:
```bibtex
@article{mehta2023marljax,
      title={marl-jax: Multi-agent Reinforcement Leaning framework for Social Generalization}, 
      author={Kinal Mehta and Anuj Mahajan and Pawan Kumar},
      year={2023},
      journal={arXiv preprint arXiv:2303.13808},
      url={https://arxiv.org/abs/2303.13808},
}
```
