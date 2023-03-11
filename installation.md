# Installation Instructions
## Docker (Recommended)
- clone this repo and then run the following commands in terminal from the root directory of this repo
- create a docker image from the Dockerfile
	```bash
  CONTAINER_TAG="marl-jax-container"
  docker build -t ${CONTAINER_TAG} -f Dockerfile .
	```
- running the docker image for a specific algorithm on meltingpot
	```bash
  ACTORS="2"
  AVAIL_GPUS="0"
  EXP_NAME="test"
  ENV_NAME="meltingpot"
  MAP_NAME="prisoners_dilemma_in_the_matrix__repeated"
  ALGO="IMPALA"
  STEPS="10000"
  SEED="0"
  docker run --gpus all --name ${EXP_NAME} -v $(pwd):/root/code ${CONTAINER_TAG} \
	misc_scripts/train.py --async_distributed \
	--num_actors ${ACTORS} --available_gpus ${AVAIL_GPUS} \
	--algo_name ${ALGO} --env_name ${ENV_NAME} --map_name ${MAP_NAME} \
  --num_steps ${STEPS} --seed ${SEED}
	```

## Local
- create new conda environment
	```bash
  conda create -n jax_rl python=3.9
	```
- <details>
  <summary>[OPTIONAL] Install Cuda and CuDNN in conda if not available in system</summary>

  - ```bash
    conda install cudnn # This automatically installs cuda too
    ```
  - ```bash
    conda install -c nvidia cuda-nvcc=11.3 # match the version of cuda installed in the above step
    ```
  </details>
- install requirements
	```bash
  pip install -r requirements.txt
	```
- install lab2d
  - install using pip ([link](https://github.com/deepmind/meltingpot/))
    ```bash
    pip install https://github.com/deepmind/lab2d/releases/download/release_candidate_2022-03-24/dmlab2d-1.0-cp39-cp39-manylinux_2_31_x86_64.whl
    ```
- install meltingpot from git [link](https://github.com/deepmind/meltingpot/)
	```bash
	git clone -b main https://github.com/deepmind/meltingpot
  cd meltingpot
  git reset --hard 9d3c74e68f9b506571706dd8be89e0809a8b4744
  curl -L https://storage.googleapis.com/dm-meltingpot/meltingpot-assets-2.1.0.tar.gz \
      | tar -xz --directory=meltingpot
  pip install -e .
	```
- install acme from git [link](https://github.com/deepmind/acme/)
  ```bash
  git clone https://github.com/deepmind/acme
  cd acme
  git reset --hard 4525ade7015c46f33556e18d76a8d542b916f264
  pip install -e .
  ```
- Additional step is using conda environment

  ```bash
  # conda activate ENVNAME
  # cd $CONDA_PREFIX
  mkdir -p $CONDA_PREFIX/etc/conda/activate.d
  mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
  echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
  echo 'unset LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
  ```
- Run IMPALA on `prisoners_dilemma_in_the_matrix__repeated`
	```bash
	CUDA_VISIBLE_DEVICES=-1 python misc_scripts/train.py --async_distributed --available_gpus 0 --num_actors 2 \
  --algo_name IMPALA --env_name meltingpot --map_name prisoners_dilemma_in_the_matrix__repeated
	```
