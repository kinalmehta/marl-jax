
eval "$(conda shell.bash hook)"
conda create -n marl-jax python=3.9 -y

which python
conda activate marl-jax
which python

pip install -r requirements.txt


mkdir gits
cd gits


pip install https://github.com/deepmind/lab2d/releases/download/release_candidate_2022-03-24/dmlab2d-1.0-cp39-cp39-manylinux_2_31_x86_64.whl 
if [ "$?" -eq "0" ]
then
  echo "Installed prebuilt binary"
else
  echo "Prebuilt binary incompatible, Compiling from source"
  conda install -c conda-forge gcc=12.3 gxx=12.3 bazel=5.2 openjdk=8 -y
  git clone https://github.com/deepmind/lab2d.git
  cd lab2d
  git reset --hard 94e37d189aee8d309bc1dfae7227676c77636e88
  bazel build -c opt --define=lua=5_2 //dmlab2d:dmlab2d_wheel
  pip install bazel-bin/dmlab2d/dmlab2d-1.0-*.whl
  cd ..
fi


# install melting pot
git clone -b main https://github.com/deepmind/meltingpot
cd meltingpot
git reset --hard 9d3c74e68f9b506571706dd8be89e0809a8b4744
curl -L https://storage.googleapis.com/dm-meltingpot/meltingpot-assets-2.1.0.tar.gz \
    | tar -xz --directory=meltingpot
pip install -e .
cd ..


git clone https://github.com/deepmind/acme
cd acme
git reset --hard 4525ade7015c46f33556e18d76a8d542b916f264
pip install -e .
cd ..


mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'unset LD_LIBRARY_PATH' > $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
