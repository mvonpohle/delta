#!/bin/bash
# run  this on pleiades front  end
# Needs conda installed first - https://docs.conda.io/en/latest/miniconda.html#linux-installers

# download command: wget https://github.com/mvonpohle/delta/raw/gpu_hackathon/hackathon/conda_env_setup.sh && chmod u+x conda_env_setup.sh && source ./conda_env_setup.sh

mkdir /nobackupnfs2/mvonpohl/delta_gpu_project/project_folders/${USER}/
cd /nobackupnfs2/mvonpohl/delta_gpu_project/project_folders/${USER}/
cp --recursive /nobackupnfs2/mvonpohl/delta_gpu_project/models /nobackupnfs2/mvonpohl/delta_gpu_project/project_folders/${USER}/
git clone -b gpu_hackathon https://github.com/mvonpohle/delta.git
cd delta

module load cuda/11.0
conda create --name delta --yes "python<3.9" gdal "cudnn=>8.0"
conda activate delta
conda install --yes pylint ipython jupyterlab pip # development addons
pip install -e . # inside delta root folder | development option

