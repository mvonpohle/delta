#!/bin/bash -i

#source /home/mvonpohl/miniconda3/etc/profile.d/conda.sh
source ${HOME}/.profile
export MODULEPATH=$MODULEPATH:/nasa/modulefiles/sles12
#module avail
module load cuda/11.0
conda activate delta
python "$@"