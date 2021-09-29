# module load cuda/11.0

# echo $MPI_COMM_WORLD
# echo 'asdas'
# if [ "$HOSTNAME" = "r101i2n15" ]; then
#     echo $HOSTNAME 'same'
#     TF_CONFIG='{"cluster": {"worker": ["r101i2n15:12345", "r101i2n16:23456"]}, "task": {"type": "worker", "index": 0} }'
# else
#     echo $HOSTNAME 'different'
#     TF_CONFIG='{"cluster": {"worker": ["r101i2n15:12345", "r101i2n16:23456"]}, "task": {"type": "worker", "index": 1} }'
# fi

# export TF_CONFIG
# echo $HOSTNAME $TF_CONFIG
# export CUDA_VISIBLE_DEVICES='0,1,2,3'

HOSTFILE=hostfile.txt
NUM_WORKERS=2  # should be equal to number of hosts; each host is a worker

mpiexec -np $NUM_WORKERS --hostfile $HOSTFILE python multi_worker_example.py