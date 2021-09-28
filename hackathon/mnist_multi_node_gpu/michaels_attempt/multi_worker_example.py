import os
# os.environ.pop('TF_CONFIG', None)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
try:
    print(os.environ["TF_CONFIG"])
except:
    print("Couldn't get TF_CONFIG")
print(os.environ["CUDA_VISIBLE_DEVICES"])

import tensorflow as tf
import mnist

# from mpi4py import MPI
# import time
# import portpicker
# import argparse
# import pdb


# get worker index
# parser = argparse.ArgumentParser()
# parser.add_argument('--job_idx', type=int, help='Job index', default=0)
# args = parser.parse_args()
#
# worker_idx = MPI.COMM_WORLD.rank + args.job_idx
# size = MPI.COMM_WORLD.size

# print(f'Worker = {worker_idx} (Total number of MPI ranks {size - 1})')
# # print(f'[{worker_idx}] TF_CONFIG original value: {os.environ["TF_CONFIG"]}')
#
# if worker_idx != 0:  # wait if not worker 0 ('chief')
#     time.sleep(10)
#
# # load cluster general cluster configuration
# if worker_idx == 0:
#     with open('/nobackupnfs2/mvonpohl/delta_gpu_project/project_folders/msaragoc/test_multiworker_training_keras/cluster.json', 'r') as json_file:
#         tf_config = json.load(json_file)
#     # set ports for each node in the cluster
#     tf_config['cluster']['worker'] = [f'{worker}:{portpicker.pick_unused_port()}' for worker in tf_config['cluster']['worker']]
#     with open('/nobackupnfs2/mvonpohl/delta_gpu_project/project_folders/msaragoc/test_multiworker_training_keras/cluster_with_ports.json', 'w') as json_file:
#         json.dump(tf_config, json_file)
# else:
#     with open('/nobackupnfs2/mvonpohl/delta_gpu_project/project_folders/msaragoc/test_multiworker_training_keras/cluster_with_ports.json', 'r') as json_file:
#         tf_config = json.load(json_file)
#
# # set configuration for this particular worker
# tf_config['task']['index'] = worker_idx
#
# # serialize the cluster configuration into the environment variable `TF_CONFIG`
# os.environ['TF_CONFIG'] = json.dumps(tf_config)

# per_worker_batch_size = 64
# # tf_config = json.loads(os.environ['TF_CONFIG'])
# # pdb.set_trace()
# print(f'TF_CONFIG: {tf_config}/{os.environ["TF_CONFIG"]}')
#
# num_workers = len(tf_config['cluster']['worker'])

strategy = tf.distribute.MultiWorkerMirroredStrategy()

# global_batch_size = per_worker_batch_size * num_workers
global_batch_size = 128
multi_worker_dataset = mnist.mnist_dataset(global_batch_size)

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = mnist.build_and_compile_cnn_model()

multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)
