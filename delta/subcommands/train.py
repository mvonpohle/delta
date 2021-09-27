# Copyright © 2020, United States Government, as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All rights reserved.
#
# The DELTA (Deep Earth Learning, Tools, and Analysis) platform is
# licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Train a neural network.
"""

import sys
import time

#import logging
#logging.getLogger("tensorflow").setLevel(logging.DEBUG)

import tensorflow as tf
from multiprocessing import Process, Queue
import json
import os

from delta.config import config
from delta.imagery import imagery_dataset
from delta.ml.train import train
from delta.ml.config_parser import config_model
from delta.ml.io import save_model, load_model

#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

def main(options, worker_id):
#     print('PRINTTTTT', options)
#     aaa
#     options = {'remote': False, 'autoencoder': True}
    
    images = config.dataset.images()
    if not images:
        print('No images specified.', file=sys.stderr)
        return 1

    img = images.load(0)
    model = config_model(img.num_bands())
    if options.resume is not None:
        temp_model = load_model(options.resume)
    else:
        # this one is not built with proper scope, just used to get input and output shapes
        temp_model = model()

    start_time = time.time()
    tile_size = config.io.tile_size()
    tile_overlap = None
    stride = config.train.spec().stride

    # compute input and output sizes
    if temp_model.input_shape[1] is None:
        in_shape = None
        out_shape = temp_model.compute_output_shape((0, tile_size[0], tile_size[1], temp_model.input_shape[3]))
        out_shape = out_shape[1:3]
        tile_overlap = (tile_size[0] - out_shape[0], tile_size[1] - out_shape[1])
    else:
        in_shape = temp_model.input_shape[1:3]
        out_shape = temp_model.output_shape[1:3]

    if options.autoencoder:
        ids = imagery_dataset.AutoencoderDataset(images, in_shape, tile_shape=tile_size,
                                                 tile_overlap=tile_overlap, stride=stride,
                                                 max_rand_offset=config.train.spec().max_tile_offset)
    else:
        labels = config.dataset.labels()
        if not labels:
            print('No labels specified.', file=sys.stderr)
            return 1
        ids = imagery_dataset.ImageryDataset(images, labels, out_shape, in_shape,
                                             tile_shape=tile_size, tile_overlap=tile_overlap,
                                             stride=stride, max_rand_offset=config.train.spec().max_tile_offset)

    assert temp_model.input_shape[1] == temp_model.input_shape[2], 'Must have square chunks in model.'
    assert temp_model.input_shape[3] == ids.num_bands(), 'Model takes wrong number of bands.'
    tf.keras.backend.clear_session()

    # Try to have the internal model format we use match the output model format
    internal_model_extension = '.savedmodel'
    if options.model and ('.h5' in options.model):
        internal_model_extension = '.h5'
        
#     devices = [x.name for x in tf.config.list_logical_devices('GPU')]
    os.environ['TF_CONFIG'] = json.dumps({
        'cluster': {
            'worker': ["r101i1n0:", "r101i1n2"]
        },
        'task': {'type': 'worker', 'index': worker_id}
    })
    print(f'TF_CONFIG: {os.environ["TF_CONFIG"]}')
    strategy = tf.distribute.MultiWorkerMirroredStrategy()
#     strategy = tf.distribute.MirroredStrategy(devices=devices)
    
    try:
#         NUM_WORKERS = 2  # 2
#         procs, proc_queues = [], []
#         for worker_id in range(NUM_WORKERS):
# #             procs.append(Process(target=train, args=(model, ids, config.train.spec(), options.resume, internal_model_extension, worker_id, strategy)))
#             procs.append(Process(target=train, args=(model, ids, config.train.spec(), options.resume, internal_model_extension, worker_id)))
#             proc_queues.append(Queue())
            
#         for proc in procs:
#             proc.start()
        
#         for q_i, q in enumerate(proc_queues):
#             if q_i == 0:
#                 model = q.get()
        
#         model, _ = train(model, ids, config.train.spec(), options.resume, internal_model_extension)
#         model, _ = train(model, ids, config.train.spec(), options.resume, internal_model_extension, 1, strategy)
        model, _ = train(model, ids, config.train.spec(), options.resume, internal_model_extension, worker_id, strategy)

        if options.model is not None:  #  and worker_id == 0:
            save_model(model, options.model)
    except KeyboardInterrupt:
        print('Training cancelled.')

    stop_time = time.time()
    print('Elapsed time = ', stop_time-start_time)
    return 0
