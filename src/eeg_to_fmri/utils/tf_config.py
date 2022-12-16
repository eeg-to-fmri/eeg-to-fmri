import tensorflow as tf

import tensorflow_probability as tfp

import os

import numpy as np

import random

MAX_N_THREADS=0

def setup_tensorflow(memory_limit, device="CPU", run_eagerly=False, set_primary_memory=False, set_tf_threads=False):
	gpu = tf.config.experimental.list_physical_devices(device)[0]
	tf.config.set_soft_device_placement(True)
	tf.config.log_device_placement=True
	if(device=="GPU"):
		tf.config.experimental.set_memory_growth(gpu, True)
		tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])

	if(run_eagerly):
		tf.config.run_functions_eagerly(True)

	#set number of threads
	if(set_tf_threads):
		tf.config.experimental.set_synchronous_execution(True)
		tf.config.threading.set_inter_op_parallelism_threads(MAX_N_THREADS)
		tf.config.threading.set_intra_op_parallelism_threads(MAX_N_THREADS)

def set_seed(seed=42, enable_gpu_determinism=False):
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)
	if(enable_gpu_determinism):
		tf.config.experimental.enable_op_determinism()
