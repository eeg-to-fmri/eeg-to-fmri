from tensorflow.python.keras.utils import conv_utils

import numpy as np

from utils import conv_sat_utils

def get_possible_kernel_size_conv(input_shape, output_shape, next_input_shape=None):
	return conv_sat_utils.conv_sat(input_shape, output_shape, next_input_shape=next_input_shape).solve()

def get_possible_kernel_size_deconv(input_shape, output_shape, next_input_shape=None):
	return conv_sat_utils.conv_sat(input_shape, output_shape, next_input_shape=next_input_shape).solve()
	

def add_generated_dim(generated, input_shape, output_shape, func):
	possible = func(input_shape, output_shape)

	pos = list(range(len(possible)))
	generated_kernel_stride = possible[np.random.choice(pos)]
	
	generated['kernel'] += (generated_kernel_stride[0],)
	generated['stride'] += (generated_kernel_stride[1],)

	return generated