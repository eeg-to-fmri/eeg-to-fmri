import tensorflow as tf

import tensorflow_probability as tfp

from utils import conv_utils, tfp_seed

import numpy as np

import random

class LocallyConnected3D(tf.keras.layers.Layer):

	def __init__(self,
				filters,
				kernel_size,
				strides=(1, 1, 1),
				padding='valid',
				data_format=None,
				activation=None,
				use_bias=True,
				kernel_initializer='glorot_uniform',
				bias_initializer='zeros',
				kernel_regularizer=None,
				bias_regularizer=None,
				activity_regularizer=None,
				kernel_constraint=None,
				bias_constraint=None,
				implementation=3,
				**kwargs):
		super(LocallyConnected3D, self).__init__(**kwargs)
		self.filters = filters
		self.kernel_size = conv_utils.normalize_tuple(kernel_size, 3, 'kernel_size')
		self.strides = conv_utils.normalize_tuple(strides, 3, 'strides')
		self.padding = conv_utils.normalize_padding(padding)
		if self.padding != 'valid' and implementation == 1:
			raise ValueError('Invalid border mode for LocallyConnected3D '
							'(only "valid" is supported if implementation is 1): ' +
							padding)
		self.data_format = conv_utils.normalize_data_format(data_format)
		self.activation = tf.keras.activations.get(activation)
		self.use_bias = use_bias
		self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
		self.bias_initializer = tf.keras.initializers.get(bias_initializer)
		self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
		self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
		self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
		self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
		self.bias_constraint = tf.keras.constraints.get(bias_constraint)
		self.implementation = implementation
		self.input_spec = tf.keras.layers.InputSpec(ndim=5)

	def build(self, input_shape):
		if self.data_format == 'channels_last':
			input_x, input_y, input_z = input_shape[1:-1]
			input_filter = input_shape[4]
		else:
			input_x, input_y, input_z = input_shape[2:]
			input_filter = input_shape[1]
		if input_x is None or input_y is None or input_z is None:
			raise ValueError('The spatial dimensions of the inputs to '
							' a LocallyConnected2D layer '
							'should be fully-defined, but layer received '
							'the inputs shape ' + str(input_shape))
		output_x = conv_utils.conv_output_length(input_x, self.kernel_size[0],
													self.padding, self.strides[0])
		output_y = conv_utils.conv_output_length(input_y, self.kernel_size[1],
													self.padding, self.strides[1])
		output_z = conv_utils.conv_output_length(input_z, self.kernel_size[2],
													self.padding, self.strides[2])
		self.output_x = output_x
		self.output_y = output_y
		self.output_z = output_z

		if self.implementation == 1:
			raise NotImplementedError

		elif self.implementation == 2:
			if self.data_format == 'channels_first':
				self.kernel_shape = (input_filter, input_x, input_y, input_z, self.filters,
									self.output_x, self.output_y, self.output_z)
			else:
				self.kernel_shape = (input_x, input_y, input_z, input_filter,
									self.output_x, self.output_y, self.output_z, self.filters)

			self.kernel = self.add_weight(
				shape=self.kernel_shape,
				initializer=self.kernel_initializer,
				name='kernel',
				regularizer=self.kernel_regularizer,
				constraint=self.kernel_constraint)

			self.kernel_mask = get_locallyconnected_mask(
				input_shape=(input_x, input_y, input_z),
				kernel_shape=self.kernel_size,
				strides=self.strides,
				padding=self.padding,
				data_format=self.data_format,
			)

		elif self.implementation == 3:
			self.kernel_shape = (self.output_x * self.output_y * self.output_z * self.filters,
								input_x * input_y * input_z * input_filter)

			self.kernel_idxs = sorted(
				conv_utils.conv_kernel_idxs(
					input_shape=(input_x, input_y, input_z),
					kernel_shape=self.kernel_size,
					strides=self.strides,
					padding=self.padding,
					filters_in=input_filter,
					filters_out=self.filters,
					data_format=self.data_format))

			self.kernel = self.add_weight(
				shape=(len(self.kernel_idxs),),
				initializer=self.kernel_initializer,
				name='kernel',
				regularizer=self.kernel_regularizer,
				constraint=self.kernel_constraint)

		else:
			raise ValueError('Unrecognized implementation mode: %d.' %
							self.implementation)

		if self.use_bias:
			self.bias = self.add_weight(
				shape=(output_x, output_y, output_z, self.filters),
				initializer=self.bias_initializer,
				name='bias',
				regularizer=self.bias_regularizer,
				constraint=self.bias_constraint)
		else:
			self.bias = None
		if self.data_format == 'channels_first':
			self.input_spec = tf.keras.layers.InputSpec(ndim=5, axes={1: input_filter})
		else:
			self.input_spec = tf.keras.layers.InputSpec(ndim=5, axes={-1: input_filter})
		self.built = True

	def compute_output_shape(self, input_shape):
		if self.data_format == 'channels_first':
			x = input_shape[2]
			y = input_shape[3]
			z = input_shape[4]
		elif self.data_format == 'channels_last':
			x = input_shape[1]
			y = input_shape[2]
			z = input_shape[3]

		x = conv_utils.conv_output_length(x, self.kernel_size[0],
											self.padding, self.strides[0])
		y = conv_utils.conv_output_length(y, self.kernel_size[1],
											self.padding, self.strides[1])
		z = conv_utils.conv_output_length(z, self.kernel_size[2],
											self.padding, self.strides[2])

		if self.data_format == 'channels_first':
			return (input_shape[0], self.filters, x, y, z)
		elif self.data_format == 'channels_last':
			return (input_shape[0], x, y, z, self.filters)

	def call(self, inputs):
		if self.implementation == 1:
			raise NotImplementedError

		elif self.implementation == 2:
			output = local_conv_matmul(inputs, self.kernel, self.kernel_mask,
									self.compute_output_shape(inputs.shape))
		elif self.implementation == 3:
			output = local_conv_sparse_matmul(inputs, self.kernel, self.kernel_idxs,
										self.kernel_shape,
										self.compute_output_shape(inputs.shape))
		else:
			raise ValueError('Unrecognized implementation mode: %d.' %
							self.implementation)

		if self.use_bias:
			output = tf.keras.backend.bias_add(output, self.bias, data_format=self.data_format)

		output = self.activation(output)
		return output



class _DenseVariational(tf.keras.layers.Layer):
	"""Abstract densely-connected class (private, used as implementation base).
	This layer implements the Bayesian variational inference analogue to
	a dense layer by assuming the `kernel` and/or the `bias` are drawn
	from distributions. By default, the layer implements a stochastic
	forward pass via sampling from the kernel and bias posteriors,
	```none
	kernel, bias ~ posterior
	outputs = activation(matmul(inputs, kernel) + bias)
	```
	The arguments permit separate specification of the surrogate posterior
	(`q(W|x)`), prior (`p(W)`), and divergence for both the `kernel` and `bias`
	distributions.
	"""
	def __init__(
			self,
			activation=None,
			activity_regularizer=None,
			kernel_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
			kernel_posterior_tensor_fn=lambda d: d.sample(),
			kernel_prior_fn=tfp.layers.util.default_multivariate_normal_fn,
			kernel_divergence_fn=lambda q, p, ignore: tfp.distributions.kullback_leibler.kl_divergence(q, p),
			bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(
					is_singular=True),
			bias_posterior_tensor_fn=lambda d: d.sample(),
			bias_prior_fn=None,
			bias_divergence_fn=lambda q, p, ignore: tfp.distributions.kullback_leibler.kl_divergence(q, p),
			**kwargs):
		# pylint: disable=g-doc-args
		"""Construct layer.
		Args:
			${args}
		"""
		# pylint: enable=g-doc-args
		super(_DenseVariational, self).__init__(
				activity_regularizer=activity_regularizer,
				**kwargs)
		self.activation = tf.keras.activations.get(activation)
		self.kernel_posterior_fn = kernel_posterior_fn
		self.kernel_posterior_tensor_fn = kernel_posterior_tensor_fn
		self.kernel_prior_fn = kernel_prior_fn
		self.kernel_divergence_fn = kernel_divergence_fn
		self.bias_posterior_fn = bias_posterior_fn
		self.bias_posterior_tensor_fn = bias_posterior_tensor_fn
		self.bias_prior_fn = bias_prior_fn
		self.bias_divergence_fn = bias_divergence_fn

	def call(self, inputs):
		inputs = tf.convert_to_tensor(value=inputs, dtype=self.dtype)

		outputs = self._apply_variational_kernel(inputs)
		outputs = self._apply_variational_bias(outputs)
		if self.activation is not None:
			outputs = self.activation(outputs)	# pylint: disable=not-callable
		self._apply_divergence(
				self.kernel_divergence_fn,
				self.kernel_posterior,
				self.kernel_prior,
				self.kernel_posterior_tensor,
				name='divergence_kernel')
		self._apply_divergence(
				self.bias_divergence_fn,
				self.bias_posterior,
				self.bias_prior,
				self.bias_posterior_tensor,
				name='divergence_bias')
		return outputs

	def get_config(self):
		"""Returns the config of the layer.
		A layer config is a Python dictionary (serializable) containing the
		configuration of a layer. The same layer can be reinstantiated later
		(without its trained weights) from this configuration.
		Returns:
			config: A Python dictionary of class keyword arguments and their
				serialized values.
		"""
		config = {
				'activation': (tf.keras.activations.serialize(self.activation)
											 if self.activation else None),
				'activity_regularizer':
						tf.keras.initializers.serialize(self.activity_regularizer),
		}
		function_keys = [
				'kernel_posterior_fn',
				'kernel_posterior_tensor_fn',
				'kernel_prior_fn',
				'kernel_divergence_fn',
				'bias_posterior_fn',
				'bias_posterior_tensor_fn',
				'bias_prior_fn',
				'bias_divergence_fn',
		]
		for function_key in function_keys:
			function = getattr(self, function_key)
			if function is None:
				function_name = None
				function_type = None
			else:
				function_name, function_type = tfp.layers.util.serialize_function(
						function)
			config[function_key] = function_name
			config[function_key + '_type'] = function_type
		base_config = super(_DenseVariational, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	def from_config(cls, config):
		"""Creates a layer from its config.
		This method is the reverse of `get_config`, capable of instantiating the
		same layer from the config dictionary.
		Args:
			config: A Python dictionary, typically the output of `get_config`.
		Returns:
			layer: A layer instance.
		"""
		config = config.copy()
		function_keys = [
				'kernel_posterior_fn',
				'kernel_posterior_tensor_fn',
				'kernel_prior_fn',
				'kernel_divergence_fn',
				'bias_posterior_fn',
				'bias_posterior_tensor_fn',
				'bias_prior_fn',
				'bias_divergence_fn',
		]
		for function_key in function_keys:
			serial = config[function_key]
			function_type = config.pop(function_key + '_type')
			if serial is not None:
				config[function_key] = tfp.layers.util.deserialize_function(
						serial,
						function_type=function_type)
		return cls(**config)

	def _apply_variational_bias(self, inputs):
		if self.bias_posterior is None:
			self.bias_posterior_tensor = None
			return inputs
		self.bias_posterior_tensor = self.bias_posterior_tensor_fn(
				self.bias_posterior)
		outputs = inputs
		if self.data_format == 'channels_first':
			if self.rank == 1:
				# tf.nn.bias_add does not accept a 1D input tensor.
				bias = tf.reshape(self.bias_posterior_tensor,
								[1, self.filters, 1])
				outputs += bias
			if self.rank == 2:
				outputs = tf.nn.bias_add(outputs,
										 self.bias_posterior_tensor,
										 data_format='NCHW')
			if self.rank == 3:
				# As of Mar 2017, direct addition is significantly slower than
				# bias_add when computing gradients. To use bias_add, we collapse Z
				# and Y into a single dimension to obtain a 4D input tensor.
				outputs_shape = tf.shape(outputs)
				outputs_4d = tf.reshape(outputs,
										[outputs_shape[0], outputs_shape[1],
										 outputs_shape[2] * outputs_shape[3],
										 outputs_shape[4]])
				outputs_4d = tf.nn.bias_add(outputs_4d,
											self.bias_posterior_tensor,
											data_format='NCHW')
				outputs = tf.reshape(outputs_4d, outputs_shape)
		else:

			if self.rank == 3:
				# As of Mar 2017, direct addition is significantly slower than
				# bias_add when computing gradients. To use bias_add, we collapse Z
				# and Y into a single dimension to obtain a 4D input tensor.
				outputs = tf.keras.backend.bias_add(outputs,
											self.bias_posterior_tensor,
											data_format=self.data_format)
				
		return outputs

	def _apply_divergence(self, divergence_fn, posterior, prior,
												posterior_tensor, name):
		if (divergence_fn is None or
				posterior is None or
				prior is None):
			divergence = None
			return
		divergence = tf.identity(
				divergence_fn(
						posterior, prior, posterior_tensor),
				name=name)
		self.add_loss(divergence)

class LocallyConnected3DFlipout(_DenseVariational):

	def __init__(self,
				filters,
				kernel_size,
				strides=(1, 1, 1),
				padding='valid',
				data_format=None,
				activation=None,
				trainable=True,
				kernel_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(),
				kernel_posterior_tensor_fn=lambda d: d.sample(),
				kernel_prior_fn=tfp.layers.util.default_multivariate_normal_fn,
				kernel_divergence_fn=lambda q, p, ignore: tfp.distributions.kullback_leibler.kl_divergence(q, p),
				bias_posterior_fn=tfp.layers.util.default_mean_field_normal_fn(
								is_singular=True),
				bias_posterior_tensor_fn=lambda d: d.sample(),
				bias_prior_fn=None,
				bias_divergence_fn=lambda q, p, ignore: tfp.distributions.kullback_leibler.kl_divergence(q, p),
				implementation=3,
				seed=None,
				**kwargs):
		super(LocallyConnected3DFlipout, self).__init__(**kwargs)
		self.filters = filters
		self.kernel_size = conv_utils.normalize_tuple(kernel_size, 3, 'kernel_size')
		self.strides = conv_utils.normalize_tuple(strides, 3, 'strides')
		self.padding = conv_utils.normalize_padding(padding)
		if self.padding != 'valid' and implementation == 1:
			raise ValueError('Invalid border mode for LocallyConnected3D '
							'(only "valid" is supported if implementation is 1): ' +
							padding)
		self.data_format = conv_utils.normalize_data_format(data_format)
		self.implementation = implementation
		self.input_spec = tf.keras.layers.InputSpec(ndim=5)
		self.seed=seed
		self.rank=3

	def build(self, input_shape):
		if self.data_format == 'channels_last':
			input_x, input_y, input_z = input_shape[1:-1]
			input_filter = input_shape[4]
		else:
			input_x, input_y, input_z = input_shape[2:]
			input_filter = input_shape[1]
		if input_x is None or input_y is None or input_z is None:
			raise ValueError('The spatial dimensions of the inputs to '
							' a LocallyConnected2D layer '
							'should be fully-defined, but layer received '
							'the inputs shape ' + str(input_shape))
		output_x = conv_utils.conv_output_length(input_x, self.kernel_size[0],
													self.padding, self.strides[0])
		output_y = conv_utils.conv_output_length(input_y, self.kernel_size[1],
													self.padding, self.strides[1])
		output_z = conv_utils.conv_output_length(input_z, self.kernel_size[2],
													self.padding, self.strides[2])
		self.output_x = output_x
		self.output_y = output_y
		self.output_z = output_z
	
		self.kernel_shape = (self.output_x * self.output_y * self.output_z * self.filters,
									input_x * input_y * input_z * input_filter)

		dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())

		self.kernel_idxs = sorted(
			conv_utils.conv_kernel_idxs(
				input_shape=(input_x, input_y, input_z),
				kernel_shape=self.kernel_size,
				strides=self.strides,
				padding=self.padding,
				filters_in=input_filter,
				filters_out=self.filters,
				data_format=self.data_format))

		self.kernel_posterior = self.kernel_posterior_fn(
				dtype, [len(self.kernel_idxs)], 'kernel_posterior',
				self.trainable, self.add_variable)

		if self.kernel_prior_fn is None:
			self.kernel_prior = None
		else:
			self.kernel_prior = self.kernel_prior_fn(
					dtype, [len(self.kernel_idxs)], 'kernel_prior',
					self.trainable, self.add_variable)

		if self.bias_posterior_fn is None:
			self.bias_posterior = None
		else:
			self.bias_posterior = self.bias_posterior_fn(
					dtype, [output_x, output_y, output_z, self.filters], 'bias_posterior',
					self.trainable, self.add_variable)

		if self.bias_prior_fn is None:
			self.bias_prior = None
		else:
			self.bias_prior = self.bias_prior_fn(
					dtype, [output_x, output_y, output_z, self.filters], 'bias_prior',
					self.trainable, self.add_variable)

		self.built = True


		if self.data_format == 'channels_first':
			self.input_spec = tf.keras.layers.InputSpec(ndim=5, axes={1: input_filter})
		else:
			self.input_spec = tf.keras.layers.InputSpec(ndim=5, axes={-1: input_filter})
		self.built = True

	def compute_output_shape(self, input_shape):
		if self.data_format == 'channels_first':
			x = input_shape[2]
			y = input_shape[3]
			z = input_shape[4]
		elif self.data_format == 'channels_last':
			x = input_shape[1]
			y = input_shape[2]
			z = input_shape[3]

		x = conv_utils.conv_output_length(x, self.kernel_size[0],
											self.padding, self.strides[0])
		y = conv_utils.conv_output_length(y, self.kernel_size[1],
											self.padding, self.strides[1])
		z = conv_utils.conv_output_length(z, self.kernel_size[2],
											self.padding, self.strides[2])

		if self.data_format == 'channels_first':
			return (input_shape[0], self.filters, x, y, z)
		elif self.data_format == 'channels_last':
			return (input_shape[0], x, y, z, self.filters)

	def _apply_variational_kernel(self, inputs):


		if (not isinstance(self.kernel_posterior, tfp.distributions.independent.Independent) or
				not isinstance(self.kernel_posterior.distribution, tfp.distributions.normal.Normal)):
			raise TypeError(
					'`DenseFlipout` requires '
					'`kernel_posterior_fn` produce an instance of '
					'`tfd.Independent(tfd.Normal)` '
					'(saw: \"{}\").'.format(self.kernel_posterior.name))
		self.kernel_posterior_affine = tfp.distributions.normal.Normal(
				loc=tf.zeros_like(self.kernel_posterior.distribution.loc),
				scale=self.kernel_posterior.distribution.scale)
		self.kernel_posterior_affine_tensor = (
				self.kernel_posterior_tensor_fn(self.kernel_posterior_affine))
		self.kernel_posterior_tensor = None

		input_shape = tf.shape(inputs)
		batch_shape = tf.expand_dims(input_shape[0], 0)

		if self.data_format == 'channels_first':
			channels = input_shape[1]
		else:
			channels = input_shape[-1]

		seed_stream = tfp_seed.SeedStream(self.seed, salt='LocallyConnected3DFlipout')

		sign_input = tfp.random.rademacher(
				tf.concat([batch_shape,
									 tf.expand_dims(channels, 0)], 0),
				dtype=inputs.dtype,
				seed=seed_stream())
		sign_output = tfp.random.rademacher(
				tf.concat([batch_shape,
									 tf.expand_dims(self.filters, 0)], 0),
				dtype=inputs.dtype,
				seed=seed_stream())

		if self.data_format == 'channels_first':
			for _ in range(self.rank):
				sign_input = tf.expand_dims(sign_input, -1)	# 2D ex: (B, C, 1, 1)
				sign_output = tf.expand_dims(sign_output, -1)
		else:
			for _ in range(self.rank):
				sign_input = tf.expand_dims(sign_input, 1)	# 2D ex: (B, 1, 1, C)
				sign_output = tf.expand_dims(sign_output, 1)

		outputs = local_conv_sparse_matmul(inputs, self.kernel_posterior.distribution.loc, self.kernel_idxs,
										self.kernel_shape,
										self.compute_output_shape(inputs.shape))


		perturbed_inputs = local_conv_sparse_matmul(inputs*sign_input, self.kernel_posterior_affine_tensor, self.kernel_idxs,
										self.kernel_shape,
										self.compute_output_shape(inputs.shape))*sign_output

		return outputs+perturbed_inputs


def get_locallyconnected_mask(input_shape, kernel_shape, strides, padding,
															data_format):
	mask = conv_utils.conv_kernel_mask(
			input_shape=input_shape,
			kernel_shape=kernel_shape,
			strides=strides,
			padding=padding)

	ndims = int(mask.ndim / 2)

	if data_format == 'channels_first':
		mask = np.expand_dims(mask, 0)
		mask = np.expand_dims(mask, -ndims - 1)

	elif data_format == 'channels_last':
		mask = np.expand_dims(mask, ndims)
		mask = np.expand_dims(mask, -1)

	else:
		raise ValueError('Unrecognized data_format: ' + str(data_format))

	return mask


def local_conv_matmul(inputs, kernel, kernel_mask, output_shape):
	inputs_flat = tf.reshape(inputs, (tf.shape(inputs)[0], -1))

	kernel = kernel_mask * kernel
	kernel = make_2d(kernel, split_dim=tf.keras.backend.ndim(kernel) // 2)

	output_flat = tf.linalg.matmul(inputs_flat, kernel, b_is_sparse=True)
	output = tf.reshape(output_flat, [
			tf.shape(output_flat)[0],
	] + list(output_shape)[1:])
	return output

def local_conv_sparse_matmul(inputs, kernel, kernel_idxs, kernel_shape, 
							output_shape):
	inputs_flat = tf.reshape(inputs, (tf.shape(inputs)[0], -1))
	output_flat = tf.raw_ops.SparseTensorDenseMatMul(
			a_indices=kernel_idxs,
			a_values=kernel,
			a_shape=kernel_shape,
			b=inputs_flat,
			adjoint_b=True)

	output_flat_transpose = tf.transpose(output_flat)

	output_reshaped = tf.reshape(output_flat_transpose, [
			tf.shape(output_flat_transpose)[0],
	] + list(output_shape)[1:])
	return output_reshaped

def make_2d(tensor, split_dim):
	shape = tf.shape(tensor)
	in_dims = shape[:split_dim]
	out_dims = shape[split_dim:]

	in_size = tf.reduce_prod(in_dims)
	out_size = tf.reduce_prod(out_dims)

	return tf.reshape(tensor, (in_size, out_size))