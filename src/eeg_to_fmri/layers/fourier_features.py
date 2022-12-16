import tensorflow as tf

import numpy as np

from eeg_to_fmri.regularizers.activity_regularizers import InOfDistribution, MaxBatchNorm

_SUPPORTED_RBF_KERNEL_TYPES = ['gaussian']

def _get_default_scale(initializer, input_dim):
	if (isinstance(initializer, str) and
			initializer.lower() == 'gaussian'):
		return np.sqrt(input_dim / 2.0)
	return 1.0

def _get_random_features_initializer(initializer, shape, seed=None):
	"""Returns Initializer object for random features."""

	def _get_cauchy_samples(loc, scale, shape):
		probs = np.random.uniform(low=0., high=1., size=shape)
		return loc + scale * np.tan(np.pi * (probs - 0.5))

	random_features_initializer = initializer
	if isinstance(initializer, str):
		if initializer.lower() == 'gaussian':
			random_features_initializer = tf.random_normal_initializer(
					stddev=1.0, seed=seed)
		elif initializer.lower() == 'laplacian':
			random_features_initializer =  tf.constant_initializer(
					_get_cauchy_samples(loc=0.0, scale=1.0, shape=shape))

		else:
			raise ValueError(
					'Unsupported kernel type: \'{}\'. Supported kernel types: {}.'.format(
							random_features_initializer, _SUPPORTED_RBF_KERNEL_TYPES))
	return random_features_initializer


class MaxNormalization(tf.keras.layers.Layer):

	def __init__(self, mu=0, l=0.01, p=1., **kwargs):
		
		super(MaxNormalization, self).__init__(activity_regularizer=MaxBatchNorm(mu=mu, l=l, p=p), **kwargs)
		
	def get_config(self):
		return {}

	@classmethod
	def from_config(cls, config):
		return cls(**config)


class Sinusoids(tf.keras.layers.Layer):


	def __init__(self, **kwargs):

		super(Sinusoids, self).__init__(**kwargs)

	def call(self, X):
		return tf.cos(X)

	def get_config(self):
		return {}

	@classmethod
	def from_config(cls, config):
		return cls(**config)

class TanhNormalization(tf.keras.layers.Layer):

	def __init__(self, **kwargs):

		super(TanhNormalization, self).__init__(**kwargs)

	def call(self, X):
		return tf.keras.activations.tanh(X)*(np.pi/2)+(np.pi/2)

	def get_config(self):
		return {}

	@classmethod
	def from_config(cls, config):
		return cls(**config)

class RandomFourierFeatures(tf.keras.layers.Layer):

	def __init__(self, output_dim, kernel_initializer='gaussian', scale=None, normalization="layer", trainable=False, units=None, seed=None, name=None, **kwargs):
		if output_dim <= 0:
			raise ValueError(
			'`output_dim` should be a positive integer. Given: {}.'.format(
			output_dim))
		if isinstance(kernel_initializer, str):
			if kernel_initializer.lower() not in _SUPPORTED_RBF_KERNEL_TYPES:
				raise ValueError(
				'Unsupported kernel type: \'{}\'. Supported kernel types: {}.'
				.format(kernel_initializer, _SUPPORTED_RBF_KERNEL_TYPES))
		if scale is not None and scale <= 0.0:
			raise ValueError('When provided, `scale` should be a positive float. '
			'Given: {}.'.format(scale))
		super(RandomFourierFeatures, self).__init__(name=name)
		self.output_dim = output_dim
		self.units=output_dim
		self.normalization=normalization
		self.kernel_initializer = kernel_initializer
		if(normalization=="layer"):
			self.layer_normalization=tf.keras.layers.LayerNormalization(beta_initializer=tf.constant_initializer(np.pi/2), gamma_initializer=tf.constant_initializer(np.pi/2), trainable=False)
			self.reg_normalization=MaxNormalization(mu=np.pi/2, l=0.5*(2/np.pi)**0.5, p=2)
		elif(normalization=="tanh"):
			self.layer_normalization=TanhNormalization()
			self.reg_normalization=MaxNormalization(mu=np.pi/2, l=0.5*(2/np.pi)**0.5, p=2)
		self.scale = scale
		self.seed=seed
		self.trainable=trainable
		super(RandomFourierFeatures, self).__init__(trainable=trainable, **kwargs)

	def build(self, input_shape):
		input_shape = tf.TensorShape(input_shape)
		# TODO(pmol): Allow higher dimension inputs. Currently the input is expected
		# to have shape [batch_size, dimension].
		if input_shape.rank != 2:
			raise ValueError(
			'The rank of the input tensor should be 2. Got {} instead.'.format(
			input_shape.ndims))
		if input_shape.dims[1].value is None:
			raise ValueError(
			'The last dimension of the inputs to `RandomFourierFeatures` '
			'should be defined. Found `None`.')
		input_dim = input_shape.dims[1].value

		kernel_initializer = _get_random_features_initializer(self.kernel_initializer, shape=(input_dim, self.output_dim), seed=self.seed)

		self.unscaled_kernel = self.add_weight(name='unscaled_kernel',
											shape=(input_dim, self.output_dim),dtype=tf.float32,
											initializer=kernel_initializer,trainable=False)

		self.bias = self.add_weight(name='bias',shape=(self.output_dim,),
									dtype=tf.float32, 
									initializer=tf.random_uniform_initializer( 
									minval=0.0, maxval=2 * np.pi, seed=self.seed),
									trainable=False)

		if self.scale is None:
			self.scale = _get_default_scale(self.kernel_initializer, input_dim)
		self.kernel_scale = self.add_weight(name='kernel_scale',shape=(1,),
											dtype=tf.float32, initializer= tf.constant_initializer(self.scale),
											trainable=self.trainable, constraint='NonNeg')
		super(RandomFourierFeatures, self).build(input_shape)

	def call(self, inputs):
		inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
		inputs = tf.cast(inputs, tf.float32)
		kernel = (1.0 / self.kernel_scale) * self.unscaled_kernel
		outputs = tf.raw_ops.MatMul(a=inputs, b=kernel)
		outputs = tf.nn.bias_add(outputs, self.bias)
		
		outputs=self.layer_normalization(outputs)

		outputs=self.reg_normalization(outputs)
		
		return outputs

	def compute_output_shape(self, input_shape):
		input_shape = tf.TensorShape(input_shape)
		input_shape = input_shape.with_rank(2)
		if input_shape.dims[-1].value is None:
			raise ValueError(
				'The innermost dimension of input shape must be defined. Given: %s' %
				input_shape)
		return input_shape[:-1].concatenate(self.output_dim)

	def get_config(self):
		kernel_initializer = self.kernel_initializer
		if not isinstance(kernel_initializer, str):
			kernel_initializer = initializers.serialize(kernel_initializer)
		config = {
			'output_dim': self.output_dim,
			'kernel_initializer': kernel_initializer,
			'scale': self.scale,
			'units': self.units,
			'normalization': self.normalization,
		}
		base_config = super(RandomFourierFeatures, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


	def lrp_call(self, inputs):
		inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
		inputs = tf.cast(inputs, tf.float32)
		kernel = (1.0 / self.kernel_scale) * self.unscaled_kernel
		outputs = tf.raw_ops.MatMul(a=inputs, b=kernel)
		outputs = tf.nn.bias_add(outputs, self.bias)/np.pi
		return tf.math.minimum(1-outputs, 3-outputs)

	"""
	x - is the input of the layer
	y - contains the so far computed relevances
	"""
	def lrp(self, x, y):

		with tf.GradientTape(watch_accessed_variables=False) as tape:
			tape.watch(x)
			
			z = self.lrp_call(x)+1e-9
			s = y/tf.reshape(z, y.shape)
			s = tf.reshape(s, z.shape)

			c = tape.gradient(tf.reduce_sum(z*s.numpy()), x)
			R = x*c

		return R


class FourierFeatures(tf.keras.layers.Layer):

	def __init__(self, output_dim, trainable=False, name=None, seed=None, **kwargs):
		if output_dim <= 0:
			raise ValueError(
			'`output_dim` should be a positive integer. Given: {}.'.format(
			output_dim))
		super(FourierFeatures, self).__init__(name=name)
		self.seed=seed
		self.output_dim = output_dim
		super(FourierFeatures, self).__init__(trainable=trainable, **kwargs)

	def build(self, input_shape):
		input_shape = tf.TensorShape(input_shape)
		# TODO(pmol): Allow higher dimension inputs. Currently the input is expected
		# to have shape [batch_size, dimension].
		if input_shape.rank != 2:
			raise ValueError(
			'The rank of the input tensor should be 2. Got {} instead.'.format(
			input_shape.ndims))
		if input_shape.dims[1].value is None:
			raise ValueError(
			'The last dimension of the inputs to `FourierFeatures` '
			'should be defined. Found `None`.')
		input_dim = input_shape.dims[1].value

		kernel_initializer = []
		for j in range(input_dim):
			proj = []
			for i in range(self.output_dim):
				proj.append(np.pi*2**(i/self.output_dim))
			kernel_initializer.append(proj)
		kernel_initializer = np.array(kernel_initializer)

		self.kernel = self.add_weight(name='kernel',
											shape=(input_dim, self.output_dim),dtype=tf.float32,
											initializer= tf.constant_initializer(kernel_initializer),trainable=False)

		super(FourierFeatures, self).build(input_shape)

	def call(self, inputs):
		inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
		inputs = tf.cast(inputs, tf.float32)
		outputs = tf.raw_ops.MatMul(a=inputs, b=self.kernel)
		return tf.cos(outputs)
		#return tf.concat([tf.cos(outputs),tf.sin(outputs)], axis=-1)

	def compute_output_shape(self, input_shape):
		input_shape = tf.TensorShape(input_shape)
		input_shape = input_shape.with_rank(2)
		if input_shape.dims[-1].value is None:
			raise ValueError(
				'The innermost dimension of input shape must be defined. Given: %s' %
				input_shape)
		return input_shape[:-1].concatenate(self.output_dim)

	def get_config(self):
		config = {
			'output_dim': self.output_dim
		}
		base_config = super(FourierFeatures, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	def lrp_call(self, inputs):
		inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
		inputs = tf.cast(inputs, tf.float32)
		outputs = tf.raw_ops.MatMul(a=inputs, b=self.kernel)/np.pi
		return tf.math.minimum(1-outputs, 3-outputs)

	"""
	x - is the input of the layer
	y - contains the so far computed relevances
	"""
	def lrp(self, x, y):

		with tf.GradientTape(watch_accessed_variables=False) as tape:
			tape.watch(x)
			
			z = self.call(x)+1e-9
			s = y/tf.reshape(z, y.shape)
			s = tf.reshape(s, z.shape)
			c = tape.gradient(tf.reduce_sum(z*s.numpy()), x)
			R = x*c

		return R




_get_default_scale

_get_random_features_initializer