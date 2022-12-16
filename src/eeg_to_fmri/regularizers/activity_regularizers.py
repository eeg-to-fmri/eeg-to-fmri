import tensorflow as tf

import numpy as np


class InOfDistribution(tf.keras.regularizers.Regularizer):

	def __init__(self, l=1.0, m=np.pi, **kwargs):
		self.l=tf.keras.backend.cast_to_floatx(l)
		self.m=tf.keras.backend.cast_to_floatx(m)

		super(InOfDistribution, self).__init__(**kwargs)

	@tf.function(autograph=True,input_signature=[tf.TensorSpec([None], tf.float32)])
	def __call__(self, x):

		return self.l*tf.reduce_sum(tf.abs((x-tf.math.reduce_min(x, axis=0))/(tf.math.reduce_max(x,axis=0)-tf.math.reduce_min(x, axis=0))-(x)/(self.m)))

	def get_config(self):
		return {"l": self.l}

	@classmethod
	def from_config(cls, config):
		return cls(**config)


class MaxBatchNorm(tf.keras.regularizers.Regularizer):

	def __init__(self, mu=0.1, l=1.0, p=1., **kwargs):
		self.l=tf.keras.backend.cast_to_floatx(l)
		self.p=tf.keras.backend.cast_to_floatx(p)
		self.mu=tf.keras.backend.cast_to_floatx(mu)

		super(MaxBatchNorm, self).__init__(**kwargs)

	@tf.function(autograph=True)
	def __call__(self, x):
		
		return self.l*tf.reduce_sum(tf.norm(x-self.mu, ord=self.p, axis=1))#over the batch

	def get_config(self):
		return {"l": self.l,
				"p": self.p,
				"mu": self.mu}

	@classmethod
	def from_config(cls, config):
		return cls(**config)



class OrganizeChannels(tf.keras.regularizers.Regularizer):
	"""
	Activity regularizer for the Topographical_Attention layer that forces the attention scores to be heterogeneous for each channel chosen.
	
	That is the feature selection property of the attention mechanism is surpressed and instead it is more a reordering of channels.
	"""

	def __init__(self, l=1.0, **kwargs):

		self.l=tf.keras.backend.cast_to_floatx(l)

		super(OrganizeChannels, self).__init__(**kwargs)

	@tf.function(autograph=True,input_signature=[tf.TensorSpec([None], tf.float32)])
	def __call__(self, x):
		"""
		minimization of this term allows the matrix to be heterogeneous row and column wise, that is every channel is selected and it reorders
		Example in numpy:

		>>> a=np.array([[0.99,0.005,0.005],[0.005,0.99,0.005],[0.005,0.005,0.99]])
		>>> b=np.array([[0.005,0.99,0.005],[0.005,0.99,0.005],[0.005,0.005,0.99]])
		>>> c=np.array([[0.7,0.1,0.2],[0.005,0.99,0.005],[0.005,0.005,0.99]])
		>>> 
		>>> -np.log(np.sum(a, axis=0))
		<<< array([-0., -0., -0.])
		>>> -np.log(np.sum(b, axis=0))
		<<< array([ 4.19970508, -0.68561891, -0.		])
		>>> -np.log(np.sum(c, axis=0))
		<<< array([ 0.34249031, -0.09075436, -0.17814619])

		a: is the optimal objective that minimizes this term
		b: is the wrong and has the highest penalty for selecting a channel more than once
		c: is the suboptimal objective that still has room to improve in terms of 
		"""
		return -self.l*tf.reduce_sum(tf.math.log(tf.reduce_sum(x, axis=-2)+1e-9))#it is the -2 axis because we need to know if a channel is being selected more than once

	def get_config(self):
		return {"l": self.l}

	@classmethod
	def from_config(cls, config):
		return cls(**config)