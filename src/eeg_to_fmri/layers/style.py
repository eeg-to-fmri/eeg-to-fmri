import tensorflow as tf
import numpy as np

class Style(tf.keras.layers.Layer):
	"""
	Style Prior layer to save model
	"""


	def __init__(self, initializer="glorot_uniform", regularizer=None, trainable=False, seed=None, name="style", **kwargs):
		"""
		initializer argument is required due to this layer having a weight
		"""

		assert initializer in ["glorot_uniform", "Constant"] or type(initializer).__name__=="Constant"

		self.initializer=initializer
		self.regularizer=regularizer
		self.seed=seed

		super(Style, self).__init__(trainable=trainable, name=name)

	def build(self, input_shape):
		"""

		"""
		if(self.initializer=="glorot_uniform"):
			initializer=tf.keras.initializers.GlorotUniform(seed=self.seed)
		else:
			#case if constant initializer
			initializer=self.initializer

		self.latent_style = self.add_weight(name='style_prior',
											shape=(input_shape[1],),
											regularizer=self.regularizer,
											initializer=initializer,
											trainable=self.trainable)

	def call(self, x,):
		"""
		simple multiplication		
		"""
		return self.latent_style*x

	def get_config(self,):
		if(type(self.initializer).__name__=="Constant"):
			raise NotImplementedError

		return {
			"initializer": self.initializer,
			"trainable": self.trainable,
			"seed": self.seed,
		}

	@classmethod
	def from_config(cls, config):
		return cls(**config)