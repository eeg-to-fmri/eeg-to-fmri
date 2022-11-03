import tensorflow as tf

import numpy as np

"""

import tensorflow as tf
from layers.latent_attention import Latent_EEG_Spatial_Attention, Latent_fMRI_Spatial_Attention

layer_x=Latent_EEG_Spatial_Attention(10,)
layer_y=Latent_fMRI_Spatial_Attention(10,)
layer_x(tf.ones((2,7,7,7)))
layer_y(tf.ones((2,7,7,7,7)))

"""


class Latent_EEG_Spatial_Attention(tf.keras.layers.Layer):
	"""
	implements attention on the latent dimensions that correspond to channels and frequency original dimensions

	The latent EEG representation $\vec{z}_x \in \mathbb{R}^{L_1 \times L_2 \times L_3}$, attention is performed 
	in $\mathbb{R}^{L_1 \times L_2}$ and the hidden dimension results in $\mathbb{R}^{H \times L_3}$

	>>> import tensorflow as tf
	>>> from layers.latent_attention import Latent_EEG_Spatial_Attention
	>>> layer=Latent_EEG_Spatial_Attention(10,)
	>>> layer(tf.ones((2,7,7,7)))
	"""



	def __init__(self, H, seed=None, **kwargs):

		self.H=H
		self.seed=seed
		
		super(Latent_EEG_Spatial_Attention, self).__init__(**kwargs)


	def build(self, input_shape):
		"""
		input_shape = (None, 7, 7, 7)
		"""
		self.A = self.add_weight('A_x',
								shape=[input_shape[1]*input_shape[2],input_shape[1]*input_shape[2]*self.H],
								initializer=tf.initializers.GlorotUniform(seed=self.seed),
								dtype=tf.float32,
								trainable=True)

	def call(self, X):
		"""

		"""

		x = tf.reshape(tf.transpose(X, perm=(0,3,1,2)), shape=(tf.shape(X)[0], tf.shape(X)[3], 
																tf.shape(X)[1]*tf.shape(X)[2]))#x \in \mathbb{R}^{B, L_3, L_1*L_2}

		w = tf.matmul(x, self.A)#z \in \mathbb{R}^{B, L_1*L_2*H}
		w = tf.reshape(w, shape=(*tf.shape(w)[0:2], tf.shape(X)[1]*tf.shape(X)[2], self.H))
		a = tf.nn.softmax(w, axis=2)

		return tf.transpose(tf.squeeze(tf.reduce_sum(tf.expand_dims(x, -1)*a, axis=2)), perm=(0,2,1))


	def get_config(self):
		return {
			'H': self.H,
			'seed': self.seed,
		}

	@classmethod
	def from_config(cls, config):
		return cls(**config)



"""
import tensorflow as tf
from layers.latent_attention import Latent_fMRI_Spatial_Attention
layer=Latent_fMRI_Spatial_Attention(10,)
layer(tf.ones((2,7,7,7,20,1)))

"""
class Latent_fMRI_Spatial_Attention(tf.keras.layers.Layer):
	"""
	implements attention on the latent dimensions that correspond to the spatial original dimensions

	The latent fMRI representation $\vec{z}_y \in \mathbb{R}^{L_1 \times L_2 \times L_3 \times T}$, attention is performed 
	in $\mathbb{R}^{L_1 \times L_2 \times L_3}$ and the hidden dimension results in $\mathbb{R}^{H \times T}$

	"""



	def __init__(self, H, seed=None, **kwargs):

		self.H=H
		self.seed=seed
		
		super(Latent_fMRI_Spatial_Attention, self).__init__(**kwargs)


	def build(self, input_shape):
		"""
		input_shape = (None, 7, 7, 7, T)
		"""
		self.A = self.add_weight('A_y',
								shape=[input_shape[1]*input_shape[2]*input_shape[3],input_shape[1]*input_shape[2]*input_shape[3]*self.H],
								initializer=tf.initializers.GlorotUniform(seed=self.seed),
								dtype=tf.float32,
								trainable=True)

	def call(self, X):
		x = tf.reshape(tf.transpose(X, perm=(0,4,1,2,3)), shape=(tf.shape(X)[0], tf.shape(X)[4], 
																tf.shape(X)[1]*tf.shape(X)[2]*tf.shape(X)[3]))#x \in \mathbb{R}^{B, L_3, L_1*L_2}

		w = tf.matmul(x, self.A)#z \in \mathbb{R}^{B, L_1*L_2*H}
		w = tf.reshape(w, shape=(*tf.shape(w)[0:2], tf.shape(X)[1]*tf.shape(X)[2]*tf.shape(X)[3], self.H))
		a = tf.nn.softmax(w, axis=2)

		return tf.transpose(tf.squeeze(tf.reduce_sum(tf.expand_dims(x, -1)*a, axis=2)), perm=(0,2,1))

	def get_config(self):
		return {
			'H': self.H,
			'seed': self.seed,
		}

	@classmethod
	def from_config(cls, config):
		return cls(**config)