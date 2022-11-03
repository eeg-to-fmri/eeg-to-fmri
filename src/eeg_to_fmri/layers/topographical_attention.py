import tensorflow as tf

from eeg_to_fmri.regularizers.activity_regularizers import OrganizeChannels

class Topographical_Attention_Scores_Regularization(tf.keras.layers.Layer):

	def __init__(self, **kwargs):
		
		super(Topographical_Attention_Scores_Regularization, self).__init__(activity_regularizer=OrganizeChannels(), **kwargs)
		
	def get_config(self):
		return {}

	@classmethod
	def from_config(cls, config):
		return cls(**config)


class Topographical_Attention_Reduction(tf.keras.layers.Layer):
	def __init__(self, **kwargs):
		
		super(Topographical_Attention_Reduction, self).__init__(**kwargs)

		
	def call(self, X, W):
		"""
		Reduction by multiplication, layers should be simple, it is good practice
		""" 

		return tf.linalg.matmul(W, X)

	def get_config(self):
		return {}

	@classmethod
	def from_config(cls, config):
		return cls(**config)



class Topographical_Attention(tf.keras.layers.Layer):
	"""
	Topographical_Attention:

		Int channels
		Int features reduce over number of features

	"""

	def __init__(self, channels, features, regularizer=None, seed=None, **kwargs):

		self.channels=channels
		self.features=features
		self.regularizer=regularizer
		self.seed=seed

		super(Topographical_Attention, self).__init__(**kwargs)

	def build(self, input_shape):
		self.A = self.add_weight('A',
								#shape=[self.channels,self.channels,self.features],
								shape=[self.channels,self.features],
								regularizer=self.regularizer,
								initializer=tf.initializers.GlorotUniform(seed=self.seed),
								dtype=tf.float32,
								trainable=True)
		
	def call(self, X):
		"""
		The defined topographical attention mechanism has an extra step:

			instead of performing the element wise product,
			one reduces the feature dimension,
			so instead of the attention weight matrix being of shape NxN
			it has shape NxNxF
			the F refers to the feature dimension that is reduced
		"""

		c = tf.tensordot(X, self.A, axes=[[2], [1]])
		#c = tf.einsum('NCF,CMF->NCM', X, self.A)
		W = tf.nn.softmax(c, axis=-1)#dimension that is reduced in the next einsum, is the one that sums to one
		self.attention_scores=W

		#if(self.organize_channels):
			#self.losses=[self.organize_regularization(W)]#### COMPUTE GRADIENTS W.R.T. self.A, acitivity regularizer is not possible
		#tf.print(tf.reduce_sum(self.organize_regularization(W)))

		return tf.linalg.matmul(W, X), self.attention_scores
		#return tf.einsum('NMF,NCM->NCF', X, W), self.attention_scores

	def lrp(self, x, y):
		#store attention scores
		self.call(x)

		with tf.GradientTape(watch_accessed_variables=False) as tape:
			tape.watch(x)
			
			z = tf.tensordot(x, self.attention_scores, axes=[[1], [2]])+1e-9
			#z = tf.einsum('NMF,NCM->NCF', x, self.attention_scores)+1e-9

			s = y/tf.reshape(z, y.shape)
			s = tf.reshape(s, z.shape)
			
			c = tape.gradient(tf.reduce_sum(z*s.numpy()), x)
			R = x*c

		return R

	def lrp_attention(self, x, y):
		#store attention scores
		self.call(x)

		with tf.GradientTape(watch_accessed_variables=False) as tape:
			tape.watch(self.attention_scores)

			z = tf.tensordot(x, self.attention_scores, axes=[[1], [2]])+1e-9
			#z = tf.einsum('NMF,NCM->NCF', x, self.attention_scores)+1e-9

			s = y/tf.reshape(z, y.shape)
			s = tf.reshape(s, z.shape)
			
			c = tape.gradient(tf.reduce_sum(z*s.numpy()), self.attention_scores)
			R = self.attention_scores*c

		return R

	def get_config(self):
		return {
			'channels': self.channels,
			'features': self.features,
			'seed': self.seed,
		}

	@classmethod
	def from_config(cls, config):
		return cls(**config)
