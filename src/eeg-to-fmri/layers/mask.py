import tensorflow as tf

import numpy as np


"""
A simple mask that formes a 3D circle with an elipse in the z-axis


Example:

	>>> layer=CircleMask(y.shape)
"""
class MRICircleMask(tf.keras.layers.Layer):
	
	def __init__(self, input_shape, radius=30.):
		super(MRICircleMask, self).__init__()
		
		if(len(input_shape)==5):
			input_shape = input_shape[1:-1]
		elif(len(input_shape)==4 and input_shape[0]==1):
			input_shape = input_shape[:-1]
		elif(len(input_shape)==4 and input_shape[0]!=1):
			input_shape = input_shape[1:]
		
		h,w,d = input_shape
		center=[h//2, w//2, d//2]

		Y, X, Z = np.ogrid[:h, :w, :d]
		dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2 + (h/d)*(Z-center[2])**2)
		
		self.mask = self.add_weight('mask',
								shape=list(input_shape+(1,)),
								initializer=tf.constant_initializer((dist_from_center <= radius).astype(np.float32)),
								dtype=tf.float32,
								trainable=False)
		
	def call(self, X):
		return X*self.mask


class LearnedMask(tf.keras.layers.Layer):



	def __init__(self, fmri_shape, seed=None, **kwargs):

		self.fmri_shape=fmri_shape
		self.seed=seed

		super(LearnedMask, self).__init__(**kwargs)

	def build(self, input_shape):

		self.L = self.add_weight('L',
								shape=list(*self.fmri_shape)+[2],
								initializer=tf.initializers.GlorotUniform(seed=self.seed),
								dtype=tf.float32,
								trainable=True)

	def call(self, X):

		tf.nn.softmax(self.L, axis=-1)


		raise NotImplementedError

	def get_config():
		return {
			'fmri_shape': self.fmri_shape,
			'seed': self.seed,
		}

	@classmethod
	def from_config(cls, config):
		return cls(**config)
