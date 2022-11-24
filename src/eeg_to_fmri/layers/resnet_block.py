import tensorflow as tf

from eeg_to_fmri.explainability import lrp

"""
Resnet-18 block that has implemented 
the backward step for LRP - Layer-Wise
Relevance Backpropagation 

Example usage:
	>>> import tensorflow as tf
	>>> import resnet_block 
	>>> layer = resnet_block.ResBlock(tf.keras.layers.Conv3D, (5,5,5), (1,1,1), 1, maxpool=False, seed=42)
	>>> x = tf.ones((1,10,10,10,1))
	>>> layer.lrp(x, layer(x))
"""
class ResBlock(tf.keras.layers.Layer):
	"""
		inputs:
			* x - Tensor
			* kernel_size - tuple
			* stride_size - tuple
			* n_channels - int
			* maxpool - bool
			* batch_norm - bool
			* weight_decay - float
			* skip_connections - bool
			* maxpool_k - tuple
			* maxpool_s - tuple
			* seed - int
	"""
	def __init__(self, operation, kernel_size, stride_size, n_channels,
						maxpool=True, batch_norm=True, 
						weight_decay=0.000, skip_connections=True,
						maxpool_k=None, maxpool_s=None,
						seed=None, **kwargs):
		super(ResBlock, self).__init__(**kwargs)

		self.operation=operation
		self.kernel_size=kernel_size
		self.stride_size=stride_size
		self.n_channels=n_channels
		self.maxpool=maxpool
		self.batch_norm=batch_norm
		self.weight_decay=weight_decay
		self.skip_connections=skip_connections
		self.maxpool_k=maxpool_k
		self.maxpool_s=maxpool_s
		self.seed=seed

		self.set_layers(getattr(tf.keras.layers, operation), kernel_size, stride_size, n_channels,
						maxpool=maxpool, batch_norm=batch_norm, 
						weight_decay=weight_decay, skip_connections=skip_connections,
						maxpool_k=maxpool_k, maxpool_s=maxpool_s, seed=seed)

	def set_layers(self, operation, kernel_size, stride_size, n_channels,
						maxpool=True, batch_norm=True, 
						weight_decay=0.000, skip_connections=True,
						maxpool_k=None, maxpool_s=None, seed=None):

		self.left_layers = []
		self.right_layers = []
		self.join_layers = []

		self.left_layers += [operation(filters=n_channels, kernel_size=kernel_size, strides=stride_size,
										kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
										bias_regularizer=tf.keras.regularizers.L2(weight_decay),
										kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
										padding="valid")]
		if(maxpool):
			self.left_layers += [tf.keras.layers.MaxPool3D(pool_size=maxpool_k, strides=maxpool_s)]
		if(batch_norm):
			self.left_layers += [tf.keras.layers.BatchNormalization()]
		self.left_layers += [tf.keras.layers.ReLU()]

		self.left_layers += [operation(filters=n_channels, kernel_size=3, strides=1,
										kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
										bias_regularizer=tf.keras.regularizers.L2(weight_decay),
										kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
										padding="same")]
		if(batch_norm):
			self.left_layers += [tf.keras.layers.BatchNormalization()]
		self.left_layers += [tf.keras.layers.ReLU()]


		self.right_layers += [operation(filters=n_channels, kernel_size=kernel_size, strides=stride_size,
											kernel_regularizer=tf.keras.regularizers.L2(weight_decay),
											bias_regularizer=tf.keras.regularizers.L2(weight_decay),
											kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
											padding="valid")]
		if(maxpool):
			self.right_layers += [tf.keras.layers.MaxPool3D(pool_size=maxpool_k, strides=maxpool_s)]
		if(batch_norm):
			self.right_layers += [tf.keras.layers.BatchNormalization()]
		
		self.join_layers += [tf.keras.layers.Add()]
		self.join_layers += [tf.keras.layers.ReLU()]


	def call(self, x):

		self.left_activations = []
		self.right_activations = []
		self.join_activations = []

		#left pass
		z_left = self.left_layers[0](x)
		self.left_activations += [z_left]
		for layer in range(1, len(self.left_layers)):
			z_left = self.left_layers[layer](z_left)
			self.left_activations += [z_left]

		#right pass
		z_right = self.right_layers[0](x)
		self.right_activations += [z_right]
		for layer in range(1, len(self.right_layers)):
			z_right = self.right_layers[layer](z_right)
			self.right_activations += [z_right]

		#join pass
		z = self.join_layers[0]([z_left, z_right])
		self.join_activations += [z]
		z = self.join_layers[1](z)
		self.join_activations += [z]
		
		return z

	def lrp(self, x, y):
		R_join = [None]*(len(self.join_layers)-1) + \
						[y]
		R_left = [None]*(len(self.left_layers))
		R_right = [None]*(len(self.right_layers))

		#begin with join block
		for layer in range(len(self.join_layers))[::-1]:
			if("batch" in self.join_layers[layer].name):
				R_join[layer] = R_join[layer+1]
				continue
			elif("add" in self.join_layers[layer].name):
				R = lrp.lrp([self.left_activations[-1], self.right_activations[-1]], R_join[layer], self.join_layers[layer])
				R_left[-1] = R
				R_right[-1] = R
				continue
			if(layer-1 >= 0):
				R_join[layer-1] = lrp.lrp(self.join_activations[layer-1], R_join[layer], self.join_layers[layer])
			else:
				raise NotImplementedError
				
		
		#left block
		for layer in range(len(self.left_layers))[::-1]:
			if("batch" in self.left_layers[layer].name):
				R_left[layer-1] = R_left[layer]
				continue
			if(layer-1 >= 0):
				R_left[layer-1] = lrp.lrp(self.left_activations[layer-1], R_left[layer], self.left_layers[layer])
			else:
				R_left_ = lrp.lrp(x, R_left[layer], self.left_layers[layer])
			
		#right block
		for layer in range(len(self.right_layers))[::-1]:
			if("batch" in self.right_layers[layer].name):
				R_right[layer-1] = R_right[layer]
				continue
			if(layer-1 >= 0):
				R_right[layer-1] = lrp.lrp(self.right_activations[layer-1], R_right[layer], self.right_layers[layer])
			else:
				R_right_ = lrp.lrp(x, R_right[layer], self.right_layers[layer])
				
		#sum of the modulos, this breaks negative feature importance
		return R_left_+R_right_

	def get_config(self):
		config = {
			"operation": self.operation,
			"kernel_size": self.kernel_size,
			"stride_size": self.stride_size,
			"n_channels": self.n_channels,
			"maxpool": self.maxpool,
			"batch_norm": self.batch_norm,
			"weight_decay": self.weight_decay,
			"skip_connections": self.skip_connections,
			"maxpool_k": self.maxpool_k,
			"maxpool_s": self.maxpool_s,
			"seed": self.seed
		}
		base_config = super(ResBlock, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

	@classmethod
	def from_config(cls, config):
		return cls(**config)


"""
Resnet-18 block that has implemented 
the backward step for LRP - Layer-Wise
Relevance Backpropagation 

Example usage:
	>>> import tensorflow as tf
	>>> import resnet_block 
	>>> layer = resnet_block.ResBlock(tf.keras.layers.Conv3D, (5,5,5), (1,1,1), 1, maxpool=False, seed=42)
	>>> x = tf.ones((1,10,10,10,1))
	>>> layer.lrp(x, layer(x))
"""
class pretrained_ResBlock(tf.keras.layers.Layer):
	"""
		inputs:
			* x - Tensor
			* kernel_size - tuple
			* stride_size - tuple
			* n_channels - int
			* maxpool - bool
			* batch_norm - bool
			* weight_decay - float
			* skip_connections - bool
			* maxpool_k - tuple
			* maxpool_s - tuple
			* seed - int
	"""
	def __init__(self, resblock, activation=None, trainable=False, regularizer=None, seed=None):
		super(pretrained_ResBlock, self).__init__()
		
		self._trainable=trainable
		self.set_layers(resblock, activation=activation, regularizer=regularizer, seed=seed)

	def set_layers(self, resblock, activation=None, regularizer=None, seed=None):

		self.left_layers = []
		self.right_layers = []
		self.join_layers = []
		
		if(self._trainable):
			kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
			bias_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
		else:
			kernel_initializer=tf.constant_initializer(resblock.left_layers[0].kernel.numpy())
			bias_initializer=tf.constant_initializer(resblock.left_layers[0].bias.numpy())
		if(regularizer is None):
			kernel_regularizer=tf.keras.regularizers.L2(float(resblock.left_layers[0].kernel_regularizer.l2))
			bias_regularizer=tf.keras.regularizers.L2(float(resblock.left_layers[0].bias_regularizer.l2))
		else:
			kernel_regularizer=regularizer
			bias_regularizer=regularizer
		self.left_layers += [getattr(tf.keras.layers, type(resblock.left_layers[0]).__name__)(
										filters=resblock.left_layers[0].filters, 
										kernel_size=resblock.left_layers[0].kernel_size, 
										strides=resblock.left_layers[0].strides,
										activation=activation,
										kernel_regularizer=kernel_regularizer,
										bias_regularizer=bias_regularizer,
										kernel_initializer=kernel_initializer,
										bias_initializer=bias_initializer,
										padding=resblock.left_layers[0].padding,
										trainable=self._trainable)]
		self.left_layers += [getattr(tf.keras.layers, type(resblock.left_layers[1]).__name__)(
										pool_size=resblock.left_layers[1].pool_size, 
										strides=resblock.left_layers[1].strides)]
		if(self._trainable):
			kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
			bias_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
		else:
			kernel_initializer=tf.constant_initializer(resblock.left_layers[4].kernel.numpy())
			bias_initializer=tf.constant_initializer(resblock.left_layers[4].bias.numpy())
		if(regularizer is None):
			kernel_regularizer=tf.keras.regularizers.L2(float(resblock.left_layers[4].bias_regularizer.l2))
			bias_regularizer=tf.keras.regularizers.L2(float(resblock.left_layers[4].bias_regularizer.l2))
		else:
			kernel_regularizer=regularizer
			bias_regularizer=regularizer
		self.left_layers += [getattr(tf.keras.layers, type(resblock.left_layers[2]).__name__)(trainable=self._trainable)]
		self.left_layers += [getattr(tf.keras.layers, type(resblock.left_layers[3]).__name__)(trainable=self._trainable)]
		self.left_layers += [getattr(tf.keras.layers, type(resblock.left_layers[4]).__name__)(
										filters=resblock.left_layers[4].filters, 
										kernel_size=resblock.left_layers[4].kernel_size, 
										strides=resblock.left_layers[4].strides,
										activation=activation,
										kernel_regularizer=kernel_regularizer,
										bias_regularizer=bias_regularizer,
										kernel_initializer=kernel_initializer,
										bias_initializer=bias_initializer,
										padding=resblock.left_layers[4].padding,
										trainable=self._trainable)]
		self.left_layers += [getattr(tf.keras.layers, type(resblock.left_layers[5]).__name__)(trainable=self._trainable)]
		self.left_layers += [getattr(tf.keras.layers, type(resblock.left_layers[6]).__name__)(trainable=self._trainable)]

		if(self._trainable):
			kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
			bias_initializer=tf.keras.initializers.GlorotUniform(seed=seed)
		else:
			kernel_initializer=tf.constant_initializer(resblock.right_layers[0].kernel.numpy())
			bias_initializer=tf.constant_initializer(resblock.right_layers[0].bias.numpy())
		if(regularizer is None):
			kernel_regularizer=tf.keras.regularizers.L2(float(resblock.right_layers[0].bias_regularizer.l2))
			bias_regularizer=tf.keras.regularizers.L2(float(resblock.right_layers[0].bias_regularizer.l2))
		else:
			kernel_regularizer=regularizer
			bias_regularizer=regularizer
		self.right_layers += [getattr(tf.keras.layers, type(resblock.right_layers[0]).__name__)(
										filters=resblock.right_layers[0].filters, 
										kernel_size=resblock.right_layers[0].kernel_size, 
										strides=resblock.right_layers[0].strides,
										activation=activation,
										kernel_regularizer=kernel_regularizer,
										bias_regularizer=bias_regularizer,
										kernel_initializer=kernel_initializer,
										bias_initializer=bias_initializer,
										padding=resblock.right_layers[0].padding,
										trainable=self._trainable)]
		self.right_layers += [getattr(tf.keras.layers, type(resblock.right_layers[1]).__name__)(
										pool_size=resblock.right_layers[1].pool_size, 
										strides=resblock.right_layers[1].strides)]
		self.right_layers += [getattr(tf.keras.layers, type(resblock.right_layers[2]).__name__)(trainable=self._trainable)]

		self.join_layers += [getattr(tf.keras.layers, type(resblock.join_layers[0]).__name__)(trainable=self._trainable)]
		self.join_layers += [getattr(tf.keras.layers, type(resblock.join_layers[1]).__name__)(trainable=self._trainable)]


	def call(self, x):

		self.left_activations = []
		self.right_activations = []
		self.join_activations = []

		#left pass
		z_left = self.left_layers[0](x)
		self.left_activations += [z_left]
		for layer in range(1, len(self.left_layers)):
			z_left = self.left_layers[layer](z_left)
			self.left_activations += [z_left]

		#right pass
		z_right = self.right_layers[0](x)
		self.right_activations += [z_right]
		for layer in range(1, len(self.right_layers)):
			z_right = self.right_layers[layer](z_right)
			self.right_activations += [z_right]

		#join pass
		z = self.join_layers[0]([z_left, z_right])
		self.join_activations += [z]
		z = self.join_layers[1](z)
		self.join_activations += [z]

		return z

	def lrp(self, x, y):
		R_join = [None]*(len(self.join_layers)-1) + \
						[y]
		R_left = [None]*(len(self.left_layers))
		R_right = [None]*(len(self.right_layers))

		#begin with join block
		for layer in range(len(self.join_layers))[::-1]:
			if("batch" in self.join_layers[layer].name):
				R_join[layer] = R_join[layer+1]
				continue
			elif("add" in self.join_layers[layer].name):
				R = lrp.lrp([self.left_activations[-1], self.right_activations[-1]], R_join[layer], self.join_layers[layer])
				R_left[-1] = R
				R_right[-1] = R
				continue
			if(layer-1 >= 0):
				R_join[layer-1] = lrp.lrp(self.join_activations[layer-1], R_join[layer], self.join_layers[layer])
			else:
				raise NotImplementedError


		#left block
		for layer in range(len(self.left_layers))[::-1]:
			if("batch" in self.left_layers[layer].name):
				R_left[layer-1] = R_left[layer]
				continue
			if(layer-1 >= 0):
				R_left[layer-1] = lrp.lrp(self.left_activations[layer-1], R_left[layer], self.left_layers[layer])
			else:
				R_left_ = lrp.lrp(x, R_left[layer], self.left_layers[layer])

		#right block
		for layer in range(len(self.right_layers))[::-1]:
			if("batch" in self.right_layers[layer].name):
				R_right[layer-1] = R_right[layer]
				continue
			if(layer-1 >= 0):
				R_right[layer-1] = lrp.lrp(self.right_activations[layer-1], R_right[layer], self.right_layers[layer])
			else:
				R_right_ = lrp.lrp(x, R_right[layer], self.right_layers[layer])

		#sum of the modulos, this breaks negative feature importance
		return R_left_+R_right_