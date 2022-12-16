import tensorflow as tf

import gc

OPTIMIZER=tf.keras.optimizers.Adam


def optimizer(name, input_shape, model, lr):
	if(name=="PathAdam"):
		return PathOptimizer(input_shape, model, lr)
	elif(name=="Adam"):
		return tf.keras.optimizers.Adam(lr)
	else:
		raise NotImplementedError


class PathOptimizer(OPTIMIZER):
	"""
	This class implements the tensorflow optimizer proposed in https://arxiv.org/abs/1506.02617

	Example:
	>>> import tensorflow as tf
	>>> 
	>>> model=tf.keras.Sequential([tf.keras.layers.Dense(2), tf.keras.layers.Dense(2)])
	>>> input_shape=(10,1)
	>>> x = tf.keras.initializers.GlorotUniform()(input_shape)
	>>> model.build(input_shape)
	>>> 
	>>> #assert computations of gradients
	>>> with tf.GradientTape() as tape:
	>>> 	tape.watch(model.trainable_variables)
	>>> 	y = model(x)
	>>> gradients=tape.gradient(y,model.trainable_variables)
	>>> 
	>>> #clone model and assign its l1 weights
	>>> path_model=tf.keras.models.clone_model(model)
	>>> for param in range(len(model.trainable_variables)):
	>>> 	path_model.trainable_variables[param].assign(tf.abs(model.trainable_variables[param]))
	>>> 
	>>> #compute scale
	>>> with tf.GradientTape() as tape:
	>>> 	tape.watch(path_model.trainable_variables)
	>>> 	y = tf.reduce_sum(path_model(tf.ones(input_shape)))
	>>> path_norm=tape.gradient(y, path_model.trainable_variables)
	>>> 
	>>> #compute ratio
	>>> sgd_norm=0.
	>>> pathsgd_norm=0.
	>>> model_params = model.trainable_variables
	>>> path_params = model.trainable_variables
	>>> for param in range(len(model_params)):
	>>> 	sgd_norm += tf.norm(gradients[param], ord=1)
	>>> 	pathsgd_norm += tf.norm(gradients[param]/path_norm[param], ord=1)
	>>> ratio = ( sgd_norm / pathsgd_norm ) ** 1
	>>> 
	>>> print("Gradients before:", gradients)
	>>> #gradient update
	>>> for param in range(len(model_params)):
	>>> 	gradients[param]=(gradients[param]/path_norm[param])*ratio
	>>> 
	>>> print("Gradients before:", gradients)
	"""

	def __init__(self, input_shape, model, lr, name="PathOptimizer", p=2, **kwargs):

		self.model=model
		self.path_norm=None
		self.ratio=None
		self.input_shape=input_shape
		self.p=p

		super(PathOptimizer, self).__init__(lr, name=name, **kwargs)


	def apply_gradients(self, grads_and_vars, name=None, **kwargs,):
		"""
		Example: 
		>>> import tensorflow as tf
		>>> from path_sgd import PathOptimizer
		>>> 
		>>> model=tf.keras.Sequential([tf.keras.layers.Dense(2), tf.keras.layers.Dense(2)])
		>>> input_shape=(10,1)
		>>> x = tf.keras.initializers.GlorotUniform()(input_shape)
		>>> model.build(input_shape)
		>>> 
		>>> with tf.GradientTape() as tape:
		>>> 	tape.watch(model.trainable_variables)
		>>> 	y = model(x)
		>>> 
		>>> gradients=tape.gradient(y,model.trainable_variables)
		>>> optimizer=PathOptimizer(input_shape, model, 0.01)
		>>> optimizer.apply_gradients(zip(gradients, model.trainable_variables))

		"""

		self.n_params=len(self.model.trainable_variables)

		self.compute_path_norm()
		
		unpacked_gradients=list(zip(*grads_and_vars))
		gradients = list(unpacked_gradients[0])
		variables = list(unpacked_gradients[1])

		if(self.ratio is None):
			#compute ratio
			sgd_norm=0.
			pathsgd_norm=0.
			for param in range(self.n_params):
				sgd_norm += tf.norm(gradients[param], ord=self.p)
				pathsgd_norm += tf.norm(gradients[param]/self.path_norm[param], ord=self.p)
			self.ratio = ( sgd_norm / pathsgd_norm ) ** (1/self.p)

		
		for param in range(self.n_params):
			gradients[param]=(gradients[param]/self.path_norm[param])*self.ratio

		gc.collect()
		
		return super().apply_gradients(zip(gradients, variables), name=name)

	def compute_path_norm(self,):

		#clone model and assign its l1 weights	
		path_model=type(self.model).from_config(self.model.get_config())

		input_shape_tensor=None
		#build input
		if(type(self.input_shape) is list):
			input_shape_tensor=tuple(tf.ones(input_shape) for input_shape in self.input_shape)
			path_model.build(*tuple(input_shape for input_shape in self.input_shape))
		else:
			input_shape_tensor=(tf.ones(self.input_shape),)
			path_model.build(self.input_shape[1:])

		for param in range(len(self.model.variables)):
			if(self.p==1):
				path_model.variables[param].assign((self.model.variables[param]**2)**0.5)
			else:
				path_model.variables[param].assign(self.model.variables[param]**self.p)

		path_model.training=False

		#compute scale
		with tf.GradientTape() as tape:
			tape.watch(path_model.trainable_variables)
			y=path_model(*input_shape_tensor)
			if(type(y) is list):
				y=tf.reduce_sum([tf.reduce_sum(y_i) for y_i in y])
			else:
				y=tf.reduce_sum(y)

		self.path_norm=tape.gradient(y, path_model.trainable_variables)

		del path_model