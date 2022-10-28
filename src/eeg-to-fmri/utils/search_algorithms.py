import tensorflow as tf

import GPyOpt

from utils import train, tf_config, print_utils


class Bayesian_Optimization:
					
	def __init__(self, iterations, model_class, input_shape):
		
		self.iterations = iterations
		self.model_class = model_class
		self.optimizer = tf.keras.optimizers.Adam
		self.input_shape = input_shape
				
	def set_hyperparameters(self, hyperparameters):
		
		self.hyperparameters = hyperparameters
		
	def set_data(self, X_train, X_val, y_train=None, y_val=None):
		self.X_train = X_train
		self.y_train = y_train
		self.X_val = X_val
		self.y_val = y_val
	
	def create_datasets(self, batch_size):
		if(tf.is_tensor(self.y_train)):
			return (tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train)).batch(batch_size), 
				tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val)).batch(1))
		return (tf.data.Dataset.from_tensor_slices(self.X_train).batch(batch_size), 
			tf.data.Dataset.from_tensor_slices(self.X_val).batch(1))


	def optimize(self, hyperparameters):
		hyperparameters = (self.input_shape,) + tuple(hyperparameters[0])
		
		with tf.device('/CPU:0'):
			model = self.model_class.build(*hyperparameters)
			train_set, val_set = self.create_datasets(int(hyperparameters[-1]))

		optimizer = self.optimizer(float(hyperparameters[1]))
		loss_fn = tf.keras.losses.MAE

		#train
		train_loss, val_loss = train.train(train_set, model, optimizer, 
										   loss_fn, epochs=int(hyperparameters[-2]), 
										   val_set=val_set, file_output=self.file_output, verbose=True)

		#optionally plot validation loss to analyze learning curve

		return val_loss[-1]

	def run(self, file_output=None, verbose=False):
		self.file_output = file_output

		optimizer = GPyOpt.methods.BayesianOptimization(f=self.optimize, 
														domain=self.hyperparameters, 
														model_type="GP_MCMC", 
														acquisition_type="EI_MCMC")
		

		print_utils.print_message("Started Optimization Process", file_output=file_output, verbose=verbose)
		
		optimizer.run_optimization(max_iter=self.iterations)

		print_utils.print_message("Finished Optimization Process", file_output=file_output, verbose=verbose)