import tensorflow as tf

import numpy as np

from eeg_to_fmri.layers.topographical_attention import Topographical_Attention


def explain(explainer, dataset, eeg=True, eeg_attention=False, fmri=False, verbose=False):
	R = None

	if(eeg_attention):
		explainer.eeg_attention=eeg_attention

	if(eeg and not fmri):
		index=0
	elif(not eeg and fmri):
		index=1

	instance = 1
	for X in dataset.repeat(1):
		if(R is None):
			R = explainer(X[index]).numpy()
		else:
			R = np.append(R, explainer(X[index]).numpy(), axis=0)
		
		if(verbose):
			print("Explaining instance", str(instance), end="\r")
		instance+=1

	return R


def lrp(x, y, layer, multiply=None):
	"""
	lrp - Layer-wise  Relevance Propagation
		Inputs:
			* x - tf.Tensor input of layer
			* y - tf.Tensor output of layer
			* layer - tf.keras.layers.Layer layer
		Outputs:
			* tf.Tensor - containing relevances
	"""
	if(type(layer) is tf.keras.layers.Reshape or type(layer) is tf.keras.layers.BatchNormalization):
		return tf.reshape(y, x.shape)

	with tf.GradientTape(watch_accessed_variables=False) as tape:
		if(type(x) is list):
			with tf.GradientTape(watch_accessed_variables=False) as tape1:
				tape.watch(x[0])
				tape1.watch(x[1])

				z = layer(x)+1e-9
				s = y/tf.reshape(z, y.shape)
				s = tf.reshape(s, z.shape)


				R = x[0]*tape.gradient(tf.reduce_sum(z*s.numpy()), x[0]) + x[1]*tape1.gradient(tf.reduce_sum(z*s.numpy()), x[1])
		else:
			tape.watch(x)
			if("multiply" in layer.name):
				z = layer(x, multiply)+1e-9
			else:
				z = layer(x)+1e-9
			s = y/tf.reshape(z, y.shape)
			s = tf.reshape(s, z.shape)
			
			c = tape.gradient(tf.reduce_sum(z*s.numpy()), x)
			R = x*c

	return R



class LRP_EEG(tf.keras.layers.Layer):
	"""
	LRP_EEG: propagates relevances through a model of type models.eeg_to_fmri.EEG_to_fMRI
	"""
	
	def __init__(self, model, attention=False, conditional_attention_style=False):
		"""
			Inputs:
				* model: models.eeg_to_fmri.EEG_to_fMRI.decoder
		"""
		assert type(model).__name__=="tf.keras.Model" or type(model).__name__=="Functional"
		
		self.model = model
		self.eeg_attention = attention
		self.layer_bias=0
		self.explain_conditional=conditional_attention_style

		super(LRP_EEG, self).__init__()
		
	
	def forward(self, X):
		"""
			Inputs:
				* X: list(tf.Tensor)
			Outputs:
				* tf.Tensor - output of model
		"""

		self.activations = []
		z = X
		
		if(self.explain_conditional):
			self.conditional_activations=[]
			for layer in self.model.layers:
				if("topo" in layer.name):
					self.conditional_activations+=[z]
					z,attention_scores=layer(z)
					self.conditional_activations+=[attention_scores]
				elif("conditional_attention_style_flatten" in layer.name):
					attention_scores_flatten=layer(attention_scores)
					self.conditional_activations+=[attention_scores_flatten]
				elif("conditional_attention_style_dense" in layer.name):
					attention_scores_dense=layer(attention_scores_flatten)
					self.conditional_activations+=[attention_scores_dense]
				elif("multiply" in layer.name):
					z = layer(z, attention_scores_dense)
				else:
					z = layer(z)
				
				self.activations += [z]

		else:
			
			for layer in self.model.layers:
				#we are ignoring the relevance of the attention scores through the conditional style flow
				if("conditional_attention_style_flatten" in layer.name):
					attention_scores_flatten=layer(attention_scores)
					self.layer_bias+=1
					continue
				if("conditional_attention_style_dense" in layer.name):
					self.attention_scores=layer(attention_scores_flatten)
					self.layer_bias+=1
					continue
				if("topo" in layer.name):
					z,attention_scores=layer(z)
				elif("multiply" in layer.name):
					z = layer(z, self.attention_scores)
				else:
					z = layer(z)
				self.activations += [z]
		
		return z
	
	
	def propagate(self, X, R, model, activations):
		"""
			Inputs:
				* X - tf.Tensor
				* R - tf.Tensor
				* model - eeg_to_fmri.EEG_to_fMRI
				* activations - list
			Outputs: 
				* tf.Tensor
		"""

		if(self.explain_conditional):
			#we are ignoring the relevance of the attention scores through the conditional style flow
			for layer in range(len(model.layers))[::-1]:
				decoder=True
				if("conditional_attention_style_flatten" in model.layers[layer].name):
					return lrp(self.conditional_activations[1], R_conditional, model.layers[layer])
				elif("conditional_attention_style_dense" in model.layers[layer].name):
					R_conditional = lrp(self.conditional_activations[2], R_conditional, model.layers[layer])
				elif("multiply" in model.layers[layer].name):
					R_conditional = lrp(self.conditional_activations[3], R, model.layers[layer], multiply=activations[layer-self.layer_bias-1])
					decoder=False
				elif(decoder):
					R = lrp(activations[layer-self.layer_bias-1], R, model.layers[layer])		
		else:
			#we are ignoring the relevance of the attention scores through the conditional style flow
			for layer in range(len(model.layers))[::-1]:
				if("conditional_attention_style" in model.layers[layer].name):
					self.layer_bias-=1
					continue
				if(self.eeg_attention and hasattr(model.layers[layer], "lrp_attention")):
					return model.layers[layer].lrp_attention(activations[layer-self.layer_bias-1], R)
				elif(hasattr(model.layers[layer], "lrp")):
					R = model.layers[layer].lrp(activations[layer-self.layer_bias-1], R)
				else:
					if(layer-1 >= 0):
						if(self.eeg_attention and type(model.layers[layer]) is Topographical_Attention):
							return model.layers[layer].lrp_attention(activations[layer-self.layer_bias-1], R)
						if("multiply" in model.layers[layer].name):
							R = lrp(activations[layer-self.layer_bias-1], R, model.layers[layer], multiply=self.attention_scores)
							continue
						R = lrp(activations[layer-self.layer_bias-1], R, model.layers[layer])
					else:
						R = lrp(X, R, model.layers[layer])
			
		return R
			
	
	def backward(self, X, R):
		"""
			Inputs:
				* X - tf.Tensor
				* R - tf.Tensor
			Outputs: 
				* tf.Tensor
		"""
		
		return self.propagate(X, R, self.model, self.activations)
		

	
	def call(self, X):
		"""
			Inptus
				* X - tf.Tensor
			Outputs: 
				* tf.Tensor
		"""
		
		y = self.forward(X)
		
		if(self.eeg_attention):
			assert type(self.model.layers[2]) is Topographical_Attention
			return self.backward(X, y)
		return self.backward(X, y)

	
class LRP(tf.keras.layers.Layer):
	
	def __init__(self, model):
		"""
			Inputs:
				* model: tf.keras.Model
		"""
		super(LRP, self).__init__()
		
		self.model = model
		
	
	def forward(self, X):
		"""
			Inputs:
				* X: tf.Tensor
			Outputs:
				* tf.Tensor - output of model
		"""

		self.activations = []

		z = X
		#forward pass
		for layer in self.model.layers:
			z = layer(z)
			self.activations += [z]
			
		return z
	
	
	def propagate(self, X, R, model, activations):
		"""
			Inputs:
				* X - tf.Tensor
				* R - tf.Tensor
				* model - tf.keras.Model
				* activations - list
			Outputs: 
				* tf.Tensor
		"""
		for layer in range(len(model.layers))[::-1]:
			if(hasattr(model.layers[layer], "lrp")):
				R = model.layers[layer].lrp(activations[layer-1], R)
			else:
				if(layer-1 >= 0):
					R = lrp(activations[layer-1], R, model.layers[layer])
				else:
					R = lrp(X, R, model.layers[layer])
		
		return R
			
	
	def backward(self, X, R):
		"""
			Inputs:
				* X - tf.Tensor
				* R - tf.Tensor
			Outputs: 
				* tf.Tensor
		"""
		
		return self.propagate(X, R, self.model, self.activations)

	
	def call(self, X):
		"""
			Inptus
				* X - tf.Tensor
			Outputs: 
				* tf.Tensor
		"""
		
		y = self.forward(X)
		
		return self.backward(X, y)