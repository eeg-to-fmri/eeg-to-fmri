import tensorflow as tf
import tensorflow_probability as tfp

from numpy.random import gamma

import math

import numpy as np
"""
Loss combinating aleatoric and epistemic_uncertainty
"""
def combined_log_loss(y_true, y_pred):
	variance = tf.math.square(y_pred[1])+1e-9
	
	return tf.reduce_mean((tf.exp(-tf.math.log(variance))*(y_pred[0] - y_true)**2)/2 + (tf.math.log(variance))/2, axis=(1,2,3))

def combined_original_loss(y_true, y_pred):
	variance = tf.math.square(y_pred[1])+1e-9
	
	return tf.reduce_mean(((1/variance)*(y_pred[0] - y_true)**2)/2 + (tf.math.log(variance))/2, axis=(1,2,3))

def combined_square_loss(y_true, y_pred):
	variance = tf.math.square(y_pred[1])
	
	return tf.reduce_mean(((1/(variance+1e-9))*(y_pred[0] - y_true)**2)/2 + variance/2, axis=(1,2,3))

def combined_log_abs_loss(y_true, y_pred):
	variance = tf.math.abs(y_pred[1])
	
	return tf.reduce_mean(((1/variance+1e-9)*(y_pred[0] - y_true)**2)/2 + (tf.math.log(variance))/2, axis=(1,2,3))

def combined_abs_loss(y_true, y_pred):
	variance = tf.math.abs(y_pred[1])
	
	return tf.reduce_mean(((1/variance+1e-9)*(y_pred[0] - y_true)**2)/2 + variance/2, axis=(1,2,3))

def combined_abs_diff_log_loss(y_true, y_pred):
	variance = tf.math.abs(y_pred[1])
	
	return tf.reduce_mean((-variance*(y_pred[0] - y_true)**2)/2 + (tf.math.log(variance))/2, axis=(1,2,3))

def combined_abs_diff_loss(y_true, y_pred):
	variance = tf.math.abs(y_pred[1])
	
	return tf.reduce_mean((variance*(y_pred[0] - y_true)**2)/2 + variance/2, axis=(1,2,3))

def combined_abs_diff_sym_loss(y_true, y_pred):
	variance = tf.math.abs(y_pred[1])
	
	return tf.reduce_mean((variance*(y_pred[0] - y_true)**2)/2 - variance/2, axis=(1,2,3))

def combined_abs_non_balanced_loss(y_true, y_pred):
	variance = tf.math.abs(y_pred[1])
	
	return tf.reduce_mean(((variance-variance**2)*(y_pred[0] - y_true)**2)/2 + (variance**2)/2, axis=(1,2,3))

def combined_abs_balanced_loss(y_true, y_pred):
	variance = tf.math.abs(y_pred[1])
	
	return tf.reduce_mean(((variance-variance**2)*(y_pred[0] - y_true)**2)/2 + (variance**2-variance)/2, axis=(1,2,3))

def epistemic_log_loss(y_true, y_pred):
	#compute variance
	variance = tf.math.sqrt(tf.reduce_mean(y_pred**2, axis=(1,2,3))-tf.reduce_mean(y_pred, axis=(1,2,3))**2)+1e-9
	
	return tf.reduce_mean((tf.exp(-tf.math.log(variance))*(y_pred - y_true)**2)/2 + (tf.math.log(variance))/2, axis=(1,2,3))

def epistemic_original_loss(y_true, y_pred):
	variance = tf.math.sqrt(tf.reduce_mean(y_pred**2, axis=(1,2,3))-tf.reduce_mean(y_pred, axis=(1,2,3))**2)+1e-9
	
	return tf.reduce_mean(((1/variance)*(y_pred - y_true)**2)/2 + (tf.math.log(variance))/2, axis=(1,2,3))

def epistemic_abs_diff_loss(y_true, y_pred):
	variance = tf.math.sqrt(tf.reduce_mean(y_pred**2, axis=(1,2,3))-tf.reduce_mean(y_pred, axis=(1,2,3))**2)+1e-9
	
	return tf.reduce_mean((-variance*(y_pred[0] - y_true)**2)/2 + variance/2, axis=(1,2,3))

def gamma_prior_loss(y_true, y_pred):
    l2_dist = ((y_pred[0] - y_true)**2)/2
    beta = tf.math.abs(y_pred[2])+1e-9#y_pred[2]
    alpha = tf.math.abs(y_pred[1])+1e-9#y_pred[1]
    abs_y_true = tf.math.abs(y_true)#y true can be negative and log of negative is not defined
    
    return tf.reduce_mean((beta+l2_dist)*abs_y_true + (alpha+1)*tf.math.log((beta+l2_dist)*abs_y_true), axis=(1,2,3))
    
class extended_balance:
	def __init__(self, K):
		self.K = K

	def combined_abs_non_balanced_loss(self, y_true, y_pred):
		variance = tf.math.abs(self.K*y_pred[1])
		
		return tf.reduce_mean(((variance-variance**2)*(y_pred[0] - y_true)**2)/2 + (variance**2)/4, axis=(1,2,3))

	def combined_abs_balanced_loss(self, y_true, y_pred):
		variance = tf.math.abs(self.K*y_pred[1])
		
		return tf.reduce_mean(((variance-variance**2)*(y_pred[0] - y_true)**2)/2 + (variance**2-variance)/4, axis=(1,2,3))


"""
Computing the prior of lambda = 1/sigma^2 using a gamma distribution parametrized by alpha and beta
for the estimation of the true parameter sigma^2 
"""
def MC_posterior_Gamma(alpha, beta):
	#alpha, beta > 0
	alpha = np.abs(alpha)+1e-9
	beta = np.abs(beta)+1e-9
	
	return gamma(alpha, 1/beta)


"""
Computing \sigma_{i}^{2}
"""
def aleatoric_uncertainty(model, X, T=10):
	
	y_std = tf.zeros(X[1].shape)#shape of fmri
	
	for i in range(T):
		y_t = model(*X)

		if(type(y_t[0]) is list):
			y_t = y_t[0]

		y_std = y_std + tf.math.square(y_t[1])
		
	return y_std/T

"""
Computing Var(y*)
"""
def epistemic_uncertainty(model, X, T=10):

	assert type(X) is tuple
	
	y_hat = tf.zeros(X[1].shape)#shape of fmri
	
	for i in range(T):

		y_t = model(*X)

		if(type(y_t[0]) is list):
			y_t = y_t[0][0]
		else:
			y_t = y_t[0]

		y_hat = y_hat + tf.math.square(y_t)
		
	return y_hat*(1/T) - tf.math.square(X[1][0])#missing empirical variance \sigma


"""
Computing E(y*)
"""
def predict_MC(model, X, T=10):

	assert type(X) is tuple
	
	y_hat = tf.zeros(X[1].shape)#shape of fmri
	
	for i in range(T):

		y_t = model(*X)

		if(type(y_t[0]) is list):
			y_t = y_t[0][0]
		else:
			y_t = y_t[0]

		y_hat = y_hat + y_t
		
	return y_hat*(1/T)