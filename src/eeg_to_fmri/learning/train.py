import tensorflow as tf

import gc

import numpy as np

from eeg_to_fmri.utils import print_utils

def apply_gradient(model, optimizer, loss_fn, x, y, return_logits=False, call_fn=None):
    with tf.GradientTape(persistent=True) as tape:
        if(type(x) is tuple):
            if(call_fn is None):
                logits=model(*x, training=True)
            else:
                logits=call_fn(model, *x)
        else:
            if(call_fn is None):
                logits=model(x, training=True)
            else:
                raise NotImplementedError
        regularization=0.
        if(len(model.losses)):
            regularization=tf.math.add_n(model.losses)
        loss = loss_fn(y, logits)+regularization
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    if(return_logits):
        return tf.reduce_mean(loss), logits[0]
    return tf.reduce_mean(loss)

def train_step(model, x, optimizer, loss_fn, u_architecture=False, return_logits=False, call_fn=None):
    """Executes one training step and returns the loss.

    This function computes the loss and gradients, and uses the latter to
    update the model's parameters.
    """
    if(tf.is_tensor(x)):
        return apply_gradient(model, optimizer, loss_fn, x, x, return_logits=return_logits, call_fn=call_fn)
    elif(u_architecture):
        return apply_gradient(model, optimizer, loss_fn, x, x[1], return_logits=return_logits, call_fn=call_fn)
    else:
        return apply_gradient(model, optimizer, loss_fn, *x, return_logits=return_logits, call_fn=call_fn)

def evaluate(X, model, loss_fn, u_architecture=False):
    loss = 0.0
    n_batches = 0
    for batch_x in X.repeat(1):
        if(u_architecture):
            loss += tf.reduce_mean(loss_fn(batch_x[1], model(batch_x, training=True))).numpy()
        elif(type(batch_x) is tuple):
            loss += tf.reduce_mean(loss_fn(batch_x[1], model(batch_x[0], training=False))).numpy()
        else:
            loss += tf.reduce_mean(loss_fn(batch_x, model(batch_x, training=False))).numpy()
        n_batches += 1
    
    return loss/n_batches

def evaluate_parameters(X, model, u_architecture=False):
    parameters = [0.0, 0.0]
    n_batches = 0
    for batch_x in X.repeat(1):
        if(u_architecture):
            prediction = model(batch_x, training=True)
            batch_x=batch_x[1]
        elif(type(batch_x) is tuple):
            prediction = model(batch_x[0], training=False)
        else:
            prediction = model(batch_x, training=False)
        
        parameters[0] += tf.reduce_mean(prediction[1]).numpy()
        if(len(prediction) > 2):
            parameters[1] += tf.reduce_mean(prediction[2]).numpy()

        n_batches += 1
    
    return (parameters[0]/n_batches, parameters[1]/n_batches)

def evaluate_l2loss(X, model, u_architecture=False):
    l2loss = 0.0
    n_batches = 0
    for batch_x in X.repeat(1):
        if(u_architecture):
            prediction = model(batch_x, training=True)
            batch_x=batch_x[1]
        elif(type(batch_x) is tuple):
            prediction = model(batch_x[0], training=False)
        else:
            prediction = model(batch_x, training=False)
        
        l2loss += tf.reduce_mean((batch_x - prediction[0])**2).numpy()
        
        n_batches += 1
    
    return l2loss/n_batches


def evaluate_additional(X, model, additional_losses):
    losses = np.zeros(len(additional_losses))
    n_batches = 0
    for batch_x in X.repeat(1):
        i = 0
        for loss_fn in additional_losses:
            if(type(batch_x) is tuple):
                prediction = model(batch_x[0], training=False)
            else:
                prediction = model(batch_x, training=False)
            
            losses[i] += tf.reduce_mean(loss_fn(batch_x, prediction[0])).numpy()
            
            i +=1
        n_batches += 1
        
    return (losses/n_batches).tolist()

def train(train_set, model, opt, loss_fn, epochs=10, val_set=None, u_architecture=False, additional_losses=[], file_output=None, verbose=False, verbose_batch=False):
    val_loss = np.empty((0,), dtype=np.float32)
    train_loss = np.empty((0,), dtype=np.float32)

    for epoch in range(epochs):

        loss = 0.0
        n_batches = 0
        
        for batch_set in train_set.repeat(1):
            batch_loss = train_step(model, batch_set, opt, loss_fn, u_architecture=u_architecture).numpy()
            loss += batch_loss
            n_batches += 1
            gc.collect()

            print_utils.print_message("Batch "+str(n_batches)+" with loss: " + str(batch_loss), file_output=file_output, verbose=verbose_batch, end="\r")

        if(val_set is not None):
            val_loss=np.append(val_loss,[evaluate(val_set, model, loss_fn, u_architecture=u_architecture)], axis=0)
        
        train_loss=np.append(train_loss,[(loss/n_batches)], axis=0)

        print_utils.print_message("Epoch " + str(epoch+1) + " with loss: " + str(train_loss[-1]), file_output=file_output, verbose=verbose)

    del train_loss, val_loss