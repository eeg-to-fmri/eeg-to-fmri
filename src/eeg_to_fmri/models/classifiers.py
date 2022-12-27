import tensorflow as tf

import numpy as np

import tensorflow_probability as tfp

from eeg_to_fmri.models.synthesizers import pretrained_EEG_to_fMRI, custom_objects

from eeg_to_fmri.layers.bayesian import DenseVariational

class LinearClassifier(tf.keras.Model):
    """
    
    
    """
    def __init__(self, n_classes=1, regularizer=None, regularizer_const=0., variational=False, aleatoric=False):
        super(LinearClassifier, self).__init__()

        self.training=True

        self.aleatoric=aleatoric
        self.variational=variational
        self.n_classes=n_classes
        self.regularizer=regularizer
        self.regularizer_const=regularizer_const

        if(type(self.regularizer) is str):
            assert self.regularizer in ["L1", "L2"]
            regularizer=getattr(tf.keras.regularizers, self.regularizer)(l=self.regularizer_const)

        #layers
        self._layers=[tf.keras.layers.Flatten()]
        if(self.variational):
            self._layers+=[DenseVariational(n_classes)]
        else:
            self._layers+=[tf.keras.layers.Dense(n_classes, kernel_regularizer=regularizer)]
        if(self.aleatoric):
            self._layers+=[tf.keras.layers.Dense(n_classes, activation=tf.keras.activations.exponential)]

    def build(self, input_shape):
        _input_shape = tf.keras.layers.Input(shape=input_shape)
        x=_input_shape
        for layer in self._layers[:-1]:
            x=layer(x)

        if(self.aleatoric):
            self.model=tf.keras.Model(_input_shape, x)
            self.aleatoric_model=tf.keras.Model(_input_shape, self._layers[-1](x))
            self.aleatoric_model.build(input_shape)
        else:
            self.model=tf.keras.Model(_input_shape, self._layers[-1](x))

        self.model.build(input_shape)

        self.built=True
        
    def call(self, X, training=False):
        if(self.aleatoric and (self.training or training)):
            """
            self.training - is set after training
            training - is set when the LinearClassifier is called from ViewLatentContrastiveClassifier
            """
            return tf.concat([tf.expand_dims(self.model(X),axis=-1), tf.expand_dims(self.aleatoric_model(X), axis=-1)], axis=-1)

        return self.model(X)

    def get_config(self,):

        return {"aleatoric": self.aleatoric,
                "variational": self.variational,
                "n_classes": self.n_classes,
                "regularizer": self.regularizer,
                "regularizer_const": self.regularizer_const,}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class PolynomialClassifier(tf.keras.Model):
    """
    
    """

    def __init__(self, n_classes=2, degree=3, regularizer=None, variational=False, aleatoric=False, **kwargs):
        super(PolynomialClassifier, self).__init__()

        self.training=True
        self.degree=degree
        self.aleatoric=aleatoric
        
        self.flatten = tf.keras.layers.Flatten()

        if(variational):
            self.linear = DenseVariational(n_classes, use_bias=False)
        else:
            self.linear = tf.keras.layers.Dense(n_classes, use_bias=False, kernel_regularizer=regularizer)
        
    def call(self, X):
        X = tf.expand_dims(X, -1)
        x = [X**0, X]

        for p in range(self.degree-1):
            x+=[x[-1]*X]

        return self.linear(self.flatten(tf.concat(x, -1)))


class ViewLatentContrastiveClassifier(tf.keras.Model):

    def __init__(self, path_network, input_shape, degree=1, activation=None, regularizer=None, regularizer_const=0., variational=False, aleatoric=False, organize_channels=False, seed=None, **kwargs):

        super(ViewLatentContrastiveClassifier, self).__init__(**kwargs)

        self.training=True

        self.path_network=path_network
        self._input_shape=input_shape
        self.degree=degree
        self.activation=activation
        self.regularizer=regularizer
        self.regularizer_const=regularizer_const
        self.variational=variational
        self.organize_channels=organize_channels
        self.aleatoric=aleatoric
        self.seed=seed

        #prepare string regularizers
        if(type(self.activation) is str):
            assert self.activation in ["linear", "relu"]
            activation=getattr(tf.keras.activations, self.activation)
        if(type(self.regularizer) is str):
            assert self.regularizer in ["L1", "L2"]
            regularizer=getattr(tf.keras.regularizers, self.regularizer)(l=self.regularizer_const)

        self.view=pretrained_EEG_to_fMRI(tf.keras.models.load_model(path_network, custom_objects=custom_objects, compile=False), self._input_shape, activation=activation, latent_contrastive=True, organize_channels=organize_channels, seed=seed)
        
        if(degree==1):
            self.clf = LinearClassifier(variational=self.variational, regularizer=regularizer, aleatoric=self.aleatoric)
        else:
            self.clf = PolynomialClassifier(degree=self.degree, variational=self.variational, regularizer=regularizer, aleatoric=self.aleatoric)

        self.flatten = tf.keras.layers.Flatten()
        
        self.dot = tf.keras.layers.Dot(axes=1, normalize=False)

    def build(self, input_shape):
        self.view.build(input_shape)
        self.clf.build(self.view.q_decoder.output_shape[1:])

        self.built=True

    def call(self, X):

        if(self.training):
            x=tf.split(X, 2, axis=1)
            x1, x2=(tf.squeeze(x[0], axis=1), tf.squeeze(x[1], axis=1))

            z1 = self.view(x1, training=self.training)#returns a list of [fmri view, latent_eeg]
            z2 = self.view(x2, training=self.training)

            s1=self.flatten(z1[1])
            s2=self.flatten(z2[1])

            return [(z1[0],z2[0]), tf.abs(s1-s2), self.clf(z1[0].numpy(), training=self.training), self.clf(z2[0].numpy(), training=self.training)]

        #also when training only for classification
        return self.clf(self.view(X, training=self.training)[0], training=self.training)

    def get_config(self,):

        return {"path_network": self.path_network,
                "input_shape": self._input_shape,
                "degree": self.degree,
                "activation": self.activation,
                "regularizer": self.regularizer,
                "regularizer_const": self.regularizer_const,
                "variational": self.variational,
                "organize_channels": self.organize_channels,
                "aleatoric": self.aleatoric,
                "seed": self.seed,}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ViewLatentLikelihoodClassifier(ViewLatentContrastiveClassifier):
    """
    Classsifies without the contrastive mechanism
    Loss at the latent representation is a binary cross entropy mechanism to separate the sinusoids according to the ground truth label

    """

    def __init__(self, **kwargs):

        super(ViewLatentLikelihoodClassifier, self).__init__(**kwargs)


    def call(self, X):

        if(self.training):
            z = self.view(X, training=self.training)#returns a list of [fmri view, latent_eeg]
            s=self.flatten(z[1])-np.pi/2
            s=s/tf.norm(s.numpy(), ord=2)
            return [s, self.clf(z[0], training=self.training)]

        return super().call(X)


    def get_config(self,):
        
        return super().get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)
