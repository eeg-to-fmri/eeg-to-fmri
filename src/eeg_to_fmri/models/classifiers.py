import tensorflow as tf

import tensorflow_probability as tfp

from models.eeg_to_fmri import pretrained_EEG_to_fMRI

from layers.bayesian import DenseVariational

class LinearClassifier(tf.keras.Model):
    """
    
    
    """
    def __init__(self, n_classes=1, regularizer=None, variational=False):
        super(LinearClassifier, self).__init__()

        self.training=True
        
        self.flatten = tf.keras.layers.Flatten()
        if(variational):
            self.linear = DenseVariational(n_classes)
        else:
            self.linear = tf.keras.layers.Dense(n_classes, kernel_regularizer=regularizer)
        
    def call(self, X):
        return self.linear(self.flatten(X))


class PolynomialClassifier(tf.keras.Model):
    """
    
    """

    def __init__(self, n_classes=2, degree=3, regularizer=None, variational=False):
        super(PolynomialClassifier, self).__init__()

        self.training=True
        self.degree=degree
        
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



class ViewClassifier(tf.keras.Model):
    """
    classifier of synthesized EEG view

    """
    
    def __init__(self, model, input_shape, degree=1, latent_clf=False, activation=None, regularizer=None, feature_selection=False, segmentation_mask=False, variational=False, seed=None):
        """
        Inputs:
            - EEG_to_fMRI: model
            - tupel: input_shape, eeg input shape
        """
        super(ViewClassifier, self).__init__()

        self.training=True
        self.latent_clf=latent_clf
        
        self.view = pretrained_EEG_to_fMRI(model, input_shape, activation=activation, regularizer=regularizer, feature_selection=feature_selection, segmentation_mask=segmentation_mask, seed=seed)
        if(self.latent_clf):
            self.view=self.view.eeg_encoder

        if(degree==1):
            self.clf = LinearClassifier(variational=variational)
        else:
            self.clf = PolynomialClassifier(degree=degree, variational=variational)
        
    def build(self, input_shape):
        self.view.build(input_shape)
        if(not self.latent_clf):
            if(self.view.aleatoric):
                self.clf.build(self.view.q_decoder.output_shape[:-1]+(2,))#additional dimension for aleatoric uncertainty
            else:
                self.clf.build(self.view.q_decoder.output_shape)
        else:
            self.clf.build(self.view.output_shape)
    
    def call(self, X):
        z=self.view(X)
        if(not self.latent_clf):
            logits=self.clf(z[0])
        else:
            logits=self.clf(z)
        
        if(self.training):
            return [logits]

        return logits


class Contrastive(tf.keras.Model):
        
    def __init__(self, model, input_shape, dimension, latent_clf=False, degree=1, activation=None, regularizer=None, feature_selection=False, segmentation_mask=False, variational=False, seed=None):

        super(Contrastive, self).__init__()

        self.latent_clf=latent_clf

        self.view=pretrained_EEG_to_fMRI(model, input_shape, activation=activation, feature_selection=feature_selection, segmentation_mask=segmentation_mask, seed=seed)
        if(self.latent_clf):
            self.view=self.view.eeg_encoder
        
        self.flatten = tf.keras.layers.Flatten()

    def build(self, input_shape):
        self.view.build(input_shape)

    def call(self, X, training=False):

        x=tf.split(X, 2, axis=1)
        x1, x2=(tf.squeeze(x[0], axis=1), tf.squeeze(x[1], axis=1))

        z1 = self.view(x1)
        z2 = self.view(x2)
        if(not self.latent_clf):
            z1, z2=(z1[0], z2[0])

        return tf.abs(z1-z2)

class ViewContrastiveClassifier(tf.keras.Model):

    def __init__(self, model, input_shape, dimension, latent_clf=False, degree=1, activation=None, regularizer=None, feature_selection=False, segmentation_mask=False, variational=False, seed=None):

        super(ViewContrastiveClassifier, self).__init__()

        self.latent_clf=latent_clf

        self.view=pretrained_EEG_to_fMRI(model, input_shape, activation=activation, feature_selection=feature_selection, segmentation_mask=segmentation_mask, seed=seed)
        if(self.latent_clf):
            self.view=self.view.eeg_encoder

        if(degree==1):
            self.clf = LinearClassifier(variational=variational, regularizer=regularizer)
        else:
            self.clf = PolynomialClassifier(degree=degree, variational=variational, regularizer=regularizer)

    def build(self, input_shape):
        self.view.build(input_shape)
        if(not self.latent_clf):
            if(self.view.aleatoric):
                self.clf.build(self.view.q_decoder.output_shape[:-1]+(2,))#additional dimension for aleatoric uncertainty
            else:
                self.clf.build(self.view.q_decoder.output_shape)
        else:
            self.clf.build(self.view.output_shape)

    def call(self, X, training=False):
        if(training):
            x=tf.split(X, 2, axis=1)
            x1, x2=(tf.squeeze(x[0], axis=1), tf.squeeze(x[1], axis=1))

            z1 = self.view(x1)
            z2 = self.view(x2)

            if(not self.latent_clf):
                z1,z2=(z1[0],z2[0])

            return [tf.abs(z1-z2), self.clf(z1), self.clf(z2)]

        z=self.view(X)
        if(not self.latent_clf):
            z=z[0]
        return self.clf(z)



class ViewLatentContrastiveClassifier(tf.keras.Model):

    def __init__(self, model, input_shape, degree=1, activation=None, regularizer=None, feature_selection=False, segmentation_mask=False, siamese_projection=False, siamese_projection_dimension=10, variational=False, seed=None):

        super(ViewLatentContrastiveClassifier, self).__init__()

        self.siamese_projection=siamese_projection

        self.view=pretrained_EEG_to_fMRI(model, input_shape, activation=activation, feature_selection=feature_selection, segmentation_mask=segmentation_mask, latent_contrastive=True, seed=seed)
        
        if(degree==1):
            self.clf = LinearClassifier(variational=variational, regularizer=regularizer)
        else:
            self.clf = PolynomialClassifier(degree=degree, variational=variational, regularizer=regularizer)

        self.flatten = tf.keras.layers.Flatten()
        
        self.dot = tf.keras.layers.Dot(axes=1, normalize=True)

    def build(self, input_shape):
        self.view.build(input_shape)
        if(self.view.aleatoric):
            self.clf.build(self.view.q_decoder.output_shape[:-1]+(2,))#additional dimension for aleatoric uncertainty
        else:
            self.clf.build(self.view.q_decoder.output_shape)

    def call(self, X, training=False):

        if(training):
            x=tf.split(X, 2, axis=1)
            x1, x2=(tf.squeeze(x[0], axis=1), tf.squeeze(x[1], axis=1))

            z1 = self.view(x1)#returns a list of [fmri view, latent_eeg]
            z2 = self.view(x2)

            s1=self.flatten(z1[1])
            s2=self.flatten(z2[1])

            if(self.siamese_projection):
                ss1=self.flatten(z1[0])
                ss2=self.flatten(z2[0])
                return [tf.abs(s1-s2), 1.-self.dot([ss1,ss2]), self.clf(z1[0].numpy()), self.clf(z2[0].numpy())]

            return [tf.abs(s1-s2), self.clf(z1[0].numpy()), self.clf(z2[0].numpy())]

        return self.clf(self.view(X)[0])