import tensorflow as tf
from tensorflow import keras

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):        
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)        

    # gets called at each train iteration
    def __call__(self, x): # your custom function here
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

    def get_config(self): # required class method
        return {"num_features": self.num_features,
                "l2reg": self.l2reg}