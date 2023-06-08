class BaseLayer:
    def __init__(self, weights=None):
        #used to distinguish trainable from non-trainable layers.
        self.trainable = False
        self.weights = weights