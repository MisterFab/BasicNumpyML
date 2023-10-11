import numpy as np

class NeuralNetMLP:
    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        self.num_classes = num_classes

        rng = np.random.RandomState(random_seed)

        self.weight_h = rng.normal(loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)

        self.weight_out = rng.normal(loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)
    
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500) 
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    
    def _int_to_onehot(self, y, num_labels):
        ary = np.zeros((y.shape[0], num_labels))
        for i, val in enumerate(y):
            ary[i, val] = 1
        return ary
    
    def forward(self, x):
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = self._sigmoid(z_h)

        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = self._sigmoid(z_out)
        return a_h, a_out
    
    def backward(self, x, a_h, a_out, y):
        y_onehot = self._int_to_onehot(y, self.num_classes)

        d_loss__d_a_out = 2.0 * (a_out - y_onehot) / y.shape[0]
        d_a_out__d_z_out = a_out * (1.0 - a_out)
        delta_out = d_loss__d_a_out * d_a_out__d_z_out
        d_z_out__dw_out = a_h
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)

        d_z_out__a_h = self.weight_out
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)
        d_a_h__d_z_h = a_h * (1.0 - a_h)
        d_z_h__d_w_h = x
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss__d_b_h = np.sum(d_loss__a_h * d_a_h__d_z_h, axis=0)

        return d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h