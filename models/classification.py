import numpy as np

class EarlyStoppingMixin:
    def _initialize_early_stopping(self):
        self.best_loss = float('inf')
        self.best_weights = None
        self.best_bias = None
        self.no_improvement_count = 0

    def _early_stopping_check(self, x_val, y_val, patience):
        val_loss = self._compute_loss(x_val, y_val)
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_weights = np.copy(self.w_)
            self.best_bias = np.copy(self.b_)
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count >= patience:
            print("Early stopping after epoch:", self.epoch, "with validation loss:", val_loss)
            self.w_ = self.best_weights
            self.b_ = self.best_bias
            return True
        return False

class Perceptron(EarlyStoppingMixin):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, x, y, x_val=None, y_val=None, early_stopping=False, patience=10):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=x.shape[1])

        self.b_ = np.float_(0.0)

        if early_stopping:
            self._initialize_early_stopping()

        for _ in range(self.n_iter):
            for xi, target in zip(x, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
            if early_stopping and x_val is not None and y_val is not None:
                if self._early_stopping_check(x_val, y_val, patience):
                    break
        return self

    def _compute_loss(self, X, y):
        predictions = self.predict(X)
        errors = self.eta * (y - predictions)
        return np.mean(errors**2)
    
    def _net_input(self, x):
        return np.dot(x, self.w_) + self.b_
    
    def predict(self, x):
        return np.where(self._net_input(x) >= 0.0, 1, 0)

class AdalineGD(EarlyStoppingMixin):
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, x, y, x_val=None, y_val=None, early_stopping=False, patience=10):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=x.shape[1])

        self.b_ = np.float_(0.0)
        self.losses_ = []

        if early_stopping:
            self._initialize_early_stopping()

        for _ in range(self.n_iter):
            net_input = self._net_input(x)
            output = self._activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * x.T.dot(errors) / x.shape[0]
            self.b_ += self.eta * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
            if early_stopping and x_val is not None and y_val is not None:
                if self._early_stopping_check(x_val, y_val, patience):
                    break
        return self
    
    def _compute_loss(self, X, y):
        predictions = self.predict(X)
        errors = y - predictions
        return np.mean(errors**2)

    def _net_input(self, x):
        return np.dot(x, self.w_) + self.b_
    
    def _activation(self, x):
        return x
    
    def predict(self, x):
        return np.where(self._activation(self._net_input(x)) >= 0.5, 1, 0)
    
class AdalineSGD(EarlyStoppingMixin):
    def __init__(self, eta=0.01, n_iter=50, shuffle=True, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state
    
    def fit(self, x, y, x_val=None, y_val=None, early_stopping=False, patience=10):
        self._initialize_weights(x.shape[1])
        self.losses_ = []

        if early_stopping:
            self._initialize_early_stopping()

        for _ in range(self.n_iter):
            if self.shuffle:
                x, y = self._shuffle(x, y)
            losses = []
            for xi, target in zip(x, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)
            if early_stopping and x_val is not None and y_val is not None:
                if self._early_stopping_check(x_val, y_val, patience):
                    break
        return self
    
    def _compute_loss(self, X, y):
        predictions = self.predict(X)
        errors = y - predictions
        return np.mean(errors**2)

    def partial_fit(self, x, y):
        if not self.w_initialized:
            self._initialize_weights(x.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(x, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(x, y)
        return self
    
    def _shuffle(self, x, y):
        r = self.rgen.permutation(len(y))
        return x[r], y[r]
    
    def _initialize_weights(self, m):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float_(0.0)
        self.w_initialized = True
    
    def _update_weights(self, xi, target):
        output = self._activation(self._net_input(xi))
        error = target - output
        self.w_ += self.eta * 2.0 * xi * error
        self.b_ += self.eta * 2.0 * error
        loss = error**2
        return loss
    
    def _net_input(self, x):
        return np.dot(x, self.w_) + self.b_
    
    def _activation(self, x):
        return x
    
    def predict(self, x):
        return np.where(self._activation(self._net_input(x)) >= 0.5, 1, 0)
    
class LogisticRegressionGD(EarlyStoppingMixin):
    def __init__(self, eta=0.01, n_iter=50, random_state=1, C=1.0):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.C = C
    
    def fit(self, x, y, x_val=None, y_val=None, early_stopping=False, patience=10):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=x.shape[1])
        self.b_ = np.float_(0.0)
        self.losses_ = []

        if early_stopping:
            self._initialize_early_stopping()

        for _ in range(self.n_iter):
            net_input = self._net_input(x)
            output = self._activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * (x.T.dot(errors) / x.shape[0] - (1.0 / self.C) * self.w_)
            self.b_ += self.eta * errors.mean()
            loss = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))) / x.shape[0])
            loss += (1 / (2 * self.C)) * np.dot(self.w_, self.w_)
            self.losses_.append(loss)
            if early_stopping and x_val is not None and y_val is not None:
                if self._early_stopping_check(x_val, y_val, patience):
                    break
        return self
    
    def _compute_loss(self, X, y):
        predictions = self.predict(X)
        errors = y - predictions
        return np.mean(errors**2)

    def _net_input(self, x):
        return np.dot(x, self.w_) + self.b_
    
    def _activation(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))
    
    def predict(self, x):
        return np.where(self._activation(self._net_input(x)) >= 0.5, 1, 0)

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y, X_val, y_val):
        self.X_train = X
        self.y_train = y
        self._tune_parameters(X_val, y_val)

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _tune_parameters(self, X_val, y_val):
        best_accuracy = 0
        best_k = self.k

        for i in range(1, 10):
            self.k = i
            y_pred = self.predict(X_val)
            accuracy = np.mean(y_val == y_pred)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = i
        
        self.k = best_k
        print("Best k:", best_k)

    def _predict(self, x):
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        most_common = np.bincount(k_nearest_labels).argmax()
        return most_common