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

class LinearRegressionGD(EarlyStoppingMixin):
    def __init__(self, eta=0.1, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, x, y, x_val=None, y_val=None, early_stopping=False, patience=10):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=x.shape[1])

        self.b_ = np.float_(0.0)
        self.losses_ = []

        self.epoch = 0
        if early_stopping:
            self._initialize_early_stopping()

        for _ in range(self.n_iter):
            output = self._net_input(x)
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
    
    def predict(self, x):
        return self._net_input(x)

import numpy as np

class RANSACRegressor():
    def __init__(self, base_estimator, max_trials=100, min_samples=None, residual_threshold=1.0, random_state=None):
        self.base_estimator = base_estimator
        self.max_trials = max_trials
        self.min_samples = min_samples
        self.residual_threshold = residual_threshold
        self.random_state = random_state
        self.best_estimator_ = None
        self.best_score_ = float('-inf')
        self.inlier_mask_ = None

    def fit(self, X, y):
        if self.min_samples is None:
            self.min_samples = X.shape[1] + 1

        max_inliers = 0
        rgen = np.random.RandomState(self.random_state)

        for _ in range(self.max_trials):
            sample_idx = rgen.choice(np.arange(X.shape[0]), size=self.min_samples, replace=False)
            X_sample = X[sample_idx]
            y_sample = y[sample_idx]

            self.base_estimator.fit(X_sample, y_sample)

            residuals = y - self.base_estimator.predict(X)

            inlier_mask = np.abs(residuals) < self.residual_threshold
            n_inliers = np.sum(inlier_mask)

            if n_inliers > max_inliers:
                max_inliers = n_inliers
                self.best_estimator_ = self.base_estimator
                self.best_score_ = n_inliers / X.shape[0]
                self.inlier_mask_ = inlier_mask

        if self.best_estimator_:
            self.best_estimator_.fit(X[self.inlier_mask_], y[self.inlier_mask_])

        return self

    def predict(self, X):
        if self.best_estimator_ is None:
            raise ValueError("Model is not fitted yet!")
        return self.best_estimator_.predict(X)
