import numpy as np

class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class DecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        
    def _gini(self, y):
        m = len(y)
        return 1.0 - sum([(np.sum(y == c) / m) ** 2 for c in np.unique(y)])
        
    def _best_split(self, X, y):
        m, n = X.shape
        if m <= 1:
            return None, None
        num_parent = [np.sum(y == c) for c in np.unique(y)]
        best_gini = 1.0 - sum((num / m) ** 2 for num in num_parent)
        best_idx, best_thr = None, None
        for idx in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * len(np.unique(y))
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[np.unique(y).tolist().index(c)] += 1
                num_right[np.unique(y).tolist().index(c)] -= 1
                gini_left = 1.0 - sum(
                    (num_left[np.unique(y).tolist().index(x)] / i) ** 2 for x in np.unique(y)
                )
                gini_right = 1.0 - sum(
                    (num_right[np.unique(y).tolist().index(x)] / (m - i)) ** 2 for x in np.unique(y)
                )
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr
    
    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(predicted_class=predicted_class)
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _train_with_depth(self, X_train, y_train, X_val, y_val, depth):
        tree = self._grow_tree(X_train, y_train, depth=depth)
        predictions = [self._predict_sample(tree, x) for x in X_val]
        accuracy = np.mean(predictions == y_val)
        return accuracy, tree

    def fit(self, X, y, X_val=None, y_val=None, max_depth_range=range(1, 11)):
        best_accuracy = -1
        best_tree = None
        X_train, y_train = X, y
        
        for depth in max_depth_range:
            accuracy, tree = self._train_with_depth(X_train, y_train, X_val, y_val, depth)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.best_max_depth = depth
                best_tree = tree
        
        print("Best max depth:", self.best_max_depth)
        self.tree_ = best_tree

    def _predict_sample(self, node, x):
        if node.left is None and node.right is None:
            return node.predicted_class
        if x[node.feature_index] < node.threshold:
            return self._predict_sample(node.left, x)
        else:
            return self._predict_sample(node.right, x)
        
    def predict(self, X):
        return [self._predict_sample(self.tree_, x) for x in X]