import numpy as np


class SimpleID3:
    """
    Arbore de decizie bazat pe castigul de informatie
    """

    def __init__(self, depth_limit=5):
        self.depth_limit = depth_limit
        self.tree_structure = None

    def _calc_entropy(self, target):
        counts = np.bincount(target)
        probs = counts / len(target)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def _get_split(self, X, y):
        best_gain = -1
        split_col, split_val = None, None

        for col in range(X.shape[1]):
            vals = np.unique(X[:, col])
            for v in vals:
                # Split binar
                mask = X[:, col] <= v
                y_left, y_right = y[mask], y[~mask]

                if len(y_left) == 0 or len(y_right) == 0: continue

                gain = self._calc_entropy(y) - (
                        (len(y_left) / len(y)) * self._calc_entropy(y_left) +
                        (len(y_right) / len(y)) * self._calc_entropy(y_right)
                )

                if gain > best_gain:
                    best_gain, split_col, split_val = gain, col, v
        return split_col, split_val

    def _grow(self, X, y, current_depth):
        if len(np.unique(y)) == 1 or current_depth >= self.depth_limit:
            return np.bincount(y).argmax()

        col, val = self._get_split(X, y)
        if col is None: return np.bincount(y).argmax()

        left_idx = X[:, col] <= val
        return {
            'col': col, 'val': val,
            'left': self._grow(X[left_idx], y[left_idx], current_depth + 1),
            'right': self._grow(X[~left_idx], y[~left_idx], current_depth + 1)
        }

    def fit(self, X, y):
        self.tree_structure = self._grow(X, y, 0)

    def _traverse(self, x, node):
        if not isinstance(node, dict): return node
        if x[node['col']] <= node['val']:
            return self._traverse(x, node['left'])
        return self._traverse(x, node['right'])

    def predict(self, X):
        return np.array([self._traverse(x, self.tree_structure) for x in X])