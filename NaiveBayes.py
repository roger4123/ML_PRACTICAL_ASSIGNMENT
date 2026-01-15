import numpy as np


class CustomNaiveBayes:
    """
    Clasificator bazat pe teorema lui Bayes, optimizat pentru date de tip basket
    """

    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.priors = {}
        self.likelihoods = {}

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]
            # Probabilitatea a priori a clasei P(C)
            self.priors[c] = X_c.shape[0] / n_samples

            # Probabilități condiționate cu Laplace: P(xi | C)
            feature_counts = np.sum(X_c, axis=0)
            total_feature_count = np.sum(feature_counts)

            self.likelihoods[c] = (feature_counts + self.smoothing) / \
                                  (total_feature_count + self.smoothing * n_features)

    def predict(self, X):
        results = []
        for sample in X:
            class_scores = {}
            for c in self.classes:
                # Folosim logaritm pentru a preveni underflow
                log_prob = np.log(self.priors[c])
                log_likelihood = np.sum(sample * np.log(self.likelihoods[c]))
                class_scores[c] = log_prob + log_likelihood
            results.append(max(class_scores, key=class_scores.get))
        return np.array(results)