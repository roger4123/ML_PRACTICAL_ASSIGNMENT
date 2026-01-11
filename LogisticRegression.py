import numpy as np

class MyLogisticRegression:
    """
    Implementare de la zero a Regresiei Logistice.
    Include: Sigmoid, Cost Function, Gradient Descent.
    """
    def __init__(self, learning_rate=0.01, num_iterations=1000, verbose=False):
        self.lr = learning_rate
        self.num_iters = num_iterations
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def sigmoid(self, z):
        # Clip for numerical stability (evită overflow la exp)
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def cost_function(self, h, y):
        m = len(y)
        epsilon = 1e-15
        cost = (-1/m) * np.sum(y * np.log(h + epsilon) + (1-y) * np.log(1 - h + epsilon))
        return cost
    
    def fit(self, X, y):
        """ Model training using Gradient Descent """
        m, n = X.shape
        
        # 1. Init. params
        self.weights = np.zeros(n)
        self.bias = 0
        self.cost_history = []
        
        for i in range(self.num_iters):
            # 2. Forward Pass (linear prediction + activation)
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            # 3. Gradient computation
            dw = (1/m) * np.dot(X.T, (y_predicted - y))
            db = (1/m) * np.sum(y_predicted - y)
            
            # 4. Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # 5. Saving cost
            cost = self.cost_function(y_predicted, y)
            self.cost_history.append(cost)
            
            if self.verbose and i % 100 == 0:
                print(f"Epoch {i}: Cost {cost:.4f}")
                
    def predict_prob(self, X):
        """ Returns probability (between 0 and 1) """
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    # def predict_score_raw(self, X):
    #     """ Returns raw score (z) before sigmoid - useful for Ranking """
    #     return np.dot(X, self.weights) + self.bias
    
    def predict(self, X, threshold=0.5):
        """ Returns class (0 or 1) """
        y_predicted_cls = [1 if i > threshold else 0 for i in self.predict_prob(X)]
        return np.array(y_predicted_cls)

def calculate_metrics(y_true, y_pred, set_name="Set"):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp+fp) > 0 else 0
    recall = tp / (tp + fn) if (tp+fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision+recall) > 0 else 0
    
    print(f"\n--- Metrics {set_name} ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")