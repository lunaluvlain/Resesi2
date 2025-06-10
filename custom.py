import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []

        for i in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 100 == 0:
                loss = - (1/n_samples) * np.sum(
                    y * np.log(y_predicted + 1e-15) + (1-y) * np.log(1 - y_predicted + 1e-15)
                )
                self.losses.append(loss)

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return [1 if i > 0.78 else 0 for i in y_predicted]

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        prob_positive = self.sigmoid(linear_model)
        return np.vstack([1 - prob_positive, prob_positive]).T

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


import numpy as np

class ManualSVM:
    def __init__(self, C=1.0, max_iter=1000, lr=0.001):
        self.C = C
        self.max_iter = max_iter
        self.lr = lr
        self.alphas = None
        self.w = None
        self.b = None
        self.support_vectors = None
        self.support_labels = None
        self.X_train = None
        self.y_train = None

    def hyperplane(self, x):
        return np.dot(self.w, x) + self.b

    def compute_w(self, alphas, y, X):
        w = np.zeros(X.shape[1])
        for i in range(len(X)):
            w += alphas[i] * y[i] * X[i]
        return w

    def compute_b(self, w, support_vectors, support_labels):
        b_sum = 0
        for i in range(len(support_vectors)):
            b_sum += support_labels[i] - np.dot(w, support_vectors[i])
        return b_sum / len(support_vectors)

    def dual_form(self, alphas, y, X):
        first_sum = np.sum(alphas)
        second_sum = 0
        for i in range(len(X)):
            for j in range(len(X)):
                second_sum += alphas[i] * alphas[j] * y[i] * y[j] * np.dot(X[i], X[j])
        return first_sum - 0.5 * second_sum

    def train(self, X, y):
        n_samples = X.shape[0]
        self.X_train = X
        self.y_train = y
        alphas = np.zeros(n_samples)
        for iter in range(1, self.max_iter + 1):
            for i in range(n_samples):
                gradient = 1 - y[i] * np.sum(alphas * y * np.dot(X, X[i]))
                alphas[i] += self.lr * gradient
                # Projection ke [0, C]
                alphas[i] = max(0, min(self.C, alphas[i]))

            if iter % 100 == 0 or iter == 1:
                loss = self.dual_form(alphas, y, X)
                print(f"Iter {iter}, Dual Objective (Loss): {loss:.4f}")

        self.alphas = alphas
        self.w = self.compute_w(alphas, y, X)
        support_indices = np.where(alphas > 1e-5)[0]
        self.support_vectors = X[support_indices]
        self.support_labels = y[support_indices]
        self.b = self.compute_b(self.w, self.support_vectors, self.support_labels)

    def decision_function(self, x):
        result = 0
        for i in range(len(self.X_train)):
            result += self.alphas[i] * self.y_train[i] * np.dot(self.X_train[i], x)
        return result + self.b
    


    def predict(self, X):
        predictions = np.array([np.sign(self.decision_function(x)) for x in X])
        return np.where(predictions == -1, 0, 1)
