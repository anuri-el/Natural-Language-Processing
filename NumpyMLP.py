import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


class NumpyMLP:
    def __init__(self, layer_sizes, lr=0.01, momentum=0.9, l2=1e-4, epochs=100, batch_size=32, random_state=42):
        np.random.seed(random_state)
        self.sizes = layer_sizes
        self.lr = lr
        self.momentum = momentum
        self.l2 = l2
        self.epochs = epochs
        self.batch_size = batch_size
        self.history = []
        self.W = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i]) for i in range(len(layer_sizes)-1)]
        self.b = [np.zeros(layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def _relu_grad(x):
        return (x > 0).astype(float)

    @staticmethod
    def _softmax(x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)
    
    def _forward(self, X):
        acts, zs = [X], []
        for i,(W,b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b; zs.append(z)
            acts.append(self._relu(z) if i < len(self.W)-1 else self._softmax(z))
        return acts, zs

    def _loss(self, y_hat, y_oh):
        return -np.mean(np.sum(y_oh * np.log(y_hat + 1e-9), axis=1))

    def fit(self, X, y, X_val=None, y_val=None):
        n_classes = len(np.unique(y))
        enc = LabelEncoder().fit(y)
        yi  = enc.transform(y)
        self.enc = enc
        Y   = np.eye(n_classes)[yi]
        n   = X.shape[0]
        for ep in range(self.epochs):
            idx = np.random.permutation(n)
            ep_loss = 0
            for start in range(0, n, self.batch_size):
                sl = idx[start:start+self.batch_size]
                Xb, Yb = X[sl], Y[sl]
                acts, zs = self._forward(Xb)
                ep_loss += self._loss(acts[-1], Yb)

                delta = acts[-1] - Yb
                grads_W, grads_b = [], []
                for i in range(len(self.W)-1, -1, -1):
                    gW = acts[i].T @ delta / len(Xb) + self.l2 * self.W[i]
                    gb = delta.mean(axis=0)
                    grads_W.insert(0, gW); grads_b.insert(0, gb)
                    if i > 0:
                        delta = (delta @ self.W[i].T) * self._relu_grad(zs[i-1])

                for i in range(len(self.W)):
                    self.vW[i] = self.momentum*self.vW[i] - self.lr*grads_W[i]
                    self.vb[i] = self.momentum*self.vb[i] - self.lr*grads_b[i]
                    self.W[i] += self.vW[i]; self.b[i] += self.vb[i]

            ep_loss /= max(1, n // self.batch_size)
            rec = {"epoch": ep+1, "loss": round(ep_loss, 5)}
            if X_val is not None:
                pv = self.predict(X_val)
                rec["val_acc"] = round(accuracy_score(y_val, pv), 4)
            self.history.append(rec)
        return self

    def predict_proba(self, X):
        acts, _ = self._forward(X)
        return acts[-1]

    def predict(self, X):
        return self.enc.inverse_transform(self.predict_proba(X).argmax(axis=1))