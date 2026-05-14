import numpy as np

class NumpyAutoencoder:

    def __init__(self, input_dim, hidden_dim=64, code_dim=16, lr=0.005, epochs=80, batch_size=16):
        scale = 0.05
        self.lr = lr; self.epochs = epochs; self.batch_size = batch_size

        self.We1 = np.random.randn(input_dim, hidden_dim) * scale
        self.be1 = np.zeros(hidden_dim)
        self.We2 = np.random.randn(hidden_dim, code_dim) * scale
        self.be2 = np.zeros(code_dim)

        self.Wd1 = np.random.randn(code_dim, hidden_dim) * scale
        self.bd1 = np.zeros(hidden_dim)
        self.Wd2 = np.random.randn(hidden_dim, input_dim) * scale
        self.bd2 = np.zeros(input_dim)
        self.history = []

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def _rg(x):
        return (x > 0).astype(float)

    def encode(self, X):
        h1 = self._relu(X @ self.We1 + self.be1)
        return self._relu(h1 @ self.We2 + self.be2)

    def _decode(self, code):
        h1 = self._relu(code @ self.Wd1 + self.bd1)
        return h1 @ self.Wd2 + self.bd2 

    def fit(self, X):
        n = X.shape[0]
        for ep in range(self.epochs):
            idx = np.random.permutation(n); ep_loss = 0
            for s in range(0, n, self.batch_size):
                sl = idx[s:s+self.batch_size]; Xb = X[sl]

                z_e1 = Xb @ self.We1 + self.be1;  h_e1 = self._relu(z_e1)
                z_e2 = h_e1 @ self.We2 + self.be2; code = self._relu(z_e2)
                z_d1 = code @ self.Wd1 + self.bd1; h_d1 = self._relu(z_d1)
                recon = h_d1 @ self.Wd2 + self.bd2
                loss  = np.mean((recon - Xb)**2); ep_loss += loss

                dL   = 2*(recon - Xb) / len(Xb)
                dWd2 = h_d1.T @ dL; dbd2 = dL.mean(0)
                dh_d1= dL @ self.Wd2.T * self._rg(z_d1)
                dWd1 = code.T @ dh_d1; dbd1 = dh_d1.mean(0)
                dcode= dh_d1 @ self.Wd1.T * self._rg(z_e2)
                dWe2 = h_e1.T @ dcode; dbe2 = dcode.mean(0)
                dh_e1= dcode @ self.We2.T * self._rg(z_e1)
                dWe1 = Xb.T @ dh_e1;  dbe1 = dh_e1.mean(0)
                lr = self.lr
                for W,dW,b,db in [(self.We1,dWe1,self.be1,dbe1),
                                   (self.We2,dWe2,self.be2,dbe2),
                                   (self.Wd1,dWd1,self.bd1,dbd1),
                                   (self.Wd2,dWd2,self.bd2,dbd2)]:
                    W -= lr*dW; b -= lr*db
            self.history.append(ep_loss / max(1, n//self.batch_size))
        return self
