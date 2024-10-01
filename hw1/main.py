import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt


class LinearRegressionBase:
    def __init__(self):
        self.weights = None  # 4 dim + const
        self.intercept = None

    def fit(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError


class LinearRegressionCloseform(LinearRegressionBase):
    def fit(self, X, y):
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y  # (4,n) @ n = 4
        self.intercept = self.weights[-1]
        self.weights = self.weights[:-1]
        return

    def predict(self, X):

        W = np.hstack((self.weights, self.intercept))
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        return X @ W


class LinearRegressionGradientdescent(LinearRegressionBase):
    def fit(self, X, y, learning_rate: float = 0.001, epochs: int = 1000):
        self.weights = np.ones(X.shape[1])
        self.intercept = np.ones(1)
        W = np.hstack((self.weights, self.intercept))
        X_with_bias = np.hstack((X, np.ones((X.shape[0], 1))))

        self.loss_record = []
        self.epoch_record = []
        for i in range(epochs):

            y_pred = self.predict(X)
            loss = compute_mse(y_pred, y)

            if (i) % 10000 == 0:
                logger.info(f'EPOCH {i}, loss={loss}')

            if i % 50 == 0:
                self.loss_record.append(loss)
                self.epoch_record.append(i)

            grad = self.mul_derivative(X_with_bias, self.MSE_derivative(y, y_pred))
            grad = grad / y.shape[0]
            grad[-1] *= 60  # intercept update slowly, so increase its gradient

            W -= learning_rate * grad
            self.intercept = W[-1]
            self.weights = W[:-1]

        return self.loss_record

    def predict(self, X):
        W = np.hstack((self.weights, self.intercept))
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        return X @ W

    def plot_learning_curve(self, losses):

        plt.plot(self.epoch_record, self.loss_record, label='training loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig('training_loss')

    def MSE_derivative(self, Y, Y_pred):

        Y = Y.reshape(-1)
        n = Y.shape[0]
        return - (2 / n) * (Y - Y_pred)

    def mul_derivative(self, X, grad_Y):
        '''
        Derivative of d Loss / d W
        formula Y = XW
        Y: batch
        X: batch * n
        W: n
        return n
        '''
        return grad_Y @ X


def compute_mse(prediction, ground_truth):
    ground_truth = ground_truth.reshape(-1)
    return np.mean((ground_truth - prediction) ** 2)


def main():
    train_df = pd.read_csv('./train.csv')
    train_x = train_df.drop(["Performance Index"], axis=1).to_numpy()  # (batch,n)
    train_y = train_df["Performance Index"].to_numpy()  # (batch)

    LR_CF = LinearRegressionCloseform()
    LR_CF.fit(train_x, train_y)
    logger.info(f'{LR_CF.weights=}, {LR_CF.intercept=:.4f}')

    LR_GD = LinearRegressionGradientdescent()
    losses = LR_GD.fit(train_x, train_y, learning_rate=2e-1, epochs=200000)
    LR_GD.plot_learning_curve(losses)
    logger.info(f'{LR_GD.weights=}, {LR_GD.intercept=:.4f}')

    test_df = pd.read_csv('./test.csv')
    test_x = test_df.drop(["Performance Index"], axis=1).to_numpy()
    test_y = test_df["Performance Index"].to_numpy()

    y_preds_cf = LR_CF.predict(test_x)
    y_preds_gd = LR_GD.predict(test_x)
    y_preds_diff = np.abs(y_preds_gd - y_preds_cf).mean()
    logger.info(f'Mean prediction difference: {y_preds_diff:.4f}')

    mse_cf = compute_mse(y_preds_cf, test_y)
    mse_gd = compute_mse(y_preds_gd, test_y)
    diff = (np.abs(mse_gd - mse_cf) / mse_cf) * 100
    logger.info(f'{mse_cf=:.4f}, {mse_gd=:.4f}. Difference: {diff:.3f}%')


if __name__ == '__main__':
    main()
