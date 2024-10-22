import typing as t

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

EPS = 1e-8


class LogisticRegression:
    def __init__(self, learning_rate: float = 1e-4, num_iterations: int = 100):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.intercept = None

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        """
        Implement your fitting function here.
        The weights and intercept should be kept in self.weights and self.intercept.
        """
        self.weights = np.random.randn(inputs.shape[1])
        self.intercept = np.random.randn(1)
        W = np.hstack((self.weights, self.intercept))  # (61,)
        X_with_bias = np.hstack((inputs, np.ones((inputs.shape[0], 1))))

        for i in range(self.num_iterations):
            y_pred, class_pred = self.predict(inputs)
            # loss = self.corss_entropy(targets, y_pred)  # scalar
            grad_W = self.back_propagation(X_with_bias, W, y_pred, targets)
            W -= self.learning_rate * grad_W
            self.weights = W[:-1]
            self.intercept = W[-1]

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Tuple[t.Sequence[np.float_], t.Sequence[int]]:
        """
        Implement your prediction function here.
        The return should contains
        1. sample probabilty of being class_1
        2. sample predicted class
        """
        W = np.hstack((self.weights, self.intercept))  # (61,)
        X_with_bias = np.hstack((inputs, np.ones((inputs.shape[0], 1))))  # (batch, 61)
        # activation: (batch,), numpy will automatically decide vector's direction(row or column)
        a = X_with_bias @ W  # (batch,61) @ (61,)
        y_pred = self.sigmoid(a)  # probability of class 1
        class_pred = (y_pred > 0.5)
        return y_pred, class_pred  # (batch,)

    def sigmoid(self, x):
        """
        Implement the sigmoid function.
        """
        return 1 / (1 + np.exp(-x))

    def corss_entropy(self, y, y_pred):
        loss = - (y @ np.log(y_pred + EPS) + (1 - y) @ np.log(1 - y_pred + EPS)) / y.shape[0]
        return loss

    def cross_entropy_derivative(self, y, y_pred):
        return - (y / (y_pred + EPS) - (1 - y) / (1 - y_pred + EPS))  # (batch,)

    def sigmoid_derivative(self, X, Y_grad):
        '''
        Y = sigmoid(X)
        d Loss / d X = (d Loss / d Y) * (d Y / d X)
        d Loss / d Y = Y_grad
        d Y / d X = sigmoid(X) * (1-sigmoid(X))
        '''
        return Y_grad * self.sigmoid(X) * (1 - self.sigmoid(X))

    def mul_deritivative(self, X, Y_grad):
        '''
        Y = X @ W
        d Loss / d W = (d Loss / d Y) * (d Y / d W)
        d Loss / d Y = Y_grad
        d Y / d X = X
        Y_grad: (166,)
        X: (166, 61)
        '''
        return Y_grad @  X  # (166,)

    def back_propagation(self, X, W, y_pred, y):
        grad_y_pred = self.cross_entropy_derivative(y, y_pred)
        grad_a = self.sigmoid_derivative(X @ W, grad_y_pred)
        grad_w = self.mul_deritivative(X, grad_a)
        grad_w = grad_w / y.shape[0]
        return grad_w


class FLD:
    """Implement FLD
    You can add arguments as you need,
    but don't modify those already exist variables.
    """
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None  # w0 for classifying

    def fit(
        self,
        inputs: npt.NDArray[float],
        targets: t.Sequence[int],
    ) -> None:
        X1 = inputs[targets == 1]  # datapoint of class 1
        X0 = inputs[targets == 0]

        self.m1 = np.mean(X1, axis=0)  # mean according batch
        self.m0 = np.mean(X0, axis=0)

        self.sb = (self.m1 - self.m0) @ (self.m1 - self.m0)
        self.sw = (X1 - self.m1).T @ (X1 - self.m1) + (X0 - self.m0).T @ (X0 - self.m0)

        self.w = np.linalg.inv(self.sw) @ (self.m1 - self.m0)
        self.slope = (self.m1 @ self.w + self.m0 @ self.w) / 2

    def predict(
        self,
        inputs: npt.NDArray[float],
    ) -> t.Sequence[t.Union[int, bool]]:

        X = inputs @ self.w  # (batch,)
        return X > self.slope

    def plot_projection(self, inputs: npt.NDArray[float], targets, name):
        def project(x, y, w, b):
            # w * x + -1 * y + b = 0
            x_proj = x - w * (w * x + (-1) * y + b) / (w ** 2 + (-1) ** 2)
            y_proj = w * x_proj + b
            return x_proj, y_proj

        w = self.w[1] / self.w[0]  # w[1] change from y, w[0] change from x => y/x = slope
        b = 0

        plt.title(f'*{name}* Project Line: w={round(w,5)}, b={round(b,5)}')

        x_line = np.linspace(-1.5, 1.5, 100)
        y_line = w * x_line + b
        plt.plot(x_line, y_line, color='black')

        X1 = inputs[targets == 1]
        X0 = inputs[targets == 0]

        plt.scatter(X1[:, 0], X1[:, 1], color='red', label='class 1', s=10)
        plt.scatter(X0[:, 0], X0[:, 1], color='blue', label='class 0', s=10)

        proj_x1, proj_y1 = project(X1[:, 0], X1[:, 1], w, b)

        proj_x0, proj_y0 = project(X0[:, 0], X0[:, 1], w, b)
        plt.scatter(proj_x1, proj_y1, color='red', label='class 1', s=10)
        plt.scatter(proj_x0, proj_y0, color='blue', label='class 0', s=10)

        for i in range(X1.shape[0]):
            plt.plot([X1[i, 0], proj_x1[i]], [X1[i, 1], proj_y1[i]], color='lightblue', alpha=0.5)
        for i in range(X0.shape[0]):
            plt.plot([X0[i, 0], proj_x0[i]], [X0[i, 1], proj_y0[i]], color='lightblue', alpha=0.5)

        plt.savefig(f'{name}.png')


def compute_auc(y_trues, y_preds):
    fpr, tpr, thresholds = roc_curve(y_trues, y_preds, pos_label=1)  # fpr: x-axis, tpr: y-axis
    return auc(fpr, tpr)


def accuracy_score(y_trues, y_preds):
    return np.sum(y_trues == y_preds) / y_trues.shape[0]
    raise NotImplementedError


def set_seed(seed):
    np.random.seed(seed)


def main():
    set_seed(529)
    # Read data
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    # Part1: Logistic Regression
    x_train = train_df.drop(['target'], axis=1).to_numpy()  # (n_samples, n_features), 166,60
    y_train = train_df['target'].to_numpy()  # (n_samples, )
    print(y_train.shape)

    x_test = test_df.drop(['target'], axis=1).to_numpy()
    y_test = test_df['target'].to_numpy()

    LR = LogisticRegression(
        learning_rate=10,  # You can modify the parameters as you want
        num_iterations=100000,  # You can modify the parameters as you want
    )
    LR.fit(x_train, y_train)
    y_pred_probs, y_pred_classes = LR.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred_classes)
    auc_score = compute_auc(y_test, y_pred_probs)
    logger.info(f'LR: Weights: {LR.weights[:5]}, Intercep: {LR.intercept}')
    logger.info(f'LR: Accuracy={accuracy:.4f}, AUC={auc_score:.4f}')

    # Part2: FLD
    cols = ['10', '20']  # Dont modify  # only take 2 feature?
    x_train = train_df[cols].to_numpy()  # (166,2)
    y_train = train_df['target'].to_numpy()
    x_test = test_df[cols].to_numpy()
    y_test = test_df['target'].to_numpy()
    FLD_ = FLD()
    """
    (TODO): Implement your code to
    1) Fit the FLD model
    2) Make prediction
    3) Compute the evaluation metrics

    Please also take care of the variables you used.
    """
    FLD_.fit(x_train, y_train)
    y_test_pred = FLD_.predict(x_test)
    accuracy = accuracy_score(y_test, y_test_pred)

    logger.info(f'FLD: m0={FLD_.m0}, m1={FLD_.m1} of {cols=}')
    logger.info(f'FLD: \nSw=\n{FLD_.sw}')
    logger.info(f'FLD: \nSb=\n{FLD_.sb}')
    logger.info(f'FLD: \nw=\n{FLD_.w}')
    logger.info(f'FLD: Accuracy={accuracy:.4f}')

    """
    (TODO): Implement your code below to plot the projection
    """
    FLD_.plot_projection(x_train, y_train, "FLD_projection_train")
    FLD_.plot_projection(x_test, y_test, "FLD_projection_test")


if __name__ == '__main__':
    main()
