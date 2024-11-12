import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier, train_weak_classifier


class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10) -> None:
        self.sample_weights = None
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(num_learners)
        ]
        self.alphas = []

    def fit(self, X_train, y_train, num_epochs: int = 500, learning_rate: float = 0.001):
        """Implement your code here"""
        losses_of_models = []
        self.dist = torch.full( (X_train.shape[0],), 1 / X_train.shape[0]) # distribution
        for model in self.learners:
            train_weak_classifier(model, X_train, y_train, num_epochs, learning_rate)
            y_pred = (model(X_train) > 0).float()
            loss = torch.sum(self.dist * torch.eq(y_pred, y_train).float())
            losses_of_models.append(loss)
            alpha = 1/2 * torch.log((1 - loss) / loss)
            self.alphas.append(alpha)
            y_pred = y_pred * 2 - 1 # 變成-1, +1
            y_train = y_train * 2 - 1
            print(y_pred.shape)
            print(y_train.shape)
            self.dist *= torch.exp(- alpha * y_train * y_pred)
            self.dist /= torch.sum(self.dist) # normalization
        
        return losses_of_models

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """Implement your code here"""
        sum = torch.zeros(X.shape[0])
        idx = 0
        for model in self.learners:
            y_pred = model(X)
            sum += self.alphas[idx] * y_pred
        return (sum >= 0.5).int(), y_pred 

    def compute_feature_importance(self) -> t.Sequence[float]:
        """Implement your code here"""
        return self.dist
