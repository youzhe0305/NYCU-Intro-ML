import typing as t
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import roc_curve, auc

def preprocess(df: pd.DataFrame):
    """
    (TODO): Implement your preprocessing function.
    """
    df = pd.get_dummies(df, columns=['person_gender', 'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file'])
    data = df.to_numpy().astype(np.float32)
    data = torch.from_numpy(data)
    return data

def preprocess_y(df: pd.DataFrame):
    data = df.astype(np.float32)
    data = torch.from_numpy(data)
    return data

class WeakClassifier(nn.Module):
    """
    Use pyTorch to implement a 1 ~ 2 layers model.
    Here, for example:
        - Linear(input_dim, 1) is a single-layer model.
        - Linear(input_dim, k) -> Linear(k, 1) is a two-layer model.

    No non-linear activation allowed.
    """
    def __init__(self, input_dim):
        super(WeakClassifier, self).__init__()
        mid_dim = 4
        self.model = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            nn.Linear(mid_dim, 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        return self.model(x)

def train_weak_classifier(model, X_train, y_train, n_iteration=100, learning_rate=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)   
    for i in range(n_iteration):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = entropy_loss(outputs, y_train)
        loss.backward()
        optimizer.step()
        if i%50==0:
            print(f"iter: {i}, loss: {loss.item()}")    

def accuracy_score(y_trues, y_preds) -> float:
    return torch.sum(torch.eq(y_preds, y_trues)) / y_trues.shape[0]


def entropy_loss(outputs, targets):
    EPS = 1e-8
    loss = - (targets @ torch.log(outputs + EPS) + (1 - targets) @ torch.log(1 - outputs + EPS)) / targets.shape[0]
    return loss


def plot_learners_roc(
    y_preds: t.List[t.Sequence[float]],
    y_trues: t.Sequence[int],
    fpath='./tmp.png',
):
    raise NotImplementedError
