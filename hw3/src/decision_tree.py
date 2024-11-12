"""
You dont have to follow the stucture of the sample code.
However, you should checkout if your class/function meet the requirements.
"""
import numpy as np


class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def _predict_tree(self, x, tree_node):
        raise NotImplementedError


# Split dataset based on a feature and threshold
def split_dataset(X, y, feature_index, threshold):
    raise NotImplementedError


# Find the best split for the dataset
def find_best_split(X, y):
    raise NotImplementedError


def entropy(y):
    raise NotImplementedError
