import pandas as pd
from loguru import logger
import random

import torch
from src import AdaBoostClassifier, BaggingClassifier, DecisionTree
from src.utils import preprocess, preprocess_y, plot_learners_roc, plot_feature_importance
from src.decision_tree import gini


def main():
    """
    Note:
    1) Part of line should not be modified.
    2) You should implement the algorithm by yourself.
    3) You can change the I/O data type as you need.
    4) You can change the hyperparameters as you want.
    5) You can add/modify/remove args in the function, but you need to fit the requirements.
    6) When plot the feature importance, the tick labels of one of the axis should be feature names.
    """
    random.seed(777)  # DON'T CHANGE THIS LINE
    torch.manual_seed(777)  # DON'T CHANGE THIS LINE
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    X_train = train_df.drop(['target'], axis=1)
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    X_test = test_df.drop(['target'], axis=1)
    y_test = test_df['target'].to_numpy()

    # (TODO): Implement you preprocessing function.
    X_train = preprocess(X_train)
    y_train = preprocess_y(y_train)
    X_test = preprocess(X_test)
    y_test = preprocess_y(y_test)
    """
    (TODO): Implement your ensemble methods.
    1. You can modify the hyperparameters as you need.
    2. You must print out logs (e.g., accuracy) with loguru.
    """

    # AdaBoost
    clf_adaboost = AdaBoostClassifier(
        input_dim=X_train.shape[1],
    )
    _ = clf_adaboost.fit(
        X_train,
        y_train,
        num_epochs=2500,
        learning_rate=5e-2,
    )
    y_pred_classes, y_pred_probs = clf_adaboost.predict_learners(X_test)
    accuracy_ = torch.sum(torch.eq(y_pred_classes, y_test)) / y_test.shape[0]
    logger.info(f'AdaBoost - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath='./Adaboost_roc.png',
    )
    feature_importance, feature_name = clf_adaboost.compute_feature_importance(train_df)
    # (TODO) Draw the feature importance
    plot_feature_importance(feature_importance, feature_name, './Adaboost_feature_importance')

    # Bagging
    clf_bagging = BaggingClassifier(
        input_dim=X_train.shape[1],
    )
    _ = clf_bagging.fit(
        X_train,
        y_train,
        num_epochs=3000,
        learning_rate=1e-3,
    )
    y_pred_classes, y_pred_probs = clf_bagging.predict_learners(X_test)
    accuracy_ = torch.sum(torch.eq(y_pred_classes, y_test)) / y_test.shape[0]
    logger.info(f'Bagging - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath='Bagging_roc.png',
    )
    feature_importance, feature_name = clf_bagging.compute_feature_importance()
    # (TODO) Draw the feature importance
    plot_feature_importance(feature_importance, feature_name, './Bagging_feature_importance')

    # Gini
    sample = torch.tensor([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    sample_gini = gini(sample)
    logger.info(f'Sample gini: {sample_gini:.4f}')

    # Decision Tree
    clf_tree = DecisionTree(
        max_depth=7,
    )
    clf_tree.fit(X_train, y_train)
    y_pred_classes = clf_tree.predict(X_test)
    accuracy_ = torch.sum(torch.eq(y_pred_classes, y_test)) / y_test.shape[0]
    logger.info(f'DecisionTree - Accuracy: {accuracy_:.4f}')

    feature_importance, feature_name = clf_tree.get_feature_importance()
    # (TODO) Draw the feature importance
    plot_feature_importance(feature_importance, feature_name, './Decision_Tree_feature_importance')


if __name__ == '__main__':
    main()
