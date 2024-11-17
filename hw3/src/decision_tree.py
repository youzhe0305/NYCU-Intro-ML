"""
You dont have to follow the stucture of the sample code.
However, you should checkout if your class/function meet the requirements.
"""
import torch

device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')


class DecisionTree:
    def __init__(self, max_depth=1):
        self.max_depth = torch.tensor(max_depth).to(device)
        self.tree = None
        self.info_gain_threshold = torch.tensor(0.02).to(device)
        self.min_n_data = torch.tensor(50).to(device)
        self.nodes = 0
        self.feature_importance = None

    def fit(self, X, y):
        self.feature_importance = torch.zeros(X.shape[1]).to(device)
        self.tree = self._grow_tree(X, y)
        self.compute_feature_importance(self.tree)

    def cal_leaf_value(self, y):  # calcualte the leaf's corresponded category
        return torch.mode(y).values  # mode: 眾數

    def _grow_tree(self, X, y, depth=0):
        self.nodes += 1
        # print(f"grow node {self.nodes}, depth={depth}, samples={X.shape[0]}")
        if depth >= self.max_depth:
            leaf_value = self.cal_leaf_value(y)
            return {"leaf": leaf_value}

        if torch.unique(y).shape[0] <= 1:  # if all are same category (pure)
            leaf_value = self.cal_leaf_value(y)
            return {"leaf": leaf_value}

        if y.shape[0] <= self.min_n_data:
            leaf_value = self.cal_leaf_value(y)
            return {"leaf": leaf_value}

        info_gain, feature_idx, threshold = find_best_split(X, y)
        # print(f"info gain: {info_gain}")
        if info_gain < self.info_gain_threshold:  # info gain too small
            leaf_value = self.cal_leaf_value(y)
            return {"leaf": leaf_value}

        split1_indices = X[:, feature_idx] < threshold
        split2_indices = X[:, feature_idx] >= threshold

        # Create subtrees
        left_subtree = self._grow_tree(X[split1_indices], y[split1_indices], depth + 1)
        right_subtree = self._grow_tree(X[split2_indices], y[split2_indices], depth + 1)

        return {"feature_idx": feature_idx, "threshold": threshold,
                "left": left_subtree, "right": right_subtree}

    def predict(self, X):
        # print("predict...")
        return torch.tensor([self._predict_tree(x, self.tree) for x in X]).to(device)

    def _predict_tree(self, x, tree_node):
        if "leaf" in tree_node:
            return tree_node["leaf"]  # return category
        if x[tree_node["feature_idx"]] < tree_node["threshold"]:
            return self._predict_tree(x, tree_node["left"])
        else:
            return self._predict_tree(x, tree_node["right"])

    def compute_feature_importance(self, tree_node):

        if "leaf" in tree_node:
            return
        self.feature_importance[tree_node['feature_idx']] += 1
        self.compute_feature_importance(tree_node['left'])
        self.compute_feature_importance(tree_node['right'])

    def get_feature_importance(self):
        feature_importance = self.feature_importance
        feature_name = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt', 'loan_int_rate',
                        'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score', 'person_gender',
                        'person_education', 'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file']
        # integrate the one hot encoded result to original result
        person_gender = torch.mean(feature_importance[8:10]).view(-1)
        person_education = torch.mean(feature_importance[10:15]).view(-1)
        person_home_ownership = torch.mean(feature_importance[15:19]).view(-1)
        loan_intent = torch.mean(feature_importance[19:25]).view(-1)
        previous_loan_defaults_on_file = torch.mean(feature_importance[25:]).view(-1)
        feature_importance = torch.cat((feature_importance[:8], person_gender, person_education,
                                        person_home_ownership, loan_intent, previous_loan_defaults_on_file))
        return feature_importance, feature_name


# Split dataset based on a feature and threshold
def split_dataset(X, y, feature_index, threshold):
    split1_indices = X[:, feature_index] < threshold
    split2_indices = X[:, feature_index] >= threshold
    return X[split1_indices], X[split2_indices], y[split1_indices], y[split2_indices]


# Find the best split for the dataset
def find_best_split(X, y):

    best_info_gain = -1e10
    best_feature_idx = -1
    best_threshold = -1
    for feature_idx in range(X.shape[1]):
        thresholds = X[:, feature_idx]  # find one of features's value
        sample_indices = torch.randint(0, thresholds.shape[0], (min(100, thresholds.shape[0]),)).to(device)
        for threshold in thresholds[sample_indices]:  # choose one data's feature as split point
            X_split1, X_split2, y_split1, y_split2 = split_dataset(X, y, feature_idx, threshold)
            info_gain = information_gain(y, y_split1, y_split2)
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature_idx = feature_idx
                best_threshold = threshold
    return best_info_gain, best_feature_idx, best_threshold


def entropy(y):
    n_c1 = torch.sum(y == 1).to(device)
    n_c2 = torch.sum(y == 0).to(device)
    p_c1 = n_c1 / (n_c1 + n_c2)
    p_c2 = n_c2 / (n_c1 + n_c2)
    EPS = 1e-10
    return - (p_c1 * torch.log(p_c1 + EPS) + p_c2 * torch.log(p_c2 + EPS))


def information_gain(y_origin, y_split1, y_split2):  # split to 2 node
    origin_entropy = entropy(y_origin)
    splited_entropy = (
        (y_split1.shape[0] / y_origin.shape[0]) * entropy(y_split1)
        + (y_split2.shape[0] / y_origin.shape[0]) * entropy(y_split2)
    )
    return origin_entropy - splited_entropy


def gini(y):
    n_c1 = torch.sum(y == 1).to(device)
    n_c2 = torch.sum(y == 0).to(device)
    p_c1 = n_c1 / (n_c1 + n_c2)
    p_c2 = n_c2 / (n_c1 + n_c2)
    return 1 - p_c1 ** 2 - p_c2 ** 2
