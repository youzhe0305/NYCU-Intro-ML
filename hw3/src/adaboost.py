import typing as t
import torch
from .utils import WeakClassifier, train_weak_classifier

devics = device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')


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
        self.dist = torch.full((X_train.shape[0],), 1 / X_train.shape[0]).to(device)  # distribution
        idx = 1
        for model in self.learners:
            # print(f"training model {idx}")
            idx += 1
            train_weak_classifier(model, X_train, y_train, num_epochs, learning_rate)
            y_pred = (model(X_train) > 0).float()
            loss = torch.sum(self.dist * torch.eq(y_pred, y_train))
            # print("loss: ", loss.item())
            losses_of_models.append(loss)
            EPS = 1e-10
            alpha = 1 / 2 * torch.log((1 - loss) / (loss + EPS) + EPS)
            if loss >= 1.0:
                alpha = torch.tensor(0).to(device)
            if loss <= 0 or torch.isnan(loss):  # to avoid alpha inf
                alpha = torch.tensor(300).to(device)
            self.alphas.append(alpha)
            self.dist *= torch.exp(- alpha * (y_train * 2 - 1) * (y_pred * 2 - 1) + EPS)  # 變成-1, 1
            self.dist /= torch.sum(self.dist + EPS)  # normalization

        return losses_of_models

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """Implement your code here"""
        sum = torch.zeros(X.shape[0]).to(device)
        alpha_sum = torch.sum(torch.stack(self.alphas))
        idx = 0
        pred_prob = []
        for model in self.learners:
            model = model.to(device)
            y_pred = model(X)
            pred_prob.append(y_pred)
            sum += self.alphas[idx] * y_pred
            idx += 1
        sum /= alpha_sum
        return (sum >= 0.5).int(), pred_prob

    def compute_feature_importance(self, df) -> t.Sequence[float]:
        """Implement your code here"""
        alpha_sum = torch.sum(torch.stack(self.alphas))
        idx = 0
        feature_importance = torch.zeros(self.learners[0].model[0].weight.shape[1]).to(device)
        for model in self.learners:
            feature_importance += torch.abs(model.model[0].weight.T @ model.model[1].weight.T
                                            * self.alphas[idx] / alpha_sum).view(-1)  # (27,mid_dim) @ (mid_dim,1)
            idx += 1
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
