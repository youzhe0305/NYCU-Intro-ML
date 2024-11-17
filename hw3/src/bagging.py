import typing as t
import torch
from .utils import WeakClassifier, train_weak_classifier


device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')


class BaggingClassifier:
    def __init__(self, input_dim: int) -> None:
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(10)
        ]

    def fit(self, X_train, y_train, num_epochs: int, learning_rate: float):
        """Implement your code here"""

        losses_of_models = []
        idx = 1
        for model in self.learners:
            # print(f"training model {idx}")
            idx += 1

            indices = torch.randint(0, X_train.shape[0], (X_train.shape[0],)).to(device)
            X_train_rand = X_train[indices]
            y_train_rand = y_train[indices]
            train_weak_classifier(model, X_train_rand, y_train_rand, num_epochs, learning_rate)
            y_pred = (model(X_train) > 0).float()
            loss = torch.sum(torch.eq(y_pred, y_train_rand)) / y_train_rand.shape[0]
            # print("loss: ", loss.item())
            losses_of_models.append(loss)
        return losses_of_models

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """Implement your code here"""
        sum = torch.zeros(X.shape[0]).to(device)
        idx = 0
        pred_prob = []
        for model in self.learners:
            model = model.to(device)
            y_pred = model(X)
            pred_prob.append(y_pred)
            sum += y_pred
            idx += 1
        sum /= len(self.learners)
        return (sum >= 0.5).int(), pred_prob

    def compute_feature_importance(self) -> t.Sequence[float]:
        """Implement your code here"""
        feature_importance = torch.zeros(self.learners[0].model[0].weight.shape[1]).to(device)
        for model in self.learners:
            feature_importance += torch.abs(model.model[0].weight.T @ model.model[1].weight.T).view(-1)
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
