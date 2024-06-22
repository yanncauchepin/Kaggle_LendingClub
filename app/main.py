import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from captum.attr import IntegratedGradients
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from mapie.classification import MapieClassifier
from sklearn.calibration import CalibratedClassifierCV
from skopt import BayesSearchCV

torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_sizes, activation_name, p_dropout):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        if isinstance(self.hidden_layer_sizes, str):
            self.hidden_layer_sizes = eval(self.hidden_layer_sizes)
        self.activation_name = activation_name
        self.p_dropout = p_dropout
        if activation_name == "Relu":
            self.activation = nn.ReLU()
        elif activation_name == "Sigmoid":
            self.activation = nn.Sigmoid()
        elif activation_name == "Softmax":
            self.activation = nn.Softmax(dim=1)
        elif activation_name == "Tanh":
            self.activation = nn.Tanh()
        elif activation_name == "Leaky_relu":
            self.activation = nn.LeakyReLU()
        else:
            raise ValueError(f'Unsupported activation: {self.activation_name}')
        layers = []
        layers.append(nn.Linear(self.input_size, self.hidden_layer_sizes[0]))
        layers.append(self.activation)
        for i in range(len(self.hidden_layer_sizes) - 1):
            layers.append(nn.Linear(self.hidden_layer_sizes[i], self.hidden_layer_sizes[i + 1]))
            layers.append(self.activation)
            layers.append(nn.Dropout(p=self.p_dropout))
        layers.append(nn.Linear(self.hidden_layer_sizes[-1], self.output_size))
        self.model = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()


    def forward_proba(self, X):
        output = self.model(X)
        output = self.sigmoid(output)
        return output.float()


    def forward(self, X):
        output = self.model(X)
        output = self.sigmoid(output)
        return output


class MLPEstimatorSklearn():
    def __init__(self, **params):
        self.input_size = params.get("input_size")
        self.output_size = params.get("output_size")
        self.hidden_layer_sizes = params.get("hidden_layer_sizes", (60, 60))
        self.activation_name = params.get("activation_name", "Relu")
        self.loss = params.get("loss", "binary_cross_entropy")
        self.optimizer_name = params.get("optimizer_name", "Adam")
        self.learning_rate = params.get("learning_rate", 1e-3)
        self.batch_size = params.get("batch_size", 50)
        self.weight_decay = params.get("weight_decay", 0)
        self.p_dropout = params.get("p_dropout", 0.2)
        self.early_stopping = params.get("early_stopping", True)
        self.epochs = params.get("epochs", 200)
        self.patience = params.get("patience", 10)
        self.verbose = params.get("verbose", True)
        self.classes_ = classes_

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = MLP(self.input_size, self.output_size, self.hidden_layer_sizes, self.activation_name, self.p_dropout).to(self.device)

        if self.loss == "binary_cross_entropy":
            self.criterion = nn.BCELoss()
        else:
            raise ValueError(f"Unsupported loss: {self.loss}")

        if self.optimizer_name == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optimizer_name == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")

    def next_batch(self, inputs, targets, batchSize):
        inputs_tensor = torch.from_numpy(inputs).float()
        targets_tensor = torch.from_numpy(targets).float().unsqueeze(1)
        for i in range(0, inputs_tensor.shape[0], batchSize):
            yield (inputs_tensor[i:i + batchSize], targets_tensor[i:i + batchSize])

    def augment_data(self, X_unlabeled, noise_level=0.1):
        noise = noise_level * torch.randn_like(X_unlabeled)
        return X_unlabeled + noise

    def fit(self, X, y, X_unlabeled=None):
        self.classes_ = np.unique(y)
        running_losses_1 = list()
        if self.early_stopping:
            best_loss = float('inf')
            count = 0
        if self.verbose:
            epoch_iterator_1 = tqdm(range(self.epochs), desc="Supervised training ; epochs", unit="epoch")
        else:
            epoch_iterator_1 = range(self.epochs)
        for epoch in epoch_iterator_1:
            samples = 0
            train_loss = 0.0
            self.model.train(True)
            for i, (batchX, batchY) in enumerate(self.next_batch(X, y, self.batch_size)):
                batchX = batchX.to(self.device)
                batchY = batchY.to(self.device)
                batchY.requires_grad = True
                self.optimizer.zero_grad()
                outputs = self.model(batchX)
                loss = self.criterion(outputs, batchY)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                samples += batchY.size(0)
            running_loss = train_loss / samples
            running_losses_1.append(running_loss)
            if self.verbose:
                epoch_iterator_1.set_postfix(train_loss=running_loss)
            if self.early_stopping:
                if running_loss < best_loss:
                    best_loss = running_loss
                    count = 0
                else:
                    count += 1
                if count >= self.patience:
                    break
        if self.verbose:
            epoch_iterator_1.close()
        self.running_losses_1 = [loss for loss in running_losses_1 if loss <= best_loss]
        if X_unlabeled is not None:
            best_loss = float('inf')
            running_losses_2 = list()
            X_unlabeled = torch.from_numpy(X_unlabeled).float()
            augmented_X_unlabeled = self.augment_data(X_unlabeled)
            self.model.eval()
            with torch.no_grad():
                pseudo_labels = self.model(X_unlabeled)
                pseudo_labels = (pseudo_labels > 0.5).float().squeeze()

            X_combined = torch.cat((torch.from_numpy(X).float(), augmented_X_unlabeled.cpu()), 0).to(self.device)
            y_combined = torch.cat((torch.from_numpy(y).float().squeeze(), pseudo_labels), 0).to(self.device)

            if self.verbose:
                epoch_iterator_2 = tqdm(range(self.epochs), desc="Semi-supervised training ; epochs", unit="epoch")
            else:
                epoch_iterator_2 = range(self.epochs)
            for epoch in epoch_iterator_2:
                samples = 0
                train_loss = 0.0
                self.model.train(True)
                dataset = TensorDataset(X_combined, y_combined)
                for i, (batchX, batchY) in enumerate(self.next_batch(X, y, self.batch_size)):
                    batchX = batchX.to(self.device)
                    batchY = batchY.to(self.device)
                    batchY.requires_grad = True
                    self.optimizer.zero_grad()
                    outputs = self.model(batchX)
                    loss = self.criterion(outputs, batchY)
                    loss.backward()
                    self.optimizer.step()
                    train_loss += loss.item()
                    samples += batchY.size(0)
                running_loss = train_loss / samples
                running_losses_2.append(running_loss)
                if self.verbose:
                    epoch_iterator_2.set_postfix(train_loss=running_loss)
                if self.early_stopping:
                    if running_loss < best_loss:
                        best_loss = running_loss
                        count = 0
                    else:
                        count += 1
                    if count >= self.patience:
                        break
            if self.verbose:
                epoch_iterator_2.close()
            self.running_losses_2 = [loss for loss in running_losses_2 if loss <= best_loss]
        return self

    def predict(self, X):
        self.model.eval()
        X = torch.from_numpy(X).float().to(self.device)
        y_pred = self.model.forward(X)
        if self.device == "cpu":
            y_pred = y_pred.cpu().detach().numpy()
        else:
            y_pred = y_pred.detach().numpy()
        return (y_pred > 0.5).astype(int)

    def predict_proba(self, X):
        self.model.eval()
        X = torch.from_numpy(X).float().to(self.device)
        y_proba = self.model.forward_proba(X)
        if self.device == "cpu":
            y_proba = y_proba.cpu().detach().numpy().astype(float)
        else:
            y_proba = y_proba.detach().numpy().astype(float)
        y_proba = y_proba.squeeze().astype(np.float32)
        return np.column_stack((1 - y_proba, y_proba))

    def get_params(self, deep=True):
        params = {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "activation_name": self.activation_name,
            "loss": self.loss,
            "optimizer_name": self.optimizer_name,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "weight_decay": self.weight_decay,
            "p_dropout": self.p_dropout,
            "early_stopping": self.early_stopping,
            "epochs": self.epochs,
            "patience": self.patience,
            "verbose": self.verbose
        }
        return params

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def get_mlp(self):
        return self.model

class MLPBinaryClassifier():
    def __init__(self, X, y, split_test, X_unlabeled=None, **params):
        self.model = MLPEstimatorSklearn(**params)
        self.X = X
        self.X_unlabeled = X_unlabeled
        self.y = y

        self.y = MLPBinaryClassifier.float_to_class(self.y).ravel()

        self.split_test = split_test
        self.split_data()

        self.standardize(self.X_train_cal)
        self.X_train_standard = self.standardize_X(self.X_train)
        self.X_cal_standard = self.standardize_X(self.X_cal)
        if isinstance(self.X_unlabeled, np.ndarray):
            self.X_unlabeled_standard = self.standardize_X(self.X_unlabeled)
        else :
            self.X_unlabeled_standard = None
        self.y_train_standard = self.y_train
        self.y_cal_standard = self.y_cal.reshape(-1,1)


    @staticmethod
    def float_to_class(y):
        threshold = 0.5
        return (y >= threshold).astype(int)

    def split_data(self):
        self.X_train_cal, self.X_test, self.y_train_cal, self.y_test = train_test_split(
            self.X, self.y, test_size=self.split_test, shuffle=True, random_state=1, stratify=self.y)
        self.X_train, self.X_cal, self.y_train, self.y_cal = train_test_split(
            self.X_train_cal, self.y_train_cal, test_size=0.25, shuffle=True, random_state=1, stratify=self.y_train_cal)

    def standardize(self, X):
        self.scaler_X_train = StandardScaler()
        self.scaler_X_train.fit(X)


    def standardize_X(self, X):
        X_new = self.scaler_X_train.transform(X)
        return X_new



    def bayes_search(self, param_bayes, n_iter, n_points=1, cv=5, scoring='accuracy',
                 verbose=3, n_jobs=1) :
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=1)
        bayes_search = BayesSearchCV(self.model, param_bayes, n_iter=n_iter,
                                     n_points=n_points, cv=cv, scoring=scoring,
                                     verbose=verbose, return_train_score=True,
                                     n_jobs=n_jobs, random_state=1)
        bayes_search.fit(self.X_train_standard, self.y_train_standard)
        results_df = pd.DataFrame(bayes_search.cv_results_)
        self.model = bayes_search.best_estimator_
        print(f'Best hyperparameters bayes search : {bayes_search.best_params_}')
        return results_df

    def randomized_search(self, param_randomized, n_iter, cv=5, scoring='accuracy',
                      verbose=3, n_jobs=1) :
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=1)
        randomized_search = RandomizedSearchCV(self.model, param_randomized,
                                               n_iter=n_iter, cv=cv, scoring=scoring,
                                               verbose=verbose, return_train_score=True,
                                               n_jobs=n_jobs, random_state=1)
        randomized_search.fit(self.X_train_standard, self.y_train_standard)
        results_df = pd.DataFrame(randomized_search.cv_results_)
        self.model = randomized_search.best_estimator_
        print(f'Best hyperparameters randomized search : {randomized_search.best_params_}')
        return results_df

    def fit(self, method="lac"):
        self.model.fit(self.X_train_standard, self.y_train_standard, self.X_unlabeled_standard)
        self.model_mapie = MapieClassifier(estimator=self.model, cv="prefit", method=method)
        self.model_mapie.fit(self.X_cal_standard, self.y_cal_standard)

    def predict(self, X, alpha=0.05):
        X_standard = self.standardize_X(X)
        y_pred, y_ps = self.model_mapie.predict(X_standard, alpha=alpha)
        return y_pred, y_ps

    @staticmethod
    def compute_metrics(metric, y_true, y_pred):
        y_pred = MLPBinaryClassifier.float_to_class(y_pred)
        accuracy = metrics.accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = metrics.recall_score(y_true, y_pred, average='weighted')
        f1 = metrics.f1_score(y_true, y_pred, average='weighted')
        metrics_dict = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
        }
        if metric != 'all':
            metrics_dict = {metric: metrics_dict[metric]}
        return metrics_dict


    def model_performance(self, metric='all'):
        y_pred_train, _ = self.predict(self.X_train)
        scores_train = MLPBinaryClassifier.compute_metrics(metric, self.y_train, y_pred_train)
        y_pred_test, _ = self.predict(self.X_test)
        scores_test = MLPBinaryClassifier.compute_metrics(metric, self.y_test, y_pred_test)
        data = {}
        for key, value in scores_train.items():
            data['Train Set - '+key] = [value]
        for key, value in scores_test.items():
            data['Test Set - '+key] = [value]
        df_scores = pd.DataFrame(data=data).T
        df_scores.columns = ['Scores']
        return df_scores

    def model_performance_test(self, X_test, y_test, metric='all'):
        y_pred_test, _ = self.predict(X_test)
        scores_test = MLPBinaryClassifier.compute_metrics(metric, y_test, y_pred_test)
        data = {}
        for key, value in scores_test.items():
            data['Test Set - '+key] = [value]
        df_scores = pd.DataFrame(data=data).T
        df_scores.columns = ['Scores']
        return df_scores

    def receiver_operating_characteristics(self):
        y_pred_test, _ = self.predict(self.X_test)
        fpr, tpr, thresholds = metrics.roc_curve(self.y_test, y_pred_test)
        plt.plot(fpr, tpr)
        plt.title("Receiver Operating Characteristics")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()

    def compute_integrated_gradients(self, X, baseline=None, steps=50):
        def preprocess_input(X):
            return torch.tensor(X, dtype=torch.float32)
        input_tensor = preprocess_input(X)
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
        integrated_gradients = IntegratedGradients(self.model.get_mlp())
        attributions = integrated_gradients.attribute(input_tensor, baseline, target=0, n_steps=steps)
        attributions_df = pd.DataFrame(attributions.cpu().detach().numpy(), columns=features)
        avg_attributions = attributions_df.mean(axis=0)
        avg_abs_attributions = avg_attributions.abs()
        def custom_minmax_scaler(data, feature_range=(0, 100)):
            min_val = np.min(data)
            max_val = np.max(data)
            if max_val - min_val == 0:
                return np.zeros_like(data) if feature_range[0] == 0 else np.full_like(data, feature_range[0])
            scale = (feature_range[1] - feature_range[0]) / (max_val - min_val)
            min_range = feature_range[0]
            scaled_data = scale * (data - min_val) + min_range
            return scaled_data
        normalized_data = custom_minmax_scaler(avg_abs_attributions.values.reshape(-1, 1)).astype(float)
        np.set_printoptions(suppress=True, precision=2)
        normalized_attributions = pd.DataFrame(normalized_data, columns=['attribution'], index=features)
        sorted_attributions = normalized_attributions.sort_values(by="attribution", ascending=False)
        return sorted_attributions


import pickle
with open('lending_club_mlp_binary_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

map_output = {
    0: 'Fully Paid',
    1: 'Charged Off'
    }

features = ['loan_amnt',
 'funded_amnt',
 'funded_amnt_inv',
 'int_rate',
 'fico_range_high',
 'total_pymnt',
 'total_pymnt_inv',
 'total_rec_prncp',
 'total_rec_int',
 'recoveries',
 'last_pymnt_amnt',
 'last_fico_range_high',
 'last_fico_range_low',
 'mths_since_rcnt_il',
 'mo_sin_old_rev_tl_op',
 'last_credit_pull_d']

def date_to_float(date):
    map_date = {
        'Jan': 0/12,
        'Feb': 1/12,
        'Mar': 2/12,
        'Apr': 3/12,
        'May': 4/12,
        'Jun': 5/12,
        'Jul': 6/12,
        'Aug': 7/12,
        'Sep': 8/12,
        'Oct': 9/12,
        'Nov': 10/12,
        'Dec': 11/12,
    }
    month, year = date.split('-')
    return float(year)+float(map_date[month])


def get_user_inputs():
    input_ = list()
    for feature in features :
        if feature in ["last_credit_pull_d", "last_pymnt_d"]:
            value = input(f"{feature} - Format %Mon-YEAR% : ")
            input_.append(float(date_to_float(value)))
        elif feature in ["int_rate"]:
            value = input(f"{feature} - Format float >= 1, without % : ")
            input_.append(float(value))
        else :
            input_.append(float(input(f"{feature} :")))
    return input_

import json
import os
def get_user_inputs_from_json(file_path=None):
    if file_path is None:
        file_path = '/app/data_test.json'
    with open(file_path, 'r') as file:
        data = json.load(file)
        input_ = list()
        for feature in features:
            if feature in ["last_credit_pull_d", "last_pymnt_d"]:
                input_.append(float(date_to_float(data[feature])))
            else:
                input_.append(float(data[feature]))
        return input_

def predict(inputs):
    prediction, _ = model.predict(inputs)
    return map_output[prediction[0][0]]


if __name__=='__main__':
    while True:
        print("Menu:")
        print("1. Predict loan status")
        print("2. Exit")
        choice = input("Enter your choice: ")
        if choice == '1':
            print("\n\tMenu:")
            print("\t1. Data to inform manually")
            print("\t2. Data from json file")
            choice = input("\tEnter your choice: ")
            if choice in ['1', '2']:
                if choice == '1':
                    inputs = get_user_inputs()
                else :
                    file_path = input('\tFill the file path of the json data: ')
                    inputs = get_user_inputs_from_json(file_path)
                inputs = np.array(inputs).reshape(1,-1)
                result = predict(inputs)
                print("\n=============================")
                print(f"PREDICTION RESULT: {result}")
                integrated_gradients = model.compute_integrated_gradients(inputs)
                print(f"\nFeature relevance: {integrated_gradients}")
                print("=============================\n")
        elif choice == '2':
            print("Exiting the application.")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
