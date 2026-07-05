import os

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.svm import LinearSVC
from itertools import combinations
import warnings

# -----------------------------DECISION TREE-----------------------------------------
class Node():
    def __init__(self, 
                 feature_i_star=None, 
                 th_star=None, 
                 left=None, 
                 right=None, 
                 label=None):
        
        # for decision node
        self.feature_i_star = feature_i_star
        self.th_star = th_star
        self.left = left
        self.right = right
        
        # y_hat for leaf nodes
        self.label = label

class DecisionTreeClassifier():
    def __init__(self):
        self.root = None
        
    def build_tree(self, X, Y):

        if len(np.unique(Y)) == 1:
            return Node(label=Y[0])

        n, m = X.shape

        feature_i_star, th_star = self.get_best_split(X, Y)
        #print(feature_i_star, th_star)
        if feature_i_star == -1:
             return Node(label=self.calculate_leaf_label(Y))

        #***Implemente*** o split dos dados para enviar para os filhos
        feature_values = X[:, feature_i_star]
        left_mask = feature_values<=th_star
        right_mask = feature_values>th_star
        Yi_left = Y[left_mask]
        Yi_right = Y[right_mask]
        Xi_left = X[left_mask]
        Xi_right = X[right_mask]
        #Xi_left = ????
        #Yi_left = ????

        #Xi_right = ????
        #Yi_right = ????

        if len(Xi_left) == 0 or len(Xi_right) == 0:
            return Node(label=self.calculate_leaf_label(Y))

        left_subtree = self.build_tree(Xi_left, Yi_left)
        right_subtree = self.build_tree(Xi_right, Yi_right)
        
        return Node(feature_i_star, th_star, 
                    left_subtree, right_subtree)
    
    def get_best_split(self, X, Y):
            i_star, th_star = -1, -1
            max_info_gain = -float("inf")
            
            m = X.shape[-1]
            for feature_i in range(m):
                feature_values = X[:, feature_i]
                
                feature_values_sorted = np.sort(feature_values)
                thresholds = []
                for i in range(len(feature_values)):
                    # th = (feature_values_sorted[i]+feature_values_sorted[i+1])/2
                    th = feature_values[i]
                    thresholds.append(th)

                for th in thresholds:
                                          
                        left_mask = feature_values<=th
                        right_mask = feature_values>th
                        Yi_left = Y[left_mask]
                        Yi_right = Y[right_mask]
                        Xi_left = X[left_mask]
                        Xi_right = X[right_mask]

                        if len(Xi_left)>0 and len(Xi_right)>0:
                            curr_info_gain = self.information_gain(Y, Yi_left, Yi_right)

                            if curr_info_gain>max_info_gain:
                                i_star = feature_i
                                th_star = th
                                max_info_gain = curr_info_gain
                            
            return i_star, th_star
    
    def entropy_calc(self, y):
         values, counts = np.unique(y, return_counts=True)
         p = counts / counts.sum()
         return entropy(p, base=2)

    def information_gain(self, parent, l_child, r_child):
           ita = len(l_child) / len(parent)
           return self.entropy_calc(parent) - ita*self.entropy_calc(l_child) - (1-ita)*self.entropy_calc(r_child)
    
        
    def calculate_leaf_label(self, Y):
        values, counts = np.unique(Y, return_counts=True)
        return values[np.argmax(counts)]

    def fit(self, X, Y):
        self.root = self.build_tree(X, Y)

    def predict(self, X):
        predictions = []
        for x in X:
            y_hat  = self.make_prediction(x, self.root)
            predictions.append(y_hat)

        return predictions

    def make_prediction(self, x, tree):
        
        if tree.label is not None: #Leaf
            return tree.label
        
        feature_val = x[tree.feature_i_star]
        if feature_val<=tree.th_star:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
        

# -----------------------------RANDOM FOREST-----------------------------------------

class RandomForestClassifier():
    def __init__(self,
                 n_estimators=50,
                 max_depth=14,
                 min_samples_split=35,
                 min_info_gain=1e-7,
                 svm_max_iter=2000,
                 max_features=20,
                 bootstrap=True,
                 random_state=None):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_info_gain = min_info_gain
        self.svm_max_iter = svm_max_iter
        self.max_features = max_features
        self.bootstrap = bootstrap # reposição de linhas (True) ou não (False)
        self.rng = np.random.RandomState(random_state)

        self.trees = []

    def print_info(self):
        print("Random Forest com SVM oblíquo")
        print(f"n_estimators: {self.n_estimators}")
        print(f"max_depth: {self.max_depth}")
        print(f"min_samples_split: {self.min_samples_split}")
        print(f"min_info_gain: {self.min_info_gain}")
        print(f"svm_max_iter: {self.svm_max_iter}")
        print(f"max_features: {self.max_features}")
        
    def fit(self, X, Y):
        n, m = X.shape
        self.trees = []

        for i in range(self.n_estimators):
            # montando cada estimador:
            if self.bootstrap:
                row_idx = self.rng.choice(n, size=n, replace=True)
            else:
                row_idx = np.arange(n)

            X_boot = X[row_idx]
            Y_boot = Y[row_idx]

            # --- subamostragem de atributos (colunas) ---
            if self.max_features is not None:
                k = min(self.max_features, m)
                feat_idx = self.rng.choice(m, size=k, replace=False)
            else:
                feat_idx = np.arange(m)

            X_boot_sub = X_boot[:, feat_idx]

            tree_seed = self.rng.randint(0, 2**31 - 1)

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_info_gain=self.min_info_gain,
                random_state=tree_seed
            )
            tree.fit(X_boot_sub, Y_boot)

            self.trees.append((tree, feat_idx))

        return self

    def predict(self, X):
        n = X.shape[0]

        all_labels = []
        all_certainties = []

        for tree, feat_idx in self.trees:
            X_sub = X[:, feat_idx]
            preds = tree.predict(X_sub)
            labels = [p[0] for p in preds]
            certainties = [p[1] for p in preds]
            all_labels.append(labels)
            all_certainties.append(certainties)

        all_labels = np.array(all_labels)
        all_certainties = np.array(all_certainties)

        # votação ponderada pela certeza da folha, amostra por amostra
        final_preds = []
        for j in range(n):
            votes = all_labels[:, j]
            weights = all_certainties[:, j]

            classes = np.unique(votes)
            scores = {c: weights[votes == c].sum() for c in classes}

            final_preds.append(max(scores, key=scores.get))

        return final_preds 

# -----------------------------USO-----------------------------------------
def main():
    print("Carregando dados...")
    path_data = os.path.join(os.path.dirname(__file__), "data/data.npz")
    data = np.load(path_data)
    X_train = data["X_train"]
    y_train = data["y_train"]
    print("X_train carregado de forma: ", X_train.shape)
    print("y_train carregado de forma: ", y_train.shape)

    X_test = data['X_test']

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_hat = model.predict(X_test)
    submission_df = pd.DataFrame({
    'ID': np.arange(1, len(y_hat) + 1),
    'Prediction': y_hat
    })
    submission_df.to_csv("submission.csv", index=False)
    print("Arquivo de submissão salvo em submission.csv")

# -----------------------------TESTANDO FORA DO KAGGLE-----------------------------------------
def train_val_split(X, Y, val_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    indices = rng.permutation(n)
    n_val = int(n * val_ratio)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]

def accuracy(y_true, y_pred):
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)

def teste():
    # --- Carrega os dados ---
    path_data = os.path.join(os.path.dirname(__file__), "data/data.npz")
    data = np.load(path_data)
    X_train = data["X_train"]
    y_train = data["y_train"]

    X_tr, y_tr, X_val, y_val = train_val_split(X_train, y_train, val_ratio=0.2, seed=42)

    print(f"Treino: {X_tr.shape[0]} amostras | Validação: {X_val.shape[0]} amostras")
    forest = RandomForestClassifier(
        n_estimators=50,
        max_depth=14,
        min_samples_split=35,
        random_state=42,
        max_features=23
    )
    forest.print_info()
    forest.fit(X_tr, y_tr) 

    y_val_pred = forest.predict(X_val)
    print("Acurácia validação (floresta):", accuracy(y_val, y_val_pred))

# main()
teste()