import os

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.svm import LinearSVC
from itertools import combinations
import warnings
import time

# -----------------------------DECISION TREE-----------------------------------------

class Node():
    def __init__(self,
                 weights=None,    # vetor de pesos (tamanho = n_atributos)
                 bias=None,       # viés (intercept) do split
                 left=None,
                 right=None,
                 label=None,
                 certainty=None):

        # para nó de decisão (oblíquo via SVM)
        self.weights = weights
        self.bias = bias
        self.left = left
        self.right = right

        # y_hat para nós folha
        self.label = label
        self.certainty = certainty


class ObliqueSVMDecisionTreeClassifier():
    def __init__(self,
                 max_depth=14,
                 min_samples_split=35,
                 min_info_gain=1e-7,
                 svm_C=1.0,
                 svm_max_iter=2000,
                 random_state=None):

        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_info_gain = min_info_gain
        self.svm_C = svm_C
        self.svm_max_iter = svm_max_iter
        self.rng = np.random.RandomState(random_state)


    def build_tree(self, X, Y, depth=0):

        if len(np.unique(Y)) == 1:
            return Node(label=Y[0], certainty=1.0)

        # Para em depth máximo ou número mínimo de amostras (evitar overfitting)
        if depth >= self.max_depth or len(Y) < self.min_samples_split:
            return Node(label=self.calculate_leaf_label(Y), certainty=self.calculate_certainty(Y))

        weights, bias, best_gain = self.get_best_split(X, Y)

        # se não ganhou informação suficiente, cria nó folha
        if weights is None or best_gain < self.min_info_gain:
            return Node(label=self.calculate_leaf_label(Y), certainty=self.calculate_certainty(Y))

        projection = X @ weights + bias
        left_mask = projection <= 0
        right_mask = projection > 0

        Xi_left, Yi_left = X[left_mask], Y[left_mask]
        Xi_right, Yi_right = X[right_mask], Y[right_mask]

        # Se for um nó puro, cria nó folha
        if len(Yi_left) == 0 or len(Yi_right) == 0:
            return Node(label=self.calculate_leaf_label(Y), certainty=self.calculate_certainty(Y))

        left_subtree = self.build_tree(Xi_left, Yi_left, depth + 1)
        right_subtree = self.build_tree(Xi_right, Yi_right, depth + 1)

        return Node(weights=weights, bias=bias, left=left_subtree, right=right_subtree)


    # busca do melhor split oblíquo usando SVM 
    def get_all_class_groupings(self, classes):
        """Gera todos os agrupamentos possíveis das classes em 2 grupos não-vazios,
        sem repetir. Como o SVM é binário, a gente tem que dividir as classes em 
        dois grupos pra tereinar"""

        classes = list(classes)
        groupings = []
        # tamanho do grupo A vai de 1 até metade das classes (p/ evitar duplicar complementos)
        for size in range(1, len(classes) // 2 + 1):
            for combo in combinations(classes, size):
                # evita duplicar quando size == len(classes)/2 (A|B e B|A seriam iguais)
                if size == len(classes) - size and combo[0] != classes[0]:
                    continue
                groupings.append(set(combo))
        return groupings

    def get_best_split(self, X, Y):
        parent_entropy = self.entropy_calc(Y)
        classes = np.unique(Y)

        best_gain = -float("inf")
        best_weights = None
        best_bias = None

        groupings = self.get_all_class_groupings(classes)

        for group_a in groupings:

            y_binary = np.array([0 if c in group_a else 1 for c in Y])

            if len(np.unique(y_binary)) < 2:
                print("Agrupamento degenerado, pulando...")
                continue  # agrupamento degenerado, pula

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # ignora warnings de não-convergência
                    svm = LinearSVC(C=self.svm_C, max_iter=self.svm_max_iter, dual="auto")
                    svm.fit(X, y_binary)
            except Exception:
                continue  # se a SVM falhar por algum motivo, ignora essa tentativa

            weights = svm.coef_[0]
            bias = svm.intercept_[0]

            projection = X @ weights + bias
            left_mask = projection <= 0
            right_mask = projection > 0
            Yi_left = Y[left_mask]
            Yi_right = Y[right_mask]

            if len(Yi_left) > 0 and len(Yi_right) > 0:
                curr_gain = self.information_gain(parent_entropy, Yi_left, Yi_right)

                if curr_gain > best_gain:
                    best_gain = curr_gain
                    best_weights = weights
                    best_bias = bias

        return best_weights, best_bias, best_gain

    # ---------------------------------------------------------
    # entropia / ganho de informação
    # ---------------------------------------------------------
    def entropy_calc(self, y):
        values, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return entropy(p, base=2)

    def information_gain(self, parent_entropy, l_child, r_child):
        n_left, n_right = len(l_child), len(r_child)
        ita = n_left / (n_left + n_right)
        return parent_entropy - ita * self.entropy_calc(l_child) - (1 - ita) * self.entropy_calc(r_child)

    def calculate_leaf_label(self, Y):
        values, counts = np.unique(Y, return_counts=True)
        return values[np.argmax(counts)]
    
    def calculate_certainty(self, Y):
        values, counts = np.unique(Y, return_counts=True)
        return np.max(counts) / len(Y)

    # ---------------------------------------------------------
    # treino e predição
    # ---------------------------------------------------------
    def fit(self, X, Y):
        self.root = self.build_tree(X, Y)

    def predict(self, X):
        return [self.make_prediction(x, self.root) for x in X]

    def make_prediction(self, x, tree):
        if tree.label is not None:  # folha
            return tree.label, tree.certainty

        projection = np.dot(x, tree.weights) + tree.bias
        if projection <= 0:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
        

# -----------------------------RANDOM FOREST-----------------------------------------

class ObliqueSVMRandomForestClassifier():
    def __init__(self,
                 n_estimators=50,
                 max_depth=14,
                 min_samples_split=35,
                 min_info_gain=1e-7,
                 svm_C=1.0,
                 svm_max_iter=2000,
                 max_features=20,
                 bootstrap=True,
                 random_state=None,
                 flag_certainty=False):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_info_gain = min_info_gain
        self.svm_C = svm_C #penalty error
        self.svm_max_iter = svm_max_iter
        self.max_features = max_features
        self.flag_certainty = flag_certainty
        self.bootstrap = bootstrap # reposição de linhas (True) ou não (False)
        self.rng = np.random.RandomState(random_state)

        self.trees = []
    
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

            tree = ObliqueSVMDecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_info_gain=self.min_info_gain,
                svm_C=self.svm_C,
                svm_max_iter=self.svm_max_iter,
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

        # votação ponderada pela certeza da folha (ou majoritária simples, se flag desativada)
        final_preds = []
        for j in range(n):
            votes = all_labels[:, j]

            if self.flag_certainty:
                weights = all_certainties[:, j]
                classes = np.unique(votes)
                scores = {c: weights[votes == c].sum() for c in classes}   # soma certeza por classe
                final_preds.append(max(scores, key=scores.get))
            else:
                values, counts = np.unique(votes, return_counts=True)      # conta votos por classe
                final_preds.append(values[np.argmax(counts)])

        return final_preds
    
    def print_info(self):
        print("Random Forest com SVM oblíquo")
        print(f"n_estimators: {self.n_estimators}")
        print(f"max_depth: {self.max_depth}")
        print(f"min_samples_split: {self.min_samples_split}")
        print(f"min_info_gain: {self.min_info_gain}")
        print(f"svm_C: {self.svm_C}")
        print(f"svm_max_iter: {self.svm_max_iter}")
        print(f"max_features: {self.max_features}")
        print(f"flag_certainty: {self.flag_certainty}")
            

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

    model = ObliqueSVMRandomForestClassifier()
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
    list_hyperparametro = [10, 30, 50, 70, 100]
    accuracys = []
    tempos = []
    for n in list_hyperparametro:
        inicio = time.perf_counter()
        forest = ObliqueSVMRandomForestClassifier(
            n_estimators=n,
            max_depth=14,
            min_samples_split=35,
            svm_C=1.0,
            random_state=42,
            max_features=23,
            flag_certainty=False
        )
        forest.print_info()
        forest.fit(X_tr, y_tr) 

        y_val_pred = forest.predict(X_val)
        fim = time.perf_counter()
        print(f"Tempo de execução para {n} estimadores: {fim - inicio:.2f} segundos")
        print("Acurácia validação (floresta):", accuracy(y_val, y_val_pred))
        accuracys.append(accuracy(y_val, y_val_pred))
        tempos.append(fim - inicio)
    print("Acurácias para diferentes números de estimadores:", accuracys)
    print("Tempos de execução para diferentes números de estimadores:", tempos)

# main()
teste()