import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import VotingClassifier

import scikitplot as skplt
import matplotlib
import matplotlib.pyplot as plt

def compute_performance_metrics(y, y_pred_class, y_pred_scores=None):
    accuracy = accuracy_score(y, y_pred_class)
    recall = recall_score(y, y_pred_class)
    precision = precision_score(y, y_pred_class)
    f1 = f1_score(y, y_pred_class)
    performance_metrics = (accuracy, recall, precision, f1)
    if y_pred_scores is not None:
        skplt.metrics.plot_ks_statistic(y, y_pred_scores)
        plt.show()
        y_pred_scores = y_pred_scores[:, 1]
        auroc = roc_auc_score(y, y_pred_scores)
        aupr = average_precision_score(y, y_pred_scores)
        performance_metrics = performance_metrics + (auroc, aupr)
    return performance_metrics

def print_metrics_summary(accuracy, recall, precision, f1, auroc=None, aupr=None):
    print()
    print("{metric:<18}{value:.4f}".format(metric="Accuracy:", value=accuracy))
    print("{metric:<18}{value:.4f}".format(metric="Recall:", value=recall))
    print("{metric:<18}{value:.4f}".format(metric="Precision:", value=precision))
    print("{metric:<18}{value:.4f}".format(metric="F1:", value=f1))
    if auroc is not None:
        print("{metric:<18}{value:.4f}".format(metric="AUROC:", value=auroc))
    if aupr is not None:
        print("{metric:<18}{value:.4f}".format(metric="AUPR:", value=aupr))

def get_data(path_atributes, path_labels):

    y = pd.read_csv(path_labels, index_col=0).values.squeeze()
    X = StandardScaler().fit_transform(pd.read_csv(path_atributes, index_col=0))
    
    return X, y

## Lê dados
X_train, y_train = get_data('../data/X_train_over.csv', '../data/y_train_over.csv')
X_test, y_test = get_data('../data/X_test.csv', '../data/y_test.csv')

# Cria o dicionário com diferentes mlps para o ensemble

hidden_layers = [1, 2, 3]
hidden_neurons = [10, 20]
learning_rates = [0.001, 0.01, 0.1]
index = 0

mlps_clf = []

for hl in hidden_layers:
    for hn in hidden_neurons:
        for lr in learning_rates:
            hidden_layers_size = (hn)
            if hl == 2:
                hidden_layers_size = (hn, hn)
            elif hl == 3:
                hidden_layers_size = (hn, hn, hn)

            # Para cada valor de learning rate, hidden neurons e hidden layers, cria um modelo e atribui à legenda
            mlp_ens_clf = MLPClassifier(hidden_layer_sizes=hidden_layers_size,
                                        activation='relu',
                                        learning_rate_init=lr,
                                        learning_rate='constant',
                                        max_iter=100,
                                        )
            mlp_alias = 'mlp_hl-' + str(hl) + '_hn-' + str(hn) + '_lr-' + str(lr)
            mlps_clf.append( (mlp_alias, mlp_ens_clf) )

# Crias o Ensemble
mlp_ens = VotingClassifier(mlps_clf, voting='soft')

# Treina o Ensemble
mlp_ens.fit(X_train, y_train)

# Prediz os próximos valores
ens_pred_class = mlp_ens.predict(X_test)
ens_pred_scores = mlp_ens.predict_proba(X_test)

accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, ens_pred_class, ens_pred_scores)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)