import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_squared_error
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

import scikitplot as skplt
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier

def compute_performance_metrics(y, y_pred_class, y_pred_scores=None):
    mse = mean_squared_error(y, y_pred_class)
    accuracy = accuracy_score(y, y_pred_class)
    recall = recall_score(y, y_pred_class)
    precision = precision_score(y, y_pred_class)
    f1 = f1_score(y, y_pred_class)
    performance_metrics = (mse, accuracy, recall, precision, f1)
    if y_pred_scores is not None:
        skplt.metrics.plot_ks_statistic(y, y_pred_scores)
        plt.savefig('../results/ensemble/ks.png')
        plt.clf()
        y_pred_scores = y_pred_scores[:, 1]
        auroc = roc_auc_score(y, y_pred_scores)
        aupr = average_precision_score(y, y_pred_scores)
        performance_metrics = performance_metrics + (auroc, aupr)

    return performance_metrics

def print_metrics_summary(mse, accuracy, recall, precision, f1, auroc=None, aupr=None):

    print()
    print("{metric:<18}{value:.4f}".format(metric="MSE:", value=mse))
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

X_train, y_train = get_data('../data/X_train_over.csv', '../data/y_train_over.csv')
X_test, y_test = get_data('../data/X_test.csv', '../data/y_test.csv')

mlp = MLPClassifier(hidden_layer_sizes=(20),
                                            activation='relu',
                                            learning_rate_init=0.001,
                                            learning_rate='constant',
                                            solver='sgd',
                                            max_iter=100,
                                            )
rfc = RandomForestClassifier(n_estimators=100, verbose=True, max_depth=20)
gbc = GradientBoostingClassifier(n_estimators=100)

ensemble = VotingClassifier([
    ('mlp', mlp),
    ('rfc', rfc),
    ('gbc', gbc)
], voting='soft')

cv = ShuffleSplit(n_splits=5, test_size=0.33, random_state=0)
scores = cross_validate(ensemble, X_train, y_train.round(), cv=cv, return_estimator=True)

bestGbc = scores['estimator'][np.argmax(scores['test_score'])]
y_pred = bestRfc.predict(X_test)
y_pred_scores = bestRfc.predict_proba(X_test)

mse, accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test.round(), y_pred, y_pred_scores)

results = {'MSE': [], 'ACC': [], 'REC': [], 'PREC': [], 'F1': [], 'AUROC': [], 'AUPR': []}
results['MSE'].append(mse)
results['ACC'].append(accuracy)
results['REC'].append(recall)
results['PREC'].append(precision)
results['F1'].append(f1)
results['AUROC'].append(auroc)
results['AUPR'].append(aupr)

print('Performance no conjunto de teste:')
print_metrics_summary(mse, accuracy, recall, precision, f1, auroc, aupr)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True)
plt.savefig('../results/ensemble/matrix.png')
plt.clf()

data = pd.DataFrame(results)
data.to_csv('../results/ensemble/results.csv')