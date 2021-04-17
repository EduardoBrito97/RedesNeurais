from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import VotingClassifier

import scikitplot as skplt
import matplotlib
import matplotlib.pyplot as plt

def create_sklearn_compatible_model(hidden_layers = 1, hidden_neurons = 10, learning_rate = 0.01):
    model = Sequential()
    for _ in range(hidden_layers):
        model.add(Dense(hidden_neurons, activation='tanh', input_dim=input_dim))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    return model

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

## Lê dados
X_train, y_train = get_data('../data/X_train_over.csv', '../data/y_train_over.csv')
X_test, y_test = get_data('../data/X_test.csv', '../data/y_test.csv')

# Cria o dicionário com diferentes mlps para o ensemble
mlps_clf = {}

hidden_layers = [1, 2, 3]
hidden_neurons = [10, 20]
learning_rates = [0.001, 0.01, 0.1]

for hl in hidden_layers:
    for hn in hidden_neurons:
        for lr in learning_rates:
            dict_key = 'mlp_hl-{hl}_hn-{hn}_lr-{lr}'
            # Para cada valor de learning rate, hidden neurons e hidden layers, cria um modelo e atribui à legenda
            mlps_clf[dict_key] = KerasClassifier(build_fn=create_sklearn_compatible_model, 
                                hidden_layers = hl,
                                hidden_neurons = hn,
                                learning_rate = lr,
                                batch_size=64,
                                epochs=100,
                                verbose=0)
# Crias o Ensemble
mlp_ens = VotingClassifier([mlps_clf], voting='soft')

# Treina o Ensemble
mlp_ens.fit(X_train, y_train)

# Prediz os próximos valores
ens_pred_class = mlp_ens.predict(X_test)
ens_pred_scores = mlp_ens.predict_proba(X_test)

accuracy, recall, precision, f1, auroc, aupr = compute_performance_metrics(y_test, ens_pred_class, ens_pred_scores)
print_metrics_summary(accuracy, recall, precision, f1, auroc, aupr)