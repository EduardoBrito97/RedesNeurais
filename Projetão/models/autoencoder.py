from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from keras.models import Model
from keras.utils import plot_model
from keras.layers import Dense, BatchNormalization, LeakyReLU, Input
from keras.callbacks import EarlyStopping

import pandas as pd

import matplotlib.pyplot as plt

def get_data(path_atributes, path_labels):

    y = pd.read_csv(path_labels, index_col=0).values.squeeze()
    X = pd.read_csv(path_atributes, index_col=0).values
    
    return X, y 

X_train, y_train = get_data('../data/X_train_over.csv', '../data/y_train_over.csv')
X_val, y_val = get_data('../data/X_valid.csv', '../data/y_valid.csv')

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

n_inputs = int(X_train.shape[1])

## ENCODER
visible = Input(shape=(n_inputs,))

e = Dense(2*n_inputs)(visible)
e = BatchNormalization()(e)
e = LeakyReLU()(e)

e = Dense(n_inputs)(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)

e = Dense(int(n_inputs / 5))(e)
e = BatchNormalization()(e)
e = LeakyReLU()(e)

n_bottleneck = int(n_inputs / 10)
bottleneck = Dense(n_bottleneck)(e)

## DECODER
d = Dense(int(n_inputs/5))(bottleneck)
d = BatchNormalization()(d)
d = LeakyReLU()(d)

d = Dense(n_inputs)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)

d = Dense(2*n_inputs)(d)
d = BatchNormalization()(d)
d = LeakyReLU()(d)

output = Dense(n_inputs, activation='linear')(d)

model = Model(inputs=visible, outputs=output)
model.compile(optimizer='adam', loss='mse')
#plot_model(model, to_file='../results/mlp_autoencoder/plots/model_plot.png', show_shapes=True, show_layer_names=True)

history = model.fit(X_train, X_train, epochs=200, batch_size=4096, callbacks=[EarlyStopping(patience=5)], verbose=1, validation_data=(X_val, X_val))

encoder = Model(inputs=visible, outputs=bottleneck)
encoder.save('autoencoder.h5')

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()