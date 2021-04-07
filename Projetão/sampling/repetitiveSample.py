import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle

def get_data():

    ## Pega todos os dados de treino (variáveis normalizadas e não normalizadas)
    originalTrain = pd.read_csv('originalData/BASE-PREPROCESSED(TRAIN)', sep='\t', engine='python')
    ## Pega o que vai ser útil pra treino (variáveis normalizadas)
    usefulTrain = originalTrain.loc[:, ['PROPHET_LABEL', 'PROPHET_NORM_FEATURES']]
    ## Pega todos os dados de validação (variáveis normalizadas e não normalizadas)
    originalValid = pd.read_csv('originalData/BASE-PREPROCESSED(VALIDACAO)', sep='\t', engine='python')
    ## Pega o que vai ser útil pra validação (variáveis normalizadas)
    usefulValid = originalValid.loc[:, ['PROPHET_LABEL', 'PROPHET_NORM_FEATURES']]
    
    aux = []
    ## As entradas das variáveis de entrada são vistas como string, mas queremos transformá-las em listas de float
    ## Eliminamos primeiro os colchetes que envolvem a string pra cada entrada
    usefulTrain.loc[:, 'PROPHET_NORM_FEATURES'] = usefulTrain.loc[:, 'PROPHET_NORM_FEATURES'].apply(lambda x: x.replace('[', ''))
    usefulTrain.loc[:, 'PROPHET_NORM_FEATURES'] = usefulTrain.loc[:, 'PROPHET_NORM_FEATURES'].apply(lambda x: x.replace(']', ''))
    ## Transformamos a string em lista, separando a string nas virgúlas
    usefulTrain.loc[:, 'PROPHET_NORM_FEATURES'] = usefulTrain.loc[:, 'PROPHET_NORM_FEATURES'].apply(lambda x: x.split(','))
    ## Transforma todos os elementos das listas de todas as entradas em float
    usefulTrain.loc[:, 'PROPHET_NORM_FEATURES'] = usefulTrain.loc[:, 'PROPHET_NORM_FEATURES'].apply(lambda x: np.array(x, dtype=np.float64))
    ## Transforma as listas que se encontram em cada linha do dataframe em colunas(cada valor em cada coluna)
    for label, content in tqdm(usefulTrain[['PROPHET_NORM_FEATURES']].iterrows()):
        aux.append(content['PROPHET_NORM_FEATURES'])
    
    aux = pd.DataFrame(aux)
    aux.insert(0, 'PROPHET_LABEL', usefulTrain['PROPHET_LABEL'])
    usefulTrain = aux
    
    aux = []
    ## Eliminamos primeiro os colchetes que envolvem a string pra cada entrada
    usefulValid.loc[:, 'PROPHET_NORM_FEATURES'] = usefulValid.loc[:, 'PROPHET_NORM_FEATURES'].apply(lambda x: x.replace('[', ''))
    usefulValid.loc[:, 'PROPHET_NORM_FEATURES'] = usefulValid.loc[:, 'PROPHET_NORM_FEATURES'].apply(lambda x: x.replace(']', ''))
    ## Transformamos a string em lista, separando a string nas virgúlas
    usefulValid.loc[:, 'PROPHET_NORM_FEATURES'] = usefulValid.loc[:, 'PROPHET_NORM_FEATURES'].apply(lambda x: x.split(','))
    ## Transforma todos os elementos das listas de todas as entradas em float
    usefulValid.loc[:, 'PROPHET_NORM_FEATURES'] = usefulValid.loc[:, 'PROPHET_NORM_FEATURES'].apply(lambda x: np.array(x, dtype=np.float32))
    ## Transforma as listas que se encontram em cada linha do dataframe em colunas(cada valor em cada coluna)
    for label, content in tqdm(usefulValid[['PROPHET_NORM_FEATURES']].iterrows()):
        aux.append(content['PROPHET_NORM_FEATURES'])
    
    aux = pd.DataFrame(aux)
    aux.insert(0, 'PROPHET_LABEL', usefulValid['PROPHET_LABEL'])
    usefulValid = aux

    return usefulTrain, usefulValid

def oversampling(majorClassSize, minorClass):

    limit = majorClassSize - minorClass.shape[0]

    for entry in tqdm(range(0, limit)):
        minorClass = minorClass.append(minorClass.iloc[entry, :])

    return minorClass

def separateClasses(data):

    class1 = data.loc[data['PROPHET_LABEL'] == 0]
    class2 = data.loc[data['PROPHET_LABEL'] == 1]

    return class1, class2

def joinAndShuffle(class1, class2):

    class1 = class1.append(class2)
    class1 = shuffle(class1)

    return class1

## importa dados
usefulTrain, usefulValid = get_data()

## separa classes majoritária e minoritária (sempre majoritária 0 e minoritária 1)
majorClassTrain, minorClassTrain = separateClasses(usefulTrain)
majorClassValid, minorClassValid = separateClasses(usefulValid)

## faz repetitive oversampling com classes minoritárias
minorClassTrain = oversampling(majorClassTrain.shape[0], minorClassTrain)
minorClassValid = oversampling(majorClassValid.shape[0], minorClassValid)

## junta tuda e faz o shuffle
readyDataTrain = joinAndShuffle(majorClassTrain, minorClassTrain)
readyDataValid = joinAndShuffle(majorClassValid, minorClassValid)

## elimina os índices antigos, que se tornaram inúteis pois estão fora de ordem
readyDataTrain = readyDataTrain.reset_index(drop=True)
readyDataValid = readyDataValid.reset_index(drop=True)

readyDataTrain.to_csv('readyData/readyDataTrain.csv')
readyDataValid.to_csv('readyData/readyDataValid.csv')