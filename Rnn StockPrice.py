# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 17:12:18 2020

@author: Flo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Préparation des données

dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")

#Selection de colonne : Double crochet avec le nom de la colonne
#.values pour passer d'un dataframe into Array
training_set = dataset_train[["Open"]].values

"""Standardsation = centrer les données
 Normaliser en divisant par l'écart type."""
 
from sklearn.preprocessing import MinMaxScaler
 
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled =sc.fit_transform(training_set)

""" création de la structure avec 60 timesteps et 1 sortie.
60 Timesteps : Instauration de cycle de 60 jours et trouver des tendances pour t+1. 
La définition d'un timesteps se fait içi par tatonnement.
60 jours de bourses -> 3 mois """

X_train = []
Y_train = []


""" on se place a un temps T et on commence au jour 60 """

for i in range(60, 1258):
    X_train.append(training_set_scaled[(i-60):i])
    Y_train.append(training_set_scaled[i, 0])
    
X_train = np.array(X_train)
Y_train = np.array(Y_train)


"""modification de la dimension
batch_size nombre de jour
TimeSteps (colonne)
input_dim = Nombre de Variables d'entrée (1seule car une action)"""
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

from keras.layers import Dropout 
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential

regresseur = Sequential()


"""http://www.bioinf.jku.at/publications/older/2604.pdf"""

#Couches

"""return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence."""
regresseur.add(LSTM(units = 60, return_sequences=True, 
                    input_shape= (X_train.shape[1], 1)))


"""
Désactivation de neurones aléatoires: à chaque itération z% des neurones seront désactivés
 afin d'empêcher que les neurones ne créer trop de liens entre eux"""
 
regresseur.add(Dropout(0.2))



# on peut enlever la spécification de dimension (input_shape)
regresseur.add(LSTM(units = 60, return_sequences=True))
regresseur.add(Dropout(0.2))

regresseur.add(LSTM(units = 60, return_sequences=True))
regresseur.add(Dropout(0.2))


# return_sequences --> false car dernière couche (valeur de base)
regresseur.add(LSTM(units = 60, return_sequences=False))
regresseur.add(Dropout(0.2))


# Couche de sortie

"""units = 1 car une valeur (neurone) correspondant à la valeur de l'action a t+1"""
regresseur.add(Dense(units = 1))



regresseur.compile(optimizer="adam", loss= "mean_squared_error")


#entrainement
regresseur.fit(X_train, Y_train, epochs = 100, batch_size=32)


dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price = dataset_test[["Open"]].values



#concat = concatenation de données
# 0 axe des ligne /1 axe des colonnes
dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis = 0)

inputs = dataset_total[len(dataset_total) - len(dataset_test)-60:].values
inputs = inputs.reshape(-1, 1)
#Transformattion similaire (normalisation des données)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, 80):
    X_test.append(inputs[(i -60):i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regresseur.predict(X_test)

predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#visualisation 

plt.plot(real_stock_price, color = "red", label="Prix effectif de l'action")
plt.plot(predicted_stock_price, color="green", label= "prédiction du prix de l'action")
plt.title("Prédiction")
plt.xlabel("jour")
plt.ylabel("Prix de l'action")
plt.legend()
plt.show()













