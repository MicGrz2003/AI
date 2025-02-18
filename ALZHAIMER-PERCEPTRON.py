import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#-----------------------------dane-------------------------------------
zrodlo = "source to dataset alzheimers_disease_data.csv from kaggle"
df = pd.read_csv(zrodlo)

X = df.iloc[:,1:33]
X = X.to_numpy()

y = df.iloc[:,33]
y = y.to_numpy()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
b = 1
#-----------------------------------------------------------------------

def f_sigmoid(x):
    return 1/(1+np.exp(-x*b))
def d_f_sigmoid(x):
    return f_sigmoid(x) * (1 - f_sigmoid(x))


def train_loop(X_train,y_train,X_test,y_test):

    #-----------------------------definiowanie parametrów sieci-----------------------------------
    m = len(X_train[0])
    liczba_neurony_ukryte = 30
    liczba_neurony_outputowe = 1

    wagi_wstep = np.random.uniform(-1,1, size = (liczba_neurony_ukryte ,m))
    wagi_ukryte = np.random.uniform(-1,1, size = (liczba_neurony_outputowe, liczba_neurony_ukryte))
    bias_wstep = np.random.uniform(-1,1, size = (1,liczba_neurony_ukryte))
    bias_ukryte = np.random.uniform(-1,1, size = (1,liczba_neurony_outputowe))

    learning_rate = 0.1
    epochs = 100
    straty_train = []
    straty_test = []

    #-----------------------------------petla uczenia---------------------------------------
    for i in range(epochs):
        strata_w_epoch = 0
        for j in range(len(y_train)):
            X = X_train[j].reshape(1, -1)
            y = y_train[j].reshape(1, -1)

            #-------------forward pass------------
            # warstwa ukryta
            z1 = np.dot(X, wagi_wstep.T) + bias_wstep
            a1 = f_sigmoid(z1)

            # warstwa wyjsciowa
            z2 = np.dot(a1,wagi_ukryte.T) + bias_ukryte
            a2 = f_sigmoid(z2)

            #--------f strat-----------
            strata = 0.5 * (a2 - y) ** 2 # średni bląd kwadratowy
            strata_w_epoch += np.sum(strata)

            #-----------backpropagation-----------------
            de_tyl = (a2-y)*d_f_sigmoid(z2)
            d_wagi_ukryte = np.dot(de_tyl.T,a1)
            d_bias_ukryty = de_tyl

            de_przod = np.dot(de_tyl,wagi_ukryte)
            d_wagi_wstep = np.dot(de_przod.T,X)
            d_bias_wstep = de_przod

            wagi_ukryte = wagi_ukryte - learning_rate*d_wagi_ukryte
            bias_ukryte = bias_ukryte - learning_rate*d_bias_ukryty
            wagi_wstep = wagi_wstep - learning_rate*d_wagi_wstep
            bias_wstep = bias_wstep - learning_rate * d_bias_wstep

        #///////////////////zbior testowy\\\\\\\\\\\\\\\\\\\\
        strata_test_epoch = 0
        for j in range(len(y_test)):
            Xt = X_test[j].reshape(1, -1)
            yt = y_test[j].reshape(1, -1)

            # -------------forward pass------------
            # warstwa ukryta
            z1t = np.dot(Xt, wagi_wstep.T) + bias_wstep
            a1t = f_sigmoid(z1t)

            # warstwa wyjsciowa
            z2t = np.dot(a1t, wagi_ukryte.T) + bias_ukryte
            a2t = f_sigmoid(z2t)

            # --------f strat-----------
            stratat = 0.5 * (a2t - yt) ** 2  # średni bląd kwadratowy
            strata_test_epoch += np.sum(stratat)

        #\\\\\\\\\\\\\\\\\\\\\\\\//////////////////////////

        #print
        straty_train.append(strata_w_epoch/len(y_train)) # srednia straty w epoce
        straty_test.append(strata_test_epoch / len(y_test))
        print(f"epoka: {i}, strata_trening: {straty_train[i]}, strata_test: {straty_test[i]}")

    print("ukończono uczenie sieci!")
    return straty_train,straty_test

def wizu(sttr, stte):
    Xw = np.arange(len(sttr))
    X2w = np.arange(len(stte))

    plt.figure(figsize=(10, 6))
    plt.plot(Xw, sttr, label="Strata treningowa", color='blue', linewidth=2)
    plt.plot(X2w, stte, label="Strata testowa", color='orange', linewidth=2, linestyle='--')
    plt.title("Wykres strat treningowych i testowych w czasie epok", fontsize=14)
    plt.xlabel("Epoki", fontsize=12)
    plt.ylabel("Strata", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def wywolanie():
    sttr , stte = train_loop(X_train,y_train,X_test,y_test)
    wizu(sttr,stte)

wywolanie()