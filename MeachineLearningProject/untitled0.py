# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 00:07:20 2022

@author: nedim
"""

import numpy as np 
import pandas as pd 
import matplotlib                  
import matplotlib.pyplot as plt
import seaborn as sns              
plt.style.use('fivethirtyeight')
import sys


# About this dataset
# Age : Age of the patient

# Sex : Sex of the patient

# exang: exercise induced angina (1 = yes; 0 = no)

# ca: number of major vessels (0-3)

# cp : Chest Pain type chest pain type

# Value 1: typical angina
# Value 2: atypical angina
# Value 3: non-anginal pain
# Value 4: asymptomatic
# trtbps : resting blood pressure (in mm Hg)

# chol : cholestoral in mg/dl fetched via BMI sensor

# fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)

# rest_ecg : resting electrocardiographic results

# Value 0: normal
# Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
# Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
# thalach : maximum heart rate achieved

# target : 0= less chance of heart attack 1= more chance of heart attack


################### IMPORT DATA ####################### 
var = pd.read_csv("heart.csv")
#print(var)


################### DO WE HAVE MISSING VALUES ? ########################
cd = var.isnull().sum()
#print(cd)



#################### SEPERATING INPUT AND OUTPUT #######################
x = var.iloc[:,:13].values   #undepended values
#print(x)

y = var.iloc[:,-1].values    #depended value  
#print(y)


#################### SEPERATE TRAININING AND TESTING TO THE DATA ###########################
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

#################### SCALE OPERATION ###############################
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

########################### LogisticRegression #############################
#from sklearn.linear_model import LogisticRegression
#lr = LogisticRegression(random_state=0)
#lr.fit(X_train, y_train)

#y_pred = lr.predict(X_test)
#print(y_pred)


######################### SUPPORT VECTOR MACHINE  #############################

#from sklearn.svm import SVC
#svc = SVC(kernel='linear')
#svc.fit(X_train, y_train)
#y_pred = svc.predict(X_test)

######################### NAIVE BAYES(BERNOILLI) #####################################
#from sklearn.naive_bayes import BernoulliNB
#clf = BernoulliNB()
#clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)



######################### NEURAL NETWORK  #############################
import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()
classifier.add(Dense(6,activation = "relu",input_dim = 13))
classifier.add(Dense(6,activation = "relu"))
classifier.add(Dense(1,activation = "sigmoid"))
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
classifier.fit(X_train,y_train,epochs=20000)
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


########################### CONFUSION MATRIX VISULATION ###########################
from sklearn.metrics import confusion_matrix
import seaborn as sns

cf_matrix = confusion_matrix(y_test, y_pred)

group_names = ['True Neg','False Pos','False Neg','True Pos']

group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]

labels = [f"{v1}\n{v2}" for v1,v2 in zip(group_names,group_counts)]

labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')























