import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier

# read dataset
var = pd.read_csv("heart.csv")

# check if empty slot exists and remove it
cd = var.isnull().sum()

# actual data
y_true = var.head(204)
y_true = y_true.iloc[:,-1]

y_test = var.tail(100)
y_test = y_test.iloc[:,-1]

print("y_true")
print(y_true)
print("y_test")
print(y_test)

# pick 70% of the data randomly
xTrain = var.head(204)
xTrain = xTrain.iloc[:,:13]

# pick last 30% of the data
xTest = var.tail(100)
xTest = xTest.iloc[:,:13]

print("x train")
print(xTrain)
print("x test")
print(xTest)

# random forest classifier with 100 estimators
RandomForestClassifier = RandomForestClassifier(n_estimators=100)

# feature selection using sequential feature selector with at least 5 picked using random forest classifier 
featureSelector = SequentialFeatureSelector(estimator=RandomForestClassifier, n_features_to_select=5)

# fit train set and actual prediction
featureSelector.fit(xTrain,y_true)

# display which data column should be picked
print("Selected feautures")
print(featureSelector.get_support())

chosen = featureSelector.get_support()

var = pd.read_csv("heart.csv")
new_dataset = var.iloc[:,:13]
print(new_dataset)
new_dataset = new_dataset.loc[:, featureSelector.get_support()]
print(new_dataset)