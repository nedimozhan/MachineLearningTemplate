import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# True = include, False = don't include
################# age,  sex,   cp,  trestbps,chol,  fbs,   restecg, thalach, exang, oldpeak, slope, ca,   thal
FeatureColumns = [True, False, True, False,  False, False, True,    True,    False, True,    False, True, True]

# read dataset and remove unclean data
dataset = pd.read_csv("heart.csv")
cd = dataset.isnull().sum()

# Select last column
y = dataset.iloc[:,-1]

# Select rest
x = dataset.iloc[:,:13]

# apply columns to be included
x = x.loc[:, FeatureColumns]

# 80% train 20% test, seed 100
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

# random forest with 250 estimators
RF = RandomForestClassifier(n_estimators=250)
RF.fit(X_train, y_train)

# predict
y_pred = RF.predict(X_test)

# print accuracy
# 0 not at risk, 1 at risk
print(metrics.classification_report(y_test,y_pred))
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))