import pandas as pd
from featureSelection import featureSelection, BorutaSHAP, PowerSHAP, Boruta, Shapicant, Chi2, F1
from sklearn.feature_selection import SelectFdr, SelectFpr, SelectFromModel, SelectFwe, SelectKBest, SelectorMixin, SelectPercentile, SequentialFeatureSelector, GenericUnivariateSelect, RFE, RFECV, VarianceThreshold
from sklearn.preprocessing import LabelEncoder

#funcionan
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

#from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.svm import SVC, LinearSVC

from imbalanced_classification.Ensamble import Ensamble

df = pd.read_csv('./data/BreastCancerWisconsin_(Diagnostic).csv', sep=',')

le = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

data = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
target = df["diagnosis"]

fs = BorutaSHAP(data, target, model=CatBoostClassifier(n_estimators=150))
accepted, rejected, tentative = fs.select_features(plot=True)

fs = Shapicant(data, target, model=RandomForestClassifier())
accepted, rejected, tentative = fs.select_features(alpha=0.25, plot=True)

fs = Chi2(data, target)
accepted, rejected, tentative = fs.select_features(method=SelectFdr)

fs = F1(data, target)
accepted, rejected, tentative = fs.select_features(method=SelectFpr)

"""
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=0)

ens = Ensamble("BalancedBaggingClassifier", {"random_state": 0, "sampling_strategy": "auto"})
ens.fit(X_train,y_train)
y_pred = ens.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

data = data.drop(rejected, axis=1)

X_train, X_test, y_train, y_test = train_test_split(data, target, random_state=0)

ens = Ensamble("BalancedBaggingClassifier", {"random_state": 0, "sampling_strategy": "auto"})
ens.fit(X_train,y_train)
y_pred = ens.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
"""