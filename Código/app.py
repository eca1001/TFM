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
from sklearn.metrics import f1_score
from sklearn.svm import SVC, LinearSVC

#df = pd.read_csv('./data/breast.csv')
#data = df.drop(['liked'], axis=1)
#target = df['liked']

df = pd.read_csv('./data/BreastCancerWisconsin_(Diagnostic).csv', sep=',')

le = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

data = df.drop(['diagnosis', 'Unnamed: 32'], axis=1)
target = df["diagnosis"] #.map({"B":0, "M": 1})

fs = PowerSHAP(data, target)
fs.select_features()

