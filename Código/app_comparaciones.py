import time
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, ParameterSampler
import pandas as pd
import numpy as np
from featureSelection import BorutaSHAP, PowerSHAP, Boruta, Shapicant, Chi2, F1
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression

import time

from sklearn.feature_selection import SelectFdr, SelectFpr, SelectFromModel, SelectFwe, SelectKBest, SelectorMixin, SelectPercentile, SequentialFeatureSelector, GenericUnivariateSelect, RFE, RFECV, VarianceThreshold

from imbalanced_classification.models import IMBCatboostClassifier
from imbalanced_classification.Combine import Combine
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import csv
import copy

def lanzador(data, target, models, params, args):

    archivo = "resultados.csv"

    with open(archivo, mode='w', newline='') as archivo_csv:

        escritor_csv = csv.writer(archivo_csv)

        ens = Combine("SMOTETomek")
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
        X_res, y_res = ens.fit_resample(X_train, y_train)

        fila = ["Tecnica", "Modelo", "Parametros", "fit_kwargs", "Tiempo", "Accuracy Todas", "Accuracy Aceptadas", "F1 score Todas", "F1 score Aceptadas", "len(rejected)", "Rejected"]
        escritor_csv.writerow(fila)
        for method in ["Boruta", "Shapicant", "BorutaShap", "PowerShap"]:
        #for method in ["Boruta", "BorutaShap", "PowerShap"]:
        #for method in ["Shapicant"]:
            for model in models:
                if method == "BorutaShap": 
                    param = params['BorutaSHAP']
                elif method == "PowerShap":
                    param = params['PowerSHAP']
                elif method == "Boruta":
                    param = params['Boruta']
                elif method == "Shapicant": 
                    param = params['Shapicant']
                else:
                    raise ValueError("The method must be BorutaShap, PowerShap, Boruta or Shapicant.")
                
                for par in param:
                    if method == "BorutaShap":
                        arg = args['BorutaSHAP']
                    elif method == "PowerShap":
                        arg = args['PowerSHAP']
                    elif method == "Boruta":
                        arg = args['Boruta']
                    elif method == "Shapicant":
                        arg = args['Shapicant']
                    
                    for fit in arg:

                        start_time = time.time()
                        print()
                        print(method, model, par, fit)
                        print("------------------------------------------------------------------------------------------")
                        
                        if model.__class__.__name__ == "RandomForestClassifier":
                            mod = RandomForestClassifier(**model.get_params())
                        elif model.__class__.__name__ == "RandomForestRegressor":
                            mod = RandomForestRegressor(**model.get_params())
                        else:
                            mod = copy.deepcopy(model)

                        try:
                            if method == "BorutaShap":
                                fs = BorutaSHAP(data=X_res, target=y_res, model=mod, fit_kwargs=fit, **par)
                                accepted, rejected, tentative = fs.select_features(plot=False)
                            elif method == "PowerShap":
                                fs = PowerSHAP(data=X_res, target=y_res, model=mod, fit_kwargs=fit, **par)
                                accepted, rejected, tentative = fs.select_features()
                            elif method == "Boruta":
                                fs = Boruta(data=X_res, target=y_res, model=mod, **par)
                                accepted, rejected, tentative = fs.select_features()
                            elif method == "Shapicant":
                                fs = Shapicant(data=X_res, target=y_res, model=mod, fit_kwargs=fit, **par)
                                accepted, rejected, tentative = fs.select_features()
                        
                            elapsed_time = time.time() - start_time
                            
                            mod.fit(X_res, y_res)
                            y_pred = mod.predict(X_test)
                            bas = balanced_accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred)

                            X_res2 = X_res.drop(rejected, axis=1)
                            X_test2 = X_test.drop(rejected, axis=1)

                            mod.fit(X_res2, y_res)
                            y_pred = mod.predict(X_test2)
                            basR = balanced_accuracy_score(y_test, y_pred)
                            f1R = f1_score(y_test, y_pred)

                            print(bas, basR, f1, f1R)
                            
                            if 'catboostclassifier' in str(type(mod)).lower():
                                aux = mod.get_params()
                                aux2 = ', '.join([f'{key}={value}' for key, value in aux.items()])
                            
                                fila = [method, f"CatBoostClassifier({aux2})", par, fit, elapsed_time, bas, basR, f1, f1R, len(rejected), rejected]
                                escritor_csv.writerow(fila)

                            elif 'catboostregressor' in str(type(mod)).lower():
                                aux = mod.get_params()
                                aux2 = ', '.join([f'{key}={value}' for key, value in aux.items()])
                            
                                fila = [method, f"CatBoostRegressor({aux2})", par, fit, elapsed_time, bas, basR, f1, f1R, len(rejected), rejected]
                                escritor_csv.writerow(fila)

                            else:
                                fila = [method, model, par, fit, elapsed_time, bas, basR, f1, f1R, len(rejected), rejected]
                                escritor_csv.writerow(fila)

                        except Exception:
                            pass


def lanzadorTradicionales(data, target, methods, models, params):
    archivo = "resultadosTradicionales.csv"
    with open(archivo, mode='w', newline='') as archivo_csv:
        
        escritor_csv = csv.writer(archivo_csv)

        ens = Combine("SMOTETomek")
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
        X_res, y_res = ens.fit_resample(X_train, y_train)

        fila = ["Tecnica", "Metodo", "Modelo", "Parametros", "Tiempo", "Accuracy Todas", "Accuracy Aceptadas", "F1 score Todas", "F1 score Aceptadas", "len(rejected)", "rejected"]
        escritor_csv.writerow(fila)
        
        for tech in ["Chi2", "F1C", "F1R"]:
            for method in methods:
                if tech == "Chi2":
                    param = params['Chi2']
                elif tech == "F1C" or tech=="F1R":
                    param = params['F1']
                for model in models:
                    for fit in param:    
                        start_time = time.time()
                        print()
                        print(tech, method, model, fit)
                        print("------------------------------------------------------------------------------------------")

                        if model.__class__.__name__ == "RandomForestClassifier":
                            mod = RandomForestClassifier(**model.get_params())
                        elif model.__class__.__name__ == "RandomForestRegressor":
                            mod = RandomForestRegressor(**model.get_params())
                        else:
                            mod = copy.deepcopy(model)
                        
                        try:
                            if tech == "Chi2":
                                fs = Chi2(data=X_res, target=y_res)
                                accepted, rejected, tentative = fs.select_features(method=method, params=fit)
                            elif tech == "F1C":
                                fs = F1(data=X_res, target=y_res)
                                accepted, rejected, tentative = fs.select_features(method=method, params=fit, classification=True)
                            elif tech == "F1R":
                                fs = F1(data=X_res, target=y_res)
                                accepted, rejected, tentative = fs.select_features(method=method, params=fit, classification=False)
                            else:
                                raise ValueError("The method must be Chi2 or F1.")
                        
                            elapsed_time = time.time() - start_time

                            mod.fit(X_res, y_res)
                            y_pred = mod.predict(X_test)
                            bas = balanced_accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred)

                            X_res2 = X_res.drop(rejected, axis=1)
                            X_test2 = X_test.drop(rejected, axis=1)

                            mod.fit(X_res2, y_res)
                            y_pred = mod.predict(X_test2)
                            basR = balanced_accuracy_score(y_test, y_pred)
                            f1R = f1_score(y_test, y_pred)
                            
                            fila = [tech, method, model, fit, elapsed_time, bas, basR, f1, f1R, len(rejected), rejected]
                            escritor_csv.writerow(fila)
                        except Exception:
                            pass

"""
df = pd.read_csv('./data/BreastCancerWisconsin_(Diagnostic).csv', sep=',')

le = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

data = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
target = df["diagnosis"]
"""
df = pd.read_csv('./data/IT_customer_churn.csv', sep=',')
df = df.dropna()

le = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

data = df.drop(['Churn'], axis=1)
target = df["Churn"]



models = [RandomForestClassifier(n_estimators=10), RandomForestClassifier(n_estimators=50), RandomForestClassifier(n_estimators=100),
          CatBoostClassifier(n_estimators=150, verbose=0, use_best_model=False), CatBoostClassifier(n_estimators=250, verbose=0, use_best_model=False), CatBoostClassifier(n_estimators=350, verbose=0, use_best_model=False),
          CatBoostClassifier(n_estimators=150, verbose=0, use_best_model=True), CatBoostClassifier(n_estimators=250, verbose=0, use_best_model=True), CatBoostClassifier(n_estimators=350, verbose=0, use_best_model=True),
          DecisionTreeClassifier(), DecisionTreeClassifier(max_depth=1, max_leaf_nodes=3), DecisionTreeClassifier(max_depth=10, max_leaf_nodes=5), 
          XGBClassifier(random_state=42), LogisticRegression(max_iter=100), LogisticRegression(max_iter=1000)
         ]
"""

param_grid = {
    'n_estimators': np.arange(50, 1000, 50),
    'max_depth': np.arange(3, 10),
    'learning_rate': [0.1, 0.01, 0.001],
    'colsample_bytree': [0.5, 0.7, 0.9]
}

# Generar combinaciones de parámetros aleatorios
param_sampler = ParameterSampler(param_grid, n_iter=20, random_state=42)

# Crear una lista con 20 instancias del modelo con diferentes parámetros
models = [XGBClassifier(**params) for params in param_sampler]
"""
params = {"BorutaSHAP": [{}],
          "PowerSHAP": [{}],
          "Boruta": [{}],
          "Shapicant": [{}]
          }

args = {"BorutaSHAP": [{}],
          "PowerSHAP": [{}],
          "Boruta": [{}],
          "Shapicant": [{}]
          }

#lanzador(data, target, models, params, args)

metodosTradicional = [SelectFdr, SelectFpr, SelectFwe, SelectKBest, SelectPercentile, GenericUnivariateSelect]
paramsTradicional = {"Chi2": [{}], 
                     "F1": [{}]
                    }

lanzadorTradicionales(data, target, metodosTradicional, models, paramsTradicional)