import time
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
from featureSelection import BorutaSHAP, PowerSHAP, Boruta, Shapicant, Chi2, F1
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression

import time

from sklearn.feature_selection import SelectFdr, SelectFpr, SelectFromModel, SelectFwe, SelectKBest, SelectorMixin, SelectPercentile, SequentialFeatureSelector, GenericUnivariateSelect, RFE, RFECV, VarianceThreshold

from imbalanced_classification.Ensamble import Ensamble
from sklearn.preprocessing import LabelEncoder
import csv

def lanzador(data, target, models, params, args):
    archivo = "resultados.csv"

    with open(archivo, mode='w', newline='') as archivo_csv:

        escritor_csv = csv.writer(archivo_csv)

        fila = ["Metodo", "Modelo", "Parametros", "fit_kwargs", "Tiempo", "Accuracy Todas", "Accuracy Aceptadas", "F1 score Todas", "F1 score Aceptadas"]
        escritor_csv.writerow(fila)
        for method in ["Boruta", "Shapicant", "BorutaShap", "PowerShap"]:
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
                            mod = model

                        try:
                            if method == "BorutaShap":
                                fs = BorutaSHAP(data=data, target=target, model=mod, fit_kwargs=fit, **par)
                                accepted, rejected, tentative = fs.select_features(plot=False)
                            elif method == "PowerShap":
                                fs = PowerSHAP(data=data, target=target, model=mod, fit_kwargs=fit, **par)
                                accepted, rejected, tentative = fs.select_features()
                            elif method == "Boruta":
                                fs = Boruta(data=data, target=target, model=mod, **par)
                                accepted, rejected, tentative = fs.select_features()
                            elif method == "Shapicant":
                                fs = Shapicant(data=data, target=target, model=mod, fit_kwargs=fit, **par)
                                accepted, rejected, tentative = fs.select_features()
                        
                            elapsed_time = time.time() - start_time

                            X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

                            ens = Ensamble("BalancedBaggingClassifier", {"random_state": 0, "sampling_strategy": "auto"})
                            ens.fit(X_train, y_train)
                            y_pred = ens.predict(X_test)
                            bas = balanced_accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred)
                            #print(classification_report(y_test, y_pred))
                            #print(confusion_matrix(y_test, y_pred))

                            data2 = data.drop(rejected, axis=1)
                            data2 = data2.drop(tentative, axis=1)

                            X_train, X_test, y_train, y_test = train_test_split(data2, target, test_size=0.3, random_state=42)

                            ens = Ensamble("BalancedBaggingClassifier", {"random_state": 0, "sampling_strategy": "auto"})
                            ens.fit(X_train, y_train)
                            y_pred = ens.predict(X_test)
                            basR = balanced_accuracy_score(y_test, y_pred)
                            f1R = f1_score(y_test, y_pred)
                            #print(classification_report(y_test, y_pred))
                            #print(confusion_matrix(y_test, y_pred))
                            
                            fila = [method, model, par, fit, elapsed_time, bas, basR, f1, f1R]
                            escritor_csv.writerow(fila)
                        except Exception:
                            fila = [method, model, par, fit, None, None, None, None, None]
                            escritor_csv.writerow(fila)
            
def lanzadorTradicionales(data, target, methods, params):
    
    for i in range(len(methods)):
        classification = True
        for tech in ["Chi2", "F1", "F1"]:

            start_time = time.time()
            print()
            print(tech, i, methods[i])
            print("------------------------------------------------------------------------------------------")
            
            try:
                model = methods[i].copy()
            except Exception:
                model = methods[i]

            try:
                if tech == "Chi2":
                    fs = Chi2(data=data, target=target)
                    accepted, rejected, tentative = fs.select_features(method=model, params=params[i])
                elif tech == "F1":
                    if classification == True:
                        print("classification = True")
                        fs = F1(data=data, target=target)
                        accepted, rejected, tentative = fs.select_features(method=model, params=params[i], classification=True)
                        classification = False
                    else:
                        print("classification = False")
                        fs = F1(data=data, target=target)
                        accepted, rejected, tentative = fs.select_features(method=model, params=params[i], classification=False)
                else:
                    raise ValueError("The method must be Chi2 or F1.")
            
                elapsed_time = time.time() - start_time

                X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

                ens = Ensamble("BalancedBaggingClassifier", {"random_state": 0, "sampling_strategy": "auto"})
                ens.fit(X_train, y_train)
                y_pred = ens.predict(X_test)
                print(classification_report(y_test, y_pred))
                print(confusion_matrix(y_test, y_pred))

                data2 = data.drop(rejected, axis=1)
                data2 = data2.drop(tentative, axis=1)

                X_train, X_test, y_train, y_test = train_test_split(data2, target, test_size=0.3, random_state=42)

                ens = Ensamble("BalancedBaggingClassifier", {"random_state": 0, "sampling_strategy": "auto"})
                ens.fit(X_train, y_train)
                y_pred = ens.predict(X_test)
                print(classification_report(y_test, y_pred))
                print(confusion_matrix(y_test, y_pred))
            except Exception:
                pass

"""
df = pd.read_csv('./data/data.csv')

data = df.drop(['liked'], axis=1)
target = df['liked']
"""

df = pd.read_csv('./data/BreastCancerWisconsin_(Diagnostic).csv', sep=',')

le = LabelEncoder()

for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

data = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)
target = df["diagnosis"]


models = [RandomForestClassifier(n_estimators=10), RandomForestClassifier(n_estimators=50), RandomForestClassifier(n_estimators=100),
          RandomForestRegressor(n_estimators=10), RandomForestRegressor(n_estimators=50), RandomForestRegressor(n_estimators=100), 
          CatBoostClassifier(n_estimators=150, verbose=0, use_best_model=False), CatBoostClassifier(n_estimators=250, verbose=0, use_best_model=False), CatBoostClassifier(n_estimators=350, verbose=0, use_best_model=False),
          CatBoostClassifier(n_estimators=150, verbose=0, use_best_model=True), CatBoostClassifier(n_estimators=250, verbose=0, use_best_model=True), CatBoostClassifier(n_estimators=350, verbose=0, use_best_model=True),
          CatBoostRegressor(n_estimators=150, verbose=0, use_best_model=False), CatBoostRegressor(n_estimators=250, verbose=0, use_best_model=False), CatBoostRegressor(n_estimators=350, verbose=0, use_best_model=False),
          CatBoostRegressor(n_estimators=150, verbose=0, use_best_model=True), CatBoostRegressor(n_estimators=250, verbose=0, use_best_model=True), CatBoostRegressor(n_estimators=350, verbose=0, use_best_model=True),
          DecisionTreeClassifier(), DecisionTreeClassifier(max_depth=1, max_leaf_nodes=3), DecisionTreeClassifier(max_depth=10, max_leaf_nodes=5), 
          DecisionTreeRegressor(), DecisionTreeRegressor(max_depth=1, max_leaf_nodes=3), DecisionTreeRegressor(max_depth=10, max_leaf_nodes=5),
          XGBClassifier(), XGBRegressor(),
          LogisticRegression(max_iter=100), LogisticRegression(max_iter=1000)
         ] 

params = {"BorutaSHAP": [{'n_trials': 50, 'sample': False, 'train_or_test': 'test', 'normalize': True, 'verbose': True}, {'n_trials': 100, 'sample': False, 'train_or_test': 'test', 'normalize': True, 'verbose': True},
                         {'n_trials': 50, 'sample': False, 'train_or_test': 'train', 'normalize': True, 'verbose': True}, {'n_trials': 100, 'sample': False, 'train_or_test': 'train', 'normalize': True, 'verbose': True}],
          "PowerSHAP": [{}],
          "Boruta": [{}],
          "Shapicant": [{}]
          }

args = {"BorutaSHAP": [{}],
          "PowerSHAP": [{}],
          "Boruta": [{}],
          "Shapicant": [{}]
          }

lanzador(data, target, models, params, args)

metodosTradicional = [SelectFdr, SelectFpr, SelectFwe, SelectKBest, SelectPercentile, GenericUnivariateSelect]
paramsTradicional = [{}] #{"Chi2": [{}], "F1": [{}]}

#lanzadorTradicionales(data, target, metodosTradicional, paramsTradicional)