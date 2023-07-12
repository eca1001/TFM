import time
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from featureSelection import BorutaSHAP, PowerSHAP, Boruta, Shapicant, Chi2, Fvalue

from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFdr, SelectFpr, SelectFwe

from imbalanced_classification.Combine import Combine
from sklearn.preprocessing import LabelEncoder
import csv
import copy

def lanzador(data, target, models, params, args):

    archivo = "resultados.csv"

    with open(archivo, mode='w', newline='') as archivo_csv:

        escritor_csv = csv.writer(archivo_csv)

        ens = Combine("SMOTETomek")
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
        X_res, y_res = ens.fit_resample(X_train, y_train)

        fila = ["Tecnica", "Modelo", "Parametros", "fit_kwargs", "Tiempo", "Accuracy Todas", "Accuracy Aceptadas", "F1 score Todas", "F1 score Aceptadas", "Nº de rechazadas", "Rechazadas"]
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

        fila = ["Tecnica", "Metodo", "Modelo", "Parametros", "Tiempo", "Accuracy Todas", "Accuracy Aceptadas", "F1 score Todas", "F1 score Aceptadas", "Nº de rechazadas", "Rechazadas"]
        escritor_csv.writerow(fila)
        
        for tech in ["Chi2", "FC", "FR"]:
            for method in methods:
                if tech == "Chi2":
                    param = params['Chi2']
                elif tech == "FC" or tech=="FR":
                    param = params['Fvalue']
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
                            elif tech == "FC":
                                fs = Fvalue(data=X_res, target=y_res)
                                accepted, rejected, tentative = fs.select_features(method=method, params=fit, classification=True)
                            elif tech == "FR":
                                fs = Fvalue(data=X_res, target=y_res)
                                accepted, rejected, tentative = fs.select_features(method=method, params=fit, classification=False)
                        
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
                            
                            
                            if 'catboostclassifier' in str(type(mod)).lower():
                                aux = mod.get_params()
                                aux2 = ', '.join([f'{key}={value}' for key, value in aux.items()])
                            
                                fila = [tech, method, f"CatBoostClassifier({aux2})", fit, elapsed_time, bas, basR, f1, f1R, len(rejected), rejected]
                                escritor_csv.writerow(fila)

                            elif 'catboostregressor' in str(type(mod)).lower():
                                aux = mod.get_params()
                                aux2 = ', '.join([f'{key}={value}' for key, value in aux.items()])
                            
                                fila = [tech, method, f"CatBoostRegressor({aux2})", fit, elapsed_time, bas, basR, f1, f1R, len(rejected), rejected]
                                escritor_csv.writerow(fila)

                            else:
                                fila = [tech, method, model, fit, elapsed_time, bas, basR, f1, f1R, len(rejected), rejected]
                                escritor_csv.writerow(fila)
                        except Exception:
                            pass
