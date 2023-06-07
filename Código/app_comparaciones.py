import time
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from featureSelection import featureSelection, BorutaSHAP, PowerSHAP, Boruta, Shapicant
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression

import time
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, mean_squared_error, r2_score

def lanzador(data, target, models, params, args):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)

    resultsC = []
    resultsR = []
    
    for i in range(len(models)):
        for method in ["BorutaShap", "PowerShap", "Boruta", "Shapicant"]:
            result = {}
            start_time = time.time()
            print()
            print(method, i, models, models[i])
            print("------------------------------------------------------------------------------------------")

            if method == "BorutaShap":
                fs = BorutaSHAP(data=X_train, target=y_train, model=models[i].copy(), params=params[i], args=args[i])
                accepted, rejected, tentative = fs.select_features(plot=False)
            elif method == "PowerShap":
                fs = PowerSHAP(data=X_train, target=y_train, model=models[i].copy(), params=params[i], args=args[i])
                accepted, rejected, tentative = fs.select_features()
            elif method == "Boruta":
                fs = Boruta(data=X_train, target=y_train, model=models[i].copy(), params=params[i], args=args[i])
                accepted, rejected, tentative = fs.select_features()
            elif method == "Shapicant":
                fs = Shapicant(data=X_train, target=y_train, model=models[i].copy(), params=params[i], args=args[i])
                accepted, rejected, tentative = fs.select_features()
            else:
                raise ValueError("The method must be BorutaShap, PowerShap, Boruta or Shapicant.")
            
            elapsed_time = time.time() - start_time
            
            eval_set = (X_test, y_test)
            model = models[i].copy()

            par=[]
            for attr, value in model.__dict__.items():
                if str(attr) == "_init_params":
                    par=value

            if 'catboost' in str(model).lower() and 'use_best_model' in par and par['use_best_model']==True:
                model.fit(X_train, y_train, eval_set=eval_set)
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(data=X_test)

            
            if 'classifier' in str(type(model)).lower():
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                result[method] = {"Accepted": accepted, "Rejected": rejected, "Tentative": tentative, "Accuracy": accuracy, "F1 Score": f1, "Time": elapsed_time}
                resultsC.append(result)
            
            else:
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                r2 = r2_score(y_test, y_pred)
                result[method] = {"Accepted": accepted, "Rejected": rejected, "Tentative": tentative, "mae": mae, "mse": mse, "rmse": rmse, "r2": r2, "Time": elapsed_time}
                resultsR.append(result)
            
    return resultsC, resultsR



df = pd.read_csv('./data/data.csv')

data = df.drop(['liked'], axis=1)
target = df['liked']

params = [{}, {}, {},{},{},{},{}]
models = [CatBoostClassifier(n_estimators=250, verbose=0, use_best_model=False), CatBoostRegressor(n_estimators=250, verbose=0, use_best_model=False), RandomForestClassifier(), RandomForestRegressor(), DecisionTreeClassifier(), DecisionTreeRegressor(), LogisticRegression()], 
args = [{"n_trials": 100, "sample": False, "train_or_test": 'test', "normalize": True, "verbose": True}, {"n_trials": 100, "sample": False, "train_or_test": 'test', "normalize": True, "verbose": True}, {},{},{},{},{}]

resultsC, resultsR = lanzador(data, target, models, params, args)

print()
print(resultsC)
print(resultsR)