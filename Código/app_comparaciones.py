import time
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
from featureSelection import featureSelection
from catboost import CatBoostClassifier, CatBoostRegressor

import time
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, mean_squared_error, r2_score

def lanzador(data, target, models, params, args, kwargs):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
    
    resultsC = []
    resultsR = []
    
    for i in range(len(models)):
        result = {}
        
        for method in ["BorutaShap", "PowerShap"]:
            start_time = time.time()
            
            print(method, models[i])

            if method == "BorutaShap":
                fs = featureSelection(data=X_train, target=y_train, method=method, model=models[i], params=params[i], kwargs=kwargs[i])
                accepted, rejected, tentative = fs.borutashap(plot=False, args=args[i])
            elif method == "PowerShap":
                fs = featureSelection(data=X_train, target=y_train, method=method, model=models[i], params=params[i])
                accepted, rejected, tentative = fs.powershap()
            else:
                raise ValueError("The method must be BorutaShap or PowerShap.")
            
            elapsed_time = time.time() - start_time
            
            eval_set = (X_test, y_test)
            model = models[i](**kwargs[i])
            if 'catboost' in str(model).lower() and 'use_best_model' in kwargs[i] and kwargs[i]['use_best_model']==True:
                model.fit(X_train, y_train, eval_set=eval_set)
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(data=X_test)

            if params[i]['classification']==True:
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                result[method] = {
                "Accepted": accepted,
                "Rejected": rejected,
                "Tentative": tentative,
                "Accuracy": accuracy,
                "F1 Score": f1,
                "Time": elapsed_time
                }
                resultsC.append(result)
            
            else:
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                r2 = r2_score(y_test, y_pred)
                result[method] = {
                "Accepted": accepted,
                "Rejected": rejected,
                "Tentative": tentative,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "Time": elapsed_time
                }
                resultsR.append(result)
            
    return resultsC, resultsR



df = pd.read_csv('./data/data.csv')

data = df.drop(['liked'], axis=1)
target = df['liked']

params = [{"classification": True}, {"classification": False}]
models = [CatBoostClassifier, CatBoostRegressor]
kwargs = [{"n_estimators": 250, "verbose": 0, "use_best_model": False}, {"n_estimators": 250, "verbose": 0, "use_best_model": False}] 
args = [{"n_trials": 100, "sample": False, "train_or_test": 'test', "normalize": True, "verbose": True}, {"n_trials": 100, "sample": False, "train_or_test": 'test', "normalize": True, "verbose": True}]

resultsC, resultsR = lanzador(data, target, models, params, args, kwargs)

print(resultsC)
print(resultsR)