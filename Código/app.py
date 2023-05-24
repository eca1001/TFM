import pandas as pd
from featureSelection import featureSelection
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.svm import SVC, LinearSVC

#load_boston (regression)
"""
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
feature_names = np.array(["CRIM", "ZN", "INDUS",  "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]) # nombre columna target "MEDV"
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

X = pd.DataFrame(data, columns=feature_names)
y = pd.Series(target, name='MEDV')
X.head()
"""
#load_breast_cancer (classification)
"""
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'], ['target']))
y = X.pop('target')
X.head()
"""

df = pd.read_csv('./data/data.csv')

data = df.drop(['liked'], axis=1)
target = df['liked']

fs = featureSelection(data, target, "BorutaShap")
accepted, rejected, tentative = fs.borutashap()


"""

df = pd.read_csv('./data/data.csv')

data = df.drop(['liked'], axis=1)
target = df['liked']

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
obj = LinearSVC()
obj.fit(X_train, y_train)
y_pred = obj.predict(X_test)
print(f1_score(y_pred, y_test))

print("-----")

df = pd.read_csv('./data/data.csv')

data = df.drop(['liked'], axis=1)
target = df['liked']

fs = featureSelection(data, target, "BorutaShap", params={"classification": False}, model=CatBoostRegressor, kwargs={"n_estimators": 250, "verbose": 0, "use_best_model": True})
accepted, rejected, tentative = fs.borutashap(args={"n_trials": 100, "sample": False, "train_or_test": 'test', "normalize": True, "verbose": True}, 
              plot=True, tentative=False)

data = df.drop(['liked'], axis=1)
data = df.drop(rejected, axis=1)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
obj = LinearSVC()
obj.fit(X_train, y_train)
y_pred = obj.predict(X_test)
print(f1_score(y_pred, y_test))

print("-----")

df = pd.read_csv('./data/data.csv')

data = df.drop(['liked'], axis=1)
target = df['liked']

fs = featureSelection(data, target, "BorutaShap", params={"classification": True}, model=CatBoostClassifier)
fs.borutashap(args={"n_trials": 100, "sample": False, "train_or_test": 'test', "normalize": True, "verbose": True},
              plot=False, tentative=True)

fs = featureSelection(data, target, "PowerShap", model=CatBoostClassifier, kwargs={"n_estimators": 250, "verbose": 0, "use_best_model": True})
accepted, rejected, tentative = fs.powershap(args={"n_trials": 100, "sample": False, "train_or_test": 'test', "normalize": True, "verbose": True})

data = df.drop(['liked'], axis=1)
data = df.drop(rejected, axis=1)

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=42)
obj = LinearSVC()
obj.fit(X_train, y_train)
y_pred = obj.predict(X_test)
print(f1_score(y_pred, y_test))
"""