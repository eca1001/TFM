from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsembleClassifier, RUSBoostClassifier, BalancedRandomForestClassifier

class Ensamble:
    def __init__(self, method="BalancedBaggingClassifier", params={"random_state": 0}):
        if method == "BalancedBaggingClassifier":
            self.method = BalancedBaggingClassifier(**params)
        elif method == "EasyEnsembleClassifier":
            self.method = EasyEnsembleClassifier(**params)
        elif method == "RUSBoostClassifier":
            self.method = RUSBoostClassifier(**params)
        elif method == "BalancedRandomForestClassifier":
            self.method = BalancedRandomForestClassifier(**params)
        else:
            raise Exception("No se ha introducido un método válido")
    
    def fit(self, X_train, y_train):        
        self.method.fit(X_train, y_train)            

    def predict(self, X_test):
        y_pred = self.method.predict(X_test)
        return y_pred