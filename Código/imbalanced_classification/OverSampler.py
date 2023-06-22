from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN

class OverSampler: 
    def __init__(self, method="SMOTE", params={}):
        if method == "SMOTE":
            self.method = SMOTE(**params)
        elif method == "ADASYN":
            self.method = ADASYN(**params)
        elif method == "Random":
            self.method = RandomOverSampler(**params)
        else:
            raise Exception("No se ha introducido un método válido")

    def fit_resample(self, X, y):
        X_resampled, y_resampled = self.method.fit_resample(X, y)
        return X_resampled, y_resampled