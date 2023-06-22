from imblearn.combine import SMOTEENN, SMOTETomek

class Combine:
    def __init__(self, method="SMOTEENN", params={}):
        if method == "SMOTEENN":
            self.method = SMOTEENN(**params)
        elif method == "SMOTETomek":
            self.method = SMOTETomek(**params)
        else:
            raise Exception("No se ha introducido un método válido")

    def fit_resample(self, X, y):
        X_resampled, y_resampled = self.method.fit_resample(X, y)
        return X_resampled, y_resampled