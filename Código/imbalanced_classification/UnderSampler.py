from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours

class UnderSampler:
    def __init__(self, method="Tomek Links", params={}):
        if method == "Tomek Links":
            self.method = TomekLinks(**params)
        elif method == "ENN":
            self.method = EditedNearestNeighbours(**params)
        elif method == "Random":
            self.method = RandomUnderSampler(**params)
        else:
            raise Exception("No se ha introducido un método válido")

    def fit_resample(self, X, y):
        X_resampled, y_resampled = self.method.fit_resample(X, y)
        return X_resampled, y_resampled