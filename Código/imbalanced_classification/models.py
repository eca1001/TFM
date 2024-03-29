from catboost import CatBoostClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

class IMBCatboostClassifier:
    def __init__(self, params={}):
        self.params = params

    def fit(self, X_train, y_train, cv=None):
        weight_class = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

        self.base_clf = CatBoostClassifier(verbose=False, class_weights=weight_class, **self.params)
        self.base_clf.fit(X_train, y_train)
        
        self.calibrated_clf = CalibratedClassifierCV(self.base_clf, cv=cv, method='isotonic')
        self.calibrated_clf.fit(X_train, y_train)
        
        return weight_class

    def predict(self, X_test):
        return self.calibrated_clf.predict(X_test)