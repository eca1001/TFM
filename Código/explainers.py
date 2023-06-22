from abc import ABC
from copy import copy
from typing import Any, Callable

import numpy as np
import shap


class ShapExplainer(ABC):

    def __init__(self, model: Any):
        assert self.supports_model(model)
        self.model = model
        

    # Should be implemented by subclass
    def select_explainer(self):
        raise NotImplementedError

    # Should be implemented by subclass
    @staticmethod
    def supports_model(model):
        raise NotImplementedError

### CATBOOST
class CatboostExplainer(ShapExplainer):
    @staticmethod
    def supports_model(model) -> bool:
        from catboost import CatBoostClassifier, CatBoostRegressor
        
        supported_models = [CatBoostRegressor, CatBoostClassifier]
        return isinstance(model, tuple(supported_models))

    def select_explainer(self, with_params):
        if with_params:
            return shap.TreeExplainer(self.model, feature_perturbation = "tree_path_dependent")
        else:
            return shap.TreeExplainer, {'feature_perturbation': "tree_path_dependent"}


### LGBM
class LGBMExplainer(ShapExplainer):
    @staticmethod
    def supports_model(model) -> bool:
        from lightgbm import LGBMClassifier, LGBMRegressor

        supported_models = [LGBMClassifier, LGBMRegressor]
        return isinstance(model, tuple(supported_models))

    def select_explainer(self, with_params):
        if with_params:
            return shap.TreeExplainer(self.model, feature_perturbation = "tree_path_dependent")
        else:
            return shap.TreeExplainer, {'feature_perturbation': "tree_path_dependent"}


### XGBOOST
class XGBoostExplainer(ShapExplainer):
    @staticmethod
    def supports_model(model) -> bool:
        from xgboost import XGBClassifier, XGBRegressor

        supported_models = [XGBClassifier, XGBRegressor]
        return isinstance(model, tuple(supported_models))

    def select_explainer(self, with_params):
        if with_params:
            return shap.TreeExplainer(self.model, feature_perturbation = "tree_path_dependent")
        else:
            return shap.TreeExplainer, {'feature_perturbation': "tree_path_dependent"}


### RANDOMFOREST
class EnsembleExplainer(ShapExplainer):
    @staticmethod
    def supports_model(model) -> bool:
        from sklearn.ensemble._forest import ForestClassifier, ForestRegressor
        from sklearn.ensemble._gb import BaseGradientBoosting

        supported_models = [ForestRegressor, ForestClassifier, BaseGradientBoosting]
        return issubclass(type(model), tuple(supported_models))

    def select_explainer(self, with_params):
        if with_params:
            return shap.TreeExplainer(self.model, feature_perturbation = "tree_path_dependent")
        else:
            return shap.TreeExplainer, {'feature_perturbation': "tree_path_dependent"}
        

### TREE
class TreeExplainer(ShapExplainer):
    @staticmethod
    def supports_model(model) -> bool:
        from sklearn.tree import BaseDecisionTree, DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor

        supported_models = [BaseDecisionTree, DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier, ExtraTreeRegressor]
        return issubclass(type(model), tuple(supported_models))

    def select_explainer(self, with_params):
        if with_params:
            return shap.TreeExplainer(self.model, feature_perturbation = "tree_path_dependent")
        else:
            return shap.TreeExplainer, {'feature_perturbation': "tree_path_dependent"}


### LINEAR
class LinearExplainer(ShapExplainer):
    @staticmethod
    def supports_model(model) -> bool:
        from sklearn.linear_model._base import LinearClassifierMixin, LinearModel
        from sklearn.linear_model._stochastic_gradient import BaseSGD

        supported_models = [LinearClassifierMixin, LinearModel, BaseSGD]
        return issubclass(type(model), tuple(supported_models))

    def select_explainer(self, with_params):
        from sklearn.base import clone

        if with_params:
            modelo = clone(self.model)
            modelo.fit(X, y)
            return shap.explainers.Linear(model=modelo, masker=X)
        else:
            return shap.explainers.Linear, {'masker': X}


### DEEP LEARNING
class DeepLearningExplainer(ShapExplainer):
    @staticmethod
    def supports_model(model) -> bool:
        import tensorflow as tf 

        supported_models = [tf.keras.Model] 
        return isinstance(model, tuple(supported_models))

    def select_explainer(self, X_train, Y_train, X_val, Y_val, random_seed, **kwargs) -> np.array:
        import tensorflow as tf

        tf.compat.v1.disable_v2_behavior()

        # Fit the model
        PowerShap_model = tf.keras.models.clone_model(self.model)
        metrics = kwargs.get("nn_metric")
        PowerShap_model.compile(
            loss=kwargs["loss"],
            optimizer=kwargs["optimizer"],
            metrics=metrics if metrics is None else [metrics],
        )
        _ = PowerShap_model.fit(
            X_train,
            Y_train,
            batch_size=kwargs["batch_size"],
            epochs=kwargs["epochs"],
            validation_data=(X_val, Y_val),
            verbose=False,
        )
        # Calculate the shap values
        C_explainer = shap.DeepExplainer(PowerShap_model, X_train)
        return C_explainer.shap_values(X_val)
