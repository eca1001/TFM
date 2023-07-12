from borutaShap import BorutaShap
from powershap import PowerShap
from boruta import BorutaPy
from shapicant import PandasSelector
import shap
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod

from explainers import CatboostExplainer, EnsembleExplainer, LGBMExplainer, TreeExplainer, XGBoostExplainer, IMBCatboostExplainer, LinearExplainer, DeepLearningExplainer
from sklearn.feature_selection import chi2, f_classif, f_regression
from sklearn.preprocessing import LabelEncoder

import copy


class featureSelection(ABC):
    def __init__(self, data, target, model=None):
        self.X = data.copy()
        self.y = target.copy()
        self.model = copy.deepcopy(model)
        
    @abstractmethod
    def select_features(self):
        pass

    @abstractmethod
    def _print(self, accepted, rejected, tentative):
        print(str(len(accepted))  + ' attributes confirmed important: ' + str(accepted))
        print(str(len(rejected))  + ' attributes confirmed unimportant: ' + str(rejected))
        print(str(len(tentative))  + ' tentative attributes remanins: ' + str(tentative))



class BorutaSHAP(featureSelection):
    def __init__(self, data, target, model=None, importance_measure = 'Shap', classification = True, percentile = 100, pvalue = 0.05, **fit_kwargs):
        super().__init__(data, target, model)
        self.importance_measure = importance_measure
        self.classification = classification
        self.percentile = percentile
        self.pvalue = pvalue
        if fit_kwargs == {}:
            self.fit_kwargs = fit_kwargs
        else:
            self.fit_kwargs = fit_kwargs['fit_kwargs']
        
    def select_features(self, plot=True, tentative=False):
        if self.model is not None:
            modelo = self.model
            par = []
            for attr, value in self.model.__dict__.items():
                if str(attr) == "_init_params":
                    par=value
            
            if 'catboost' in str(type(modelo)).lower() and 'use_best_model' in par and par['use_best_model']==True:
                X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

                eval_set = (X_test, y_test)
                modelo = modelo.fit(X_train, y_train, eval_set=eval_set)
                Feature_Selector = BorutaShap(model=modelo, importance_measure=self.importance_measure,
                                              classification=self.classification, percentile=self.percentile, pvalue=self.pvalue)
                accepted, rejected, tentative = Feature_Selector.fit(X=self.X, y=self.y, **self.fit_kwargs, eval_set=eval_set) 
            else:
                Feature_Selector = BorutaShap(model=self.model, importance_measure=self.importance_measure,
                                              classification=self.classification, percentile=self.percentile, pvalue=self.pvalue)
                
                accepted, rejected, tentative = Feature_Selector.fit(X=self.X, y=self.y, **self.fit_kwargs)
        else:
            Feature_Selector = BorutaShap(importance_measure=self.importance_measure, classification=self.classification,
                                          percentile=self.percentile, pvalue=self.pvalue)
            accepted, rejected, tentative = Feature_Selector.fit(X=self.X, y=self.y, **self.fit_kwargs)

        if plot==True:
            Feature_Selector.plot(which_features='all')

        if tentative==True:
            accepted, rejected, tentative = Feature_Selector.TentativeRoughFix()

        return accepted, rejected, tentative

    def _print(self, accepted, rejected, tentative):
        return super()._print(accepted, rejected, tentative)

class PowerSHAP(featureSelection):
    def __init__(self, data, target, model=None, power_iterations = 10, power_alpha = 0.01, val_size = 0.2, power_req_iterations = 0.99, 
                 include_all = False, automatic = True, force_convergence = False, limit_convergence_its = 0, limit_automatic = 10, 
                 limit_incremental_iterations = 10, limit_recursive_automatic = 3, stratify = False, cv = None, show_progress = True, 
                 verbose = False, **fit_kwargs):
        
        super().__init__(data, target, model)
        self.power_iterations = power_iterations
        self.power_alpha = power_alpha
        self.val_size = val_size
        self.power_req_iterations = power_req_iterations
        self.include_all = include_all
        self.automatic = automatic
        self.force_convergence = force_convergence
        self.limit_convergence_its = limit_convergence_its
        self.limit_automatic = limit_automatic
        self.limit_incremental_iterations = limit_incremental_iterations
        self.limit_recursive_automatic = limit_recursive_automatic
        self.stratify = stratify
        self.show_progress = show_progress
        self.verbose = verbose
        if fit_kwargs == {}:
            self.fit_kwargs = fit_kwargs
        else:
            self.fit_kwargs = fit_kwargs['fit_kwargs']
        
    def select_features(self):
        if self.model is None:
            selector = PowerShap(power_iterations = self.power_iterations, power_alpha = self.power_alpha, val_size = self.val_size, 
                                 power_req_iterations = self.power_req_iterations, include_all = self.include_all, automatic = self.automatic, 
                                 force_convergence = self.force_convergence, limit_convergence_its = self.limit_convergence_its, limit_automatic = self.limit_automatic, 
                                 limit_incremental_iterations = self.limit_incremental_iterations, limit_recursive_automatic = self.limit_recursive_automatic, 
                                 stratify = self.stratify, show_progress = self.show_progress, verbose = self.verbose, fit_kwargs = self.fit_kwargs)
        else:
            selector = PowerShap(model=self.model, power_iterations = self.power_iterations, power_alpha = self.power_alpha, val_size = self.val_size, 
                                 power_req_iterations = self.power_req_iterations, include_all = self.include_all, automatic = self.automatic, 
                                 force_convergence = self.force_convergence, limit_convergence_its = self.limit_convergence_its, limit_automatic = self.limit_automatic, 
                                 limit_incremental_iterations = self.limit_incremental_iterations, limit_recursive_automatic = self.limit_recursive_automatic, 
                                 stratify = self.stratify, show_progress = self.show_progress, verbose = self.verbose, fit_kwargs = self.fit_kwargs)
        selector.fit(self.X, self.y, **self.fit_kwargs) 

        accepted = selector.transform(self.X).columns
        accepted = list(accepted)
        rejected = list(set(self.X.columns) - set(accepted))
        tentative = []

        self._print(accepted, rejected, tentative)

        return accepted, rejected, tentative
    
    def _print(self, accepted, rejected, tentative):
        super()._print(accepted, rejected, tentative)




class Boruta(featureSelection):
    def __init__(self, data, target, model=None, n_estimators = 100, perc = 100, alpha = 0.05, two_step = True, max_iter = 100, random_state = None, verbose = 0):
        super().__init__(data, target, model)
        self.n_estimators = n_estimators
        self.perc = perc
        self.alpha = alpha
        self.two_step = two_step
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        
    def select_features(self, plot=False):

        if self.model is None:
            print("No model has been specified. It will execute by default: RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)")
            selector = BorutaPy(estimator=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42), n_estimators = self.n_estimators, 
                                perc = self.perc, alpha = self.alpha, two_step = self.two_step, max_iter = self.max_iter, random_state = self.random_state, verbose = self.verbose)
        else:
            selector = BorutaPy(estimator=self.model, n_estimators = self.n_estimators, perc = self.perc, alpha = self.alpha, two_step = self.two_step, 
                                max_iter = self.max_iter, random_state = self.random_state, verbose = self.verbose)
        
        feat_selector = selector.fit(np.array(self.X), np.array(self.y))

        accepted = self.X.columns[feat_selector.support_].to_list()
        accepted = list(accepted)
        tentative = self.X.columns[feat_selector.support_weak_].to_list()
        tentative = list(tentative)
        rejected = list(set(self.X.columns) - set(accepted) - set(tentative))

        if plot:
            feature_ranks = list(zip(self.X.columns, 
                            feat_selector.ranking_, 
                            feat_selector.support_))

            for feat in feature_ranks:
                print('Feature: {:<25} Rank: {},  Keep: {}'.format(feat[0], feat[1], feat[2]))
        
        print(accepted)
        print(rejected)
        print(tentative)
        self._print(accepted, rejected, tentative)

        return accepted, rejected, tentative
    
    def _print(self, accepted, rejected, tentative):
        super()._print(accepted, rejected, tentative)
    



class Shapicant(featureSelection):
    def __init__(self, data, target, model=None, n_iter=100, verbose=1, random_state=None, **fit_kwargs):
        super().__init__(data, target, model)
        self.n_iter = n_iter
        self.verbose = verbose
        self.random_state = random_state
        if fit_kwargs == {}:
            self.fit_kwargs = fit_kwargs
        else:
            self.fit_kwargs = fit_kwargs['fit_kwargs']
        
    def select_features(self, alpha=0.2, plot=False):

        if self.model is None:
            print("No model has been specified. It will execute by default: RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)")
            explainer_type = ShapExplainerFactory.get_explainer(model=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42))
            selector = PandasSelector(estimator=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42), explainer_type=shap.TreeExplainer,
                                      n_iter=self.n_iter, verbose=self.verbose, random_state=self.random_state)
            fit_selector = selector.fit(pd.DataFrame(self.X), self.y, explainer_type_params=self.fit_kwargs)
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, random_state=0)

            if 'imbcatboost' in str(type(self.model)).lower():
                try:
                    self.model.fit(self.X, self.y, verbose=False)
                    
                except:
                    self.model.fit(self.X, self.y)
                
                explainer = ShapExplainerFactory.get_explainer(model=self.model)
                explainer_type, kwargs = explainer.select_explainer(X_train=X_train, Y_train=y_train, X_val=X_test, Y_val=y_test, with_params=False)

                selector = PandasSelector(estimator=self.model.base_clf, explainer_type=explainer_type,
                                        n_iter=self.n_iter, verbose=self.verbose, random_state=self.random_state)
                fit_selector = selector.fit(pd.DataFrame(self.X), self.y, explainer_type_params=kwargs, **self.fit_kwargs)
            else:

                explainer = ShapExplainerFactory.get_explainer(model=self.model)
                explainer_type, kwargs = explainer.select_explainer(X_train=X_train, Y_train=y_train, X_val=X_test, Y_val=y_test, with_params=False)

                selector = PandasSelector(estimator=self.model, explainer_type=explainer_type,
                                        n_iter=self.n_iter, verbose=self.verbose, random_state=self.random_state)
                fit_selector = selector.fit(pd.DataFrame(self.X), self.y, explainer_type_params=kwargs, **self.fit_kwargs)
            
        

        if plot:
            print(fit_selector.p_values_)

        accepted = fit_selector.get_features(alpha=alpha)
        tent = fit_selector.get_features(alpha=alpha+0.1)
        accepted = list(accepted)
        tentative = list(set(tent) - set(accepted))
        rejected = list(set(self.X.columns) - set(accepted) - set(tentative))
        
        self._print(accepted, rejected, tentative)

        return accepted, rejected, tentative
    
    def _print(self, accepted, rejected, tentative):
        super()._print(accepted, rejected, tentative)


class Chi2(featureSelection):
    def __init__(self, data, target):
        super().__init__(data, target)
    
    def select_features(self, method=None, params={}):
        label_encoder = LabelEncoder()
        X = self.X.copy()
        for col in self.X.columns:
            X[col] = label_encoder.fit_transform(self.X[col])
        
        y = label_encoder.fit_transform(self.y)

        if method is None:
            raise ValueError("A select method is neccesary.")
        else:
            chi = method(chi2, **params).fit(X,y)
        
        accepted = chi.get_feature_names_out()
        accepted = list(accepted)
        tentative = []
        rejected = list(set(self.X.columns) - set(accepted))
        
        self._print(accepted, rejected, tentative)

        return accepted, rejected, tentative
    
    def _print(self, accepted, rejected, tentative):
        super()._print(accepted, rejected, tentative)


class Fvalue(featureSelection):
    def __init__(self, data, target):
        super().__init__(data, target)

    def select_features(self, method=None, params={}, classification=True):
        
        label_encoder = LabelEncoder()
        X = self.X.copy()
        for col in self.X.columns:
            X[col] = label_encoder.fit_transform(self.X[col])
        
        y = label_encoder.fit_transform(self.y)

        if method is None:
            raise ValueError("A select method is neccesary.")
        else:
            if classification:
                f = method(f_classif, **params).fit(X,y)
            else:
                f = method(f_regression, **params).fit(X,y)
       
        accepted = f.get_feature_names_out()
        accepted = list(accepted)
        tentative = []
        rejected = list(set(self.X.columns) - set(accepted))
        
        self._print(accepted, rejected, tentative)

        return accepted, rejected, tentative

    def _print(self, accepted, rejected, tentative):
        super()._print(accepted, rejected, tentative)

class ShapExplainerFactory:
    
    _explainer_models = [
        CatboostExplainer,
        IMBCatboostExplainer,
        LGBMExplainer,
        XGBoostExplainer,
        EnsembleExplainer,
        TreeExplainer,
        LinearExplainer,
        DeepLearningExplainer,
    ]

    @classmethod
    def get_explainer(cls, model):
        for explainer_class in cls._explainer_models:
            try: 
                if explainer_class.supports_model(model):
                    return explainer_class(model)
            except Exception:
                pass
        raise ValueError(f"Given model ({model}) is not yet supported by our explainer models")