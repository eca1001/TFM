from borutaShap import BorutaShap
from powershap import PowerShap
import pandas as pd
from sklearn.model_selection import train_test_split


class featureSelection:
    def __init__(self, data, target, method="PowerShap", params={}, model=None, kwargs={"verbose": 0}):
        self.X = pd.DataFrame(data)
        self.y = pd.Series(target)
        self.model = model
        self.kwargs = kwargs
        self.params = params
        
        if method == "BorutaShap":
            self.method = BorutaShap

        elif method == "PowerShap":
            self.method = PowerShap

        else:
            raise ValueError("The methods must be BorutaShap or PowerShap.")

    
    def borutashap(self, args={}, plot=True, tentative=False):
        if self.method is not BorutaShap:
            raise ValueError("The method must be BorutaShap.")
        
        if self.model is not None:
            modelo = self.model(**self.kwargs)

            if 'catboost' in str(type(modelo)).lower() and 'use_best_model' in self.kwargs and self.kwargs['use_best_model']==True:
                X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)

                eval_set = (X_test, y_test)
                modelo = modelo.fit(X_train, y_train, eval_set=eval_set)
                Feature_Selector = self.method(model=modelo, **self.params)
                accepted, rejected, tentative = Feature_Selector.fit(X=self.X, y=self.y, **args, eval_set=eval_set) 
            else:
                Feature_Selector = self.method(model=self.model(**self.kwargs), **self.params)
                accepted, rejected, tentative = Feature_Selector.fit(X=self.X, y=self.y, **args)
        else:
            Feature_Selector = self.method(**self.params)
            accepted, rejected, tentative = Feature_Selector.fit(X=self.X, y=self.y, **args)

        if plot:
            Feature_Selector.plot(which_features='all')

        if tentative:
            accepted, rejected, tentative = Feature_Selector.TentativeRoughFix()

        return accepted, rejected, tentative


    def powershap(self, args={}):
        if self.method is not PowerShap:
            raise ValueError("The method must be PowerShap.")
        
        if self.model is None:
            selector = self.method(**self.params)
        else:
            selector = self.method(model=self.model(**self.kwargs), **self.params)
        selector.fit(self.X, self.y, **args) 

        accepted = selector.transform(self.X).columns
        accepted = list(accepted)
        rejected = list(set(self.X.columns) - set(accepted))
        print(str(len(accepted))  + ' attributes confirmed important: ' + str(accepted))
        print(str(len(rejected))  + ' attributes confirmed unimportant: ' + str(rejected))
        tentative=[]
        print(str(len(tentative))  + ' tentative attributes remanins: ' + str(tentative))

        return accepted, rejected, tentative

    #def fit():

    #def predict():

    #def combine(): #borutapowershap: union de ambas en 1, es decir, coger las accepted comunes de las 2, predecir su resultado, e ir añadiendo variables accepted no comunes y si mejora se añade