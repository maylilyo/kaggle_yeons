from joblib import Parallel, delayed
import warnings
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
import numpy as np

class CustomRegressor():
        def __init__(self, n_jobs=-1, verbose=0, seed=42):
            
            self.n_jobs = n_jobs
            self.verbose = verbose
            self.estimators_ = None
            self.seed = seed
            
        def _estimator_(self, X, y):
        
            warnings.simplefilter(action='ignore', category=FutureWarning)
            
            if y.name[2] == 'SCHOOL AND OFFICE SUPPLIES': # Because SCHOOL AND OFFICE SUPPLIES has weird trend, we use decision tree instead.
                r1 = ExtraTreesRegressor(n_estimators = 200, n_jobs=-1, max_depth=10, random_state=self.seed)
                r2 = RandomForestRegressor(n_estimators = 200, n_jobs=-1, random_state=self.seed)
                b1 = BaggingRegressor(base_estimator=r1,
                                    n_estimators=10,
                                    n_jobs=-1,
                                    random_state=self.seed)
                b2 = BaggingRegressor(base_estimator=r2,
                                    n_estimators=10,
                                    n_jobs=-1,
                                    random_state=self.seed)
                model = VotingRegressor([('et', b1), ('rf', b2)]) # Averaging the result
            else:
                ridge = Ridge(fit_intercept=True, solver='auto', alpha=0.75, normalize=True, random_state=self.seed)
                svr = SVR(C = 0.2, kernel = 'rbf')
                
                model = VotingRegressor([('ridge', ridge), ('svr', svr)]) # Averaging result
            model.fit(X, y)
            return model

        def fit(self, X, y):
            from tqdm.auto import tqdm
            
            
            if self.verbose == 0 :
                self.estimators_ = Parallel(n_jobs=self.n_jobs, 
                                    verbose=0,
                                    )(delayed(self._estimator_)(X, y.iloc[:, i]) for i in range(y.shape[1]))
            else :
                print('Fit Progress')
                self.estimators_ = Parallel(n_jobs=self.n_jobs, 
                                    verbose=0,
                                    )(delayed(self._estimator_)(X, y.iloc[:, i]) for i in tqdm(range(y.shape[1])))
            return
        
        def predict(self, X):
            from tqdm.auto import tqdm
            if self.verbose == 0 :
                y_pred = Parallel(n_jobs=self.n_jobs, 
                                verbose=0)(delayed(e.predict)(X) for e in self.estimators_)
            else :
                print('Predict Progress')
                y_pred = Parallel(n_jobs=self.n_jobs, 
                                verbose=0)(delayed(e.predict)(X) for e in tqdm(self.estimators_))
            
            return np.stack(y_pred, axis=1)