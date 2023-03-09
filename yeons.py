import os
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import RobustScaler, StandardScaler, Normalizer, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from warnings import filterwarnings, simplefilter
import gc

from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_log_error

# Import necessary library
from sklearn.linear_model import Ridge, LinearRegression, ElasticNet
from sklearn.svm import SVR

# Custom modules
from modules.arguments import get_args
from modules.utils import set_all_seeds
from modules.dataset import get_data
from modules.regressor import CustomRegressor


def main(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    # Data load
    x, y, xtest = get_data(args)
    
    # Linear Regression
    lnr = LinearRegression(fit_intercept = True, n_jobs = -1, normalize = True)
    lnr.fit(x, y)
    yfit_lnr = pd.DataFrame(lnr.predict(x), index = x.index, columns = y.columns).clip(0.)
    ypred_lnr = pd.DataFrame(lnr.predict(xtest), index = xtest.index, columns = y.columns).clip(0.)

    # Multi Output Regression
    svr = MultiOutputRegressor(SVR(C = 0.2, kernel = 'rbf'), n_jobs = -1)
    svr.fit(x, y)
    yfit_svr = pd.DataFrame(svr.predict(x), index = x.index, columns = y.columns).clip(0.)
    ypred_svr = pd.DataFrame(svr.predict(xtest), index = xtest.index, columns = y.columns).clip(0.)

    yfit_mean = pd.DataFrame(np.mean([yfit_svr.values, yfit_lnr.values], axis = 0), index = x.index, columns = y.columns).clip(0.)

    y_ = y.stack(['store_nbr', 'family'])
    y_['lnr'] = yfit_lnr.stack(['store_nbr', 'family'])['sales']
    y_['svr'] = yfit_svr.stack(['store_nbr', 'family'])['sales']
    y_['mean'] = yfit_mean.stack(['store_nbr', 'family'])['sales']

    # Fit Progress
    ymean = yfit_lnr.append(ypred_lnr)
    school = ymean.loc(axis = 1)['sales', :, 'SCHOOL AND OFFICE SUPPLIES']
    ymean = ymean.join(school.shift(1), rsuffix = 'lag1') # I'm also adding school lag for it's cyclic yearly.
    x = x.loc['2017-05-01':]
    x = x.join(ymean) # Concating linear result
    xtest = xtest.join(ymean)

    y = y.loc['2017-05-01':]
    print(y.isna().sum().sum())

    model = CustomRegressor(n_jobs=-1, verbose=1, seed=args.seed)
    model.fit(x, y)

    # Predict Progress
    y_pred = pd.DataFrame(model.predict(x), index=x.index, columns=y.columns)
    print(y_pred.isna().sum().sum())
    y_pred = y_pred.stack(['store_nbr', 'family']).clip(0.)
    y_ = y.stack(['store_nbr', 'family']).clip(0.)

    y_['pred'] = y_pred.values
    print(y_.groupby('family').apply(lambda r : np.sqrt(np.sqrt(mean_squared_log_error(r['sales'], r['pred'])))))

    # Looking at error
    print('RMSLE : ', np.sqrt(np.sqrt(msle(y_['sales'], y_['pred'])))) # RMSLE 값이 약간 다름
    y_pred.isna().sum()

    ypred = pd.DataFrame(model.predict(xtest), index = xtest.index, columns = y.columns).clip(0.)
    ypred = ypred.stack(['store_nbr', 'family'])
    sub = pd.read_csv(f'{args.datapath}/sample_submission.csv')
    sub['sales'] = ypred.values
    sub.to_csv(f'{os.getcwd()}/result/submission{args.submission_name}_remake.csv', index = False) # Submit


if __name__ == "__main__":
    args = get_args()
    set_all_seeds(seed=args.seed)

    gc.enable()
    filterwarnings('ignore')
    simplefilter('ignore')

    main(args)