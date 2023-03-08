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

# Visualization library(Not use currently)
from matplotlib import pyplot as plt, rcParams, style
import seaborn as sns
from plotly import express as px, graph_objects as go


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
    ypred_mean = pd.DataFrame(np.mean([ypred_lnr.values, ypred_svr.values], axis = 0), index = xtest.index, columns = y.columns).clip(0.)

    y_ = y.stack(['store_nbr', 'family'])
    y_['lnr'] = yfit_lnr.stack(['store_nbr', 'family'])['sales']
    y_['svr'] = yfit_svr.stack(['store_nbr', 'family'])['sales']
    y_['mean'] = yfit_mean.stack(['store_nbr', 'family'])['sales']

    print('='*70, 'Linear Regression', '='*70)
    print(y_.groupby('family').apply(lambda r : np.sqrt(msle(r['sales'], r['lnr']))))
    print('LNR RMSLE :', np.sqrt(msle(y, yfit_lnr)))
    print('='*70, 'SVR', '='*70)
    print(y_.groupby('family').apply(lambda r : np.sqrt(msle(r['sales'], r['svr']))))
    print('SVR RMSLE :', np.sqrt(msle(y, yfit_svr)))
    print('='*70, 'Mean', '='*70)
    print(y_.groupby('family').apply(lambda r : np.sqrt(msle(r['sales'], r['mean']))))
    print('Mean RMSLE :', np.sqrt(msle(y, yfit_mean)))


    #
    print('='*70, 'Linear Regression', '='*70)
    print(y_.groupby('family').apply(lambda r : mae(r['sales'], r['lnr'])))
    print('LNR MAE :', mae(y, yfit_lnr))
    print('='*70, 'SVR', '='*70)
    print(y_.groupby('family').apply(lambda r : mae(r['sales'], r['svr'])))
    print('SVR MAE :', mae(y, yfit_svr))
    print('='*70, 'Mean', '='*70)
    print(y_.groupby('family').apply(lambda r : mae(r['sales'], r['mean'])))
    print('Mean MAE :', mae(y, yfit_mean))

    #
    true_low = [2]
    pred_low = [4]

    print('RMSLE for low value :', np.sqrt(msle(true_low, pred_low)))
    print('MAE for low value :', mae(true_low, pred_low))

    true_high = [255]
    pred_high = [269]

    print('RMSLE for high value :', np.sqrt(msle(true_high, pred_high)))
    print('MAE for high value :', mae(true_high, pred_high))

    #
    ymean = yfit_lnr.append(ypred_lnr)
    school = ymean.loc(axis = 1)['sales', :, 'SCHOOL AND OFFICE SUPPLIES']
    ymean = ymean.join(school.shift(1), rsuffix = 'lag1') # I'm also adding school lag for it's cyclic yearly.
    x = x.loc['2017-05-01':]
    x = x.join(ymean) # Concating linear result
    xtest = xtest.join(ymean)

    y = y.loc['2017-05-01':]

    model = CustomRegressor(n_jobs=-1, verbose=1, seed=args.seed)
    model.fit(x, y)



    y_pred = pd.DataFrame(model.predict(x), index=x.index, columns=y.columns)
    y_pred = y_pred.stack(['store_nbr', 'family']).clip(0.)
    y_ = y.stack(['store_nbr', 'family']).clip(0.)

    y_['pred'] = y_pred.values
    print(y_.groupby('family').apply(lambda r : np.sqrt(np.sqrt(mean_squared_log_error(r['sales'], r['pred'])))))

    # Looking at error
    print('RMSLE : ', np.sqrt(np.sqrt(msle(y_['sales'], y_['pred']))))
    y_pred.isna().sum()

    ypred = pd.DataFrame(model.predict(xtest), index = xtest.index, columns = y.columns).clip(0.)
    ypred = ypred.stack(['store_nbr', 'family'])
    sub = pd.read_csv(f'{args.datapath}/data/sample_submission.csv')
    sub['sales'] = ypred.values
    sub.to_csv('submission7.csv', index = False) # Submit


if __name__ == "__main__":
    args = get_args()
    set_all_seeds(seed=args.seed)

    gc.enable()
    filterwarnings('ignore')
    simplefilter('ignore')

    main(args)