
# print('='*70, 'Linear Regression', '='*70)
# print(y_.groupby('family').apply(lambda r : np.sqrt(msle(r['sales'], r['lnr']))))
# print('LNR RMSLE :', np.sqrt(msle(y, yfit_lnr)))
# print('='*70, 'SVR', '='*70)
# print(y_.groupby('family').apply(lambda r : np.sqrt(msle(r['sales'], r['svr']))))
# print('SVR RMSLE :', np.sqrt(msle(y, yfit_svr)))
# print('='*70, 'Mean', '='*70)
# print(y_.groupby('family').apply(lambda r : np.sqrt(msle(r['sales'], r['mean']))))
# print('Mean RMSLE :', np.sqrt(msle(y, yfit_mean)))

# print('='*70, 'Linear Regression', '='*70)
# print(y_.groupby('family').apply(lambda r : mae(r['sales'], r['lnr'])))
# print('LNR MAE :', mae(y, yfit_lnr))
# print('='*70, 'SVR', '='*70)
# print(y_.groupby('family').apply(lambda r : mae(r['sales'], r['svr'])))
# print('SVR MAE :', mae(y, yfit_svr))
# print('='*70, 'Mean', '='*70)
# print(y_.groupby('family').apply(lambda r : mae(r['sales'], r['mean'])))
# print('Mean MAE :', mae(y, yfit_mean))


# #
# true_low = [2]
# pred_low = [4]

# print('RMSLE for low value :', np.sqrt(msle(true_low, pred_low)))
# print('MAE for low value :', mae(true_low, pred_low))

# true_high = [255]
# pred_high = [269]

# print('RMSLE for high value :', np.sqrt(msle(true_high, pred_high)))
# print('MAE for high value :', mae(true_high, pred_high))