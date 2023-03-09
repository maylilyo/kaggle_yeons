import pandas as pd
from statsmodels.tsa.deterministic import DeterministicProcess, CalendarFourier

def make_calendar(args, oil_df, holiday_df):
    # making calendar df (avg_oil이 non값이 아닌 행만 남겨두는 프로세스)
    calendar = pd.DataFrame(index = pd.date_range('2013-01-01', '2017-08-31'))
    calendar = calendar.to_period('D') # TODO : 도대체 저 to_period 왜 하는지 모르겠음
    calendar = calendar.join(oil_df.avg_oil)
    calendar['avg_oil'].fillna(method = 'ffill', inplace = True)
    calendar.dropna(inplace = True)

    # oil data를 잘 활용하기 위한 data augmentation
    for l in range(1, args.n_lags + 1) :
        calendar[f'oil_lags{l}'] = calendar.avg_oil.shift(l)
    calendar.dropna(inplace = True)

    calendar = calendar.join(holiday_df) # Joining calendar with holiday dataset
    calendar['dofw'] = calendar.index.dayofweek # Weekly day
    calendar['wd'] = 1
    calendar.loc[calendar.dofw > 4, 'wd'] = 0 # If it's saturday or sunday then it's not Weekday
    calendar.loc[calendar.type == 'Work Day', 'wd'] = 1 # If it's Work Day event then it's a workday
    calendar.loc[calendar.type == 'Transfer', 'wd'] = 0 # If it's Transfer event then it's not a work day
    calendar.loc[calendar.type == 'Bridge', 'wd'] = 0 # If it's Bridge event then it's not a work day
    calendar.loc[(calendar.type == 'Holiday') & (calendar.transferred == False), 'wd'] = 0 # If it's holiday and the holiday is not transferred then it's holiday
    calendar.loc[(calendar.type == 'Holiday') & (calendar.transferred == True), 'wd'] = 1 # If it's holiday and transferred then it's not holiday
    calendar = pd.get_dummies(calendar, columns = ['dofw'], drop_first = True) # One-hot encoding (Make sure to drop one of the columns by 'drop_first = True')
    calendar = pd.get_dummies(calendar, columns = ['type']) # One-hot encoding for type holiday (No need to drop one of the columns because there's a "No holiday" already)
    calendar.drop(['transferred'], axis = 1, inplace = True) # Unused columns

    school_season = [] # Feature for school fluctuations
    for i, r in calendar.iterrows() :
        if i.month in [4, 5, 8, 9] :
            school_season.append(1)
        else :
            school_season.append(0)
    calendar['school_season'] = school_season

    return calendar


# def get_data(args):
#     # train csv file preprocess
#     train_df = pd.read_csv(f'{args.datapath}/train.csv', parse_dates = ['date'], infer_datetime_format = True)
#     train_df = train_df.drop(['id', 'onpromotion'], axis=1, errors='ignore')
#     train_df = train_df.astype({'store_nbr' : 'category', 'family' : 'category'})
#     train_df['date'] = train_df.date.dt.to_period('D')  # date를 day 간격으로 index 변환
#     train_df = train_df.set_index(['date', 'store_nbr', 'family']).sort_index()  # date-storenumber-family 순으로 data sorting

    
#     # oil csv file preprocess
#     oil_df = pd.read_csv(f'{args.datapath}/oil.csv', parse_dates = ['date'], infer_datetime_format = True)  # 날짜를 parsing하는 column
#     oil_df = oil_df.set_index(keys=['date'], inplace=False, drop=True)
#     oil_df = oil_df.to_period('D') # TODO : 이걸 안하면 에러가 남. 진짜왜지?????
#     oil_df['avg_oil'] = oil_df['dcoilwtico'].rolling(7).mean() # 이동산술평균 책정. 왜 하는거지? 이유 필요

    
#     # holiday csv file preprocess
#     holiday_df = pd.read_csv(f'{args.datapath}/holidays_events.csv', parse_dates = ['date'], infer_datetime_format = True)
#     holiday_df = holiday_df.set_index(keys=['date'], inplace=False, drop=True)
#     holiday_df = holiday_df.to_period('D') # TODO
#     holiday_df = holiday_df[holiday_df.locale == 'National']
#     # holiday dataset에는 자국의/타국의 holiday 데이터가 섞여있음.
#     # 시점 하나에서 holiday를 관측하기 위해 타국의 정보를 지워버리는 것으로 보임.
#     # TODO : 외국 데이터도 포함해서 실험해볼 여지는 있음.
#     holiday_df = holiday_df.groupby(holiday_df.index).first() # 동일한 날짜에 있는 중복 holiday 삭제
#     holiday_df = holiday_df.drop(['locale', 'locale_name', 'description'], axis=1, errors='ignore')
    
#     # make calendar df
#     calendar = make_calendar(args, oil_df, holiday_df)

#     # school fluctuations : 학교 방학 정보. TODO : 이게 의미가 있을까?
#     school_season = [] 
#     for i, r in calendar.iterrows() :
#         if i.month in [4, 5, 8, 9] :
#             school_season.append(1)
#         else :
#             school_season.append(0)
#     calendar['school_season'] = school_season

    
#     # TODO : Deterministic..?
#     y = train_df.unstack(['store_nbr', 'family']).loc[args.start_date:args.end_date]
#     fourier = CalendarFourier(freq = 'W', order = 4)
#     dp = DeterministicProcess(index = y.index,
#                             order = 1,
#                             seasonal = False,
#                             constant = False,
#                             additional_terms = [fourier],
#                             drop = True)
#     x = dp.in_sample()
#     x = x.join(calendar)
#     xtest = dp.out_of_sample(steps = 16) # 16 because we are predicting next 16 days
#     xtest = xtest.join(calendar)

#     return x, y, xtest
    
    # TODO : stores_df는 안썼음. 왜 안썼을까?

def get_data(args):
    train_df = pd.read_csv(f'{args.datapath}/train.csv',
                    parse_dates = ['date'], infer_datetime_format = True,
                    dtype = {'store_nbr' : 'category',
                            'family' : 'category'},
                    usecols = ['date', 'store_nbr', 'family', 'sales'])
    train_df['date'] = train_df.date.dt.to_period('D')
    train_df = train_df.set_index(['date', 'store_nbr', 'family']).sort_index()


    oil_df = pd.read_csv(f'{args.datapath}/oil.csv',
                    parse_dates = ['date'], infer_datetime_format = True,
                    index_col = 'date').to_period('D')
    oil_df['avg_oil'] = oil_df['dcoilwtico'].rolling(7).mean()

    holiday_df = pd.read_csv(f'{args.datapath}/holidays_events.csv',
                    parse_dates = ['date'], infer_datetime_format = True,
                    index_col = 'date').to_period('D')
    holiday_df = holiday_df[holiday_df.locale == 'National'] # I'm only taking National holiday so there's no false positive.
    # holiday dataset에는 자국의/타국의 holiday 데이터가 섞여있음.
    # 시점 하나에서 holiday를 관측하기 위해 타국의 정보를 지워버리는 것으로 보임.
    # TODO : 외국 데이터도 포함해서 실험해볼 여지는 있음.
    holiday_df = holiday_df.groupby(holiday_df.index).first() # 동일한 날짜에 있는 중복 holiday 삭제
    holiday_df = holiday_df.drop(['locale', 'locale_name', 'description'], axis=1, errors='ignore')

    calendar = make_calendar(args, oil_df, holiday_df)

    # school_season = [] # Feature for school fluctuations
    # for i, r in calendar.iterrows() :
    #     if i.month in [4, 5, 8, 9] :
    #         school_season.append(1)
    #     else :
    #         school_season.append(0)
    # calendar['school_season'] = school_season

    y = train_df.unstack(['store_nbr', 'family']).loc[args.start_date:args.end_date]
    fourier = CalendarFourier(freq = 'W', order = 4)
    dp = DeterministicProcess(index = y.index,
                            order = 1,
                            seasonal = False,
                            constant = False,
                            additional_terms = [fourier],
                            drop = True)
    x = dp.in_sample()
    x = x.join(calendar)

    xtest = dp.out_of_sample(steps = 16) # 16 because we are predicting next 16 days
    xtest = xtest.join(calendar)
    
    return x,y,xtest