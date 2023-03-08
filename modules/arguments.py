import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datapath",
        type=str,
        default=f"{os.getcwd()}/data",
        help="directory path for dataset",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42
    )
    parser.add_argument(
        "--n_lags", 
        type=int, 
        default=3, 
        help="트렌드 예측에서의 지연 / 현재의 변수가 영향을 미치는 예측값"
    )
    parser.add_argument(
        "--start_date", 
        type=str,
        default='2017-04-30', 
        help="Start of training date"
    )
    parser.add_argument(
        "--end_date", 
        type=str, 
        default= '2017-08-15',
        help="End of training date"
    )
    # epoch
    args = parser.parse_args()
    return args
