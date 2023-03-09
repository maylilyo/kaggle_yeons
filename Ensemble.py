import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

def to_nparray(s) :
        return np.array(list(map(float, s[1:-1].split(','))))

def main():
    path1 = f'./ensemble/submission3.csv'
    path3 = f'./ensemble/submission7.csv'
    path4 = f'./ensemble/submission11.csv'

    df1 = pd.read_csv(path1)
    df3 = pd.read_csv(path3)                                                        
    df4 = pd.read_csv(path4)
    
    df1['sales']  = df1['sales'].apply(lambda x:x*0.25) + df3['sales'].apply(lambda x:x*0.5) + df4['sales'].apply(lambda x:x*0.25)
    df1.to_csv(f'./ensemble/ensemble_submit2.csv', index=False)



if __name__ == '__main__':
    main()



# sub = pd.DataFrame(df_pred[['sub_38415','sub_38558']].mean(axis=1), columns=['sales']).reset_index()
# sub.to_csv('submission.csv', index = False)