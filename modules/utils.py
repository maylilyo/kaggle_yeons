import torch
import random
import numpy as np

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.benchmark = False  # 재연이 필요하기 때문에 False로 설정. True로 하면 느린 대신 성능 증가
    # torch.backends.cudnn.deterministic = True

    print("All random seeds set to", seed)