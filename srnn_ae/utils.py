import torch
import random
import numpy as np

def set_random_seed(seed=777):
    '''
    Args:
        seed: int, random seed, default=777
    Returns:
        None
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 딥러닝 재현성을 위해 추가 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False