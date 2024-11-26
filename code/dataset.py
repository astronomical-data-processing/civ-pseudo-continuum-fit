import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
# 自定义数据集
#------------------------------------------------------
class dataset(Dataset):
    def __init__(self, image, mask, part=None, f_val=0.2, seed=1):
        np.random.seed(seed)
        N_sample = image.shape[0]
        assert f_val > 0 and f_val < 1
        f_train = 1 - f_val

        if part == 'train':
            slice = np.s_[:int(np.round(N_sample * f_train))] 
        elif part == 'val': 
            slice = np.s_[int(np.round(N_sample * f_train)):]
        else:
            slice = np.s_[0:]

        self.image = image[slice]
        self.mask = mask[slice]
        print('f_val',str(f_val*100)[0:4],'%')
    
    def __len__(self):
        return self.image.shape[0]
        
    def __getitem__(self, i):
        return self.image[i], self.mask[i]
#------------------------------------------------------ 

