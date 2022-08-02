# import albumentations
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class DataLoader(Dataset):
    """
        Dataset class for PRnet
    """
    # The data is now the SpatioTemporal Maps instead of videos

    def __init__(self, data_paths, data_type):
        
        self.data_paths = data_paths
        self.data_type = data_type
        
    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        
        frame = self.data_paths[index]
        # print(frame)
        split_arr = frame.split("_")
        p_path = split_arr[0].split("/")[-1]
        hr_data_path = f"{p_path}/{split_arr[1]}/{split_arr[2]}"
        hr_index = split_arr[-2]
        # frame_file = f"{self.data_paths[index]}/frames.npy"
        
        array = np.load(frame)
        array = array.transpose(3, 0, 1, 2)
        
        
        # target_arr = []

        hr_file = f"data/{hr_data_path}/hr.csv"
        df = pd.read_csv(hr_file, header=None)
        hrs = df[0]
        target_hr = hrs[int(hr_index)]
        # for num in range(len(hrs)):
        #     target_arr.append(hrs[num])
        
        if self.data_type == 'train':
            return torch.tensor(array, dtype=torch.float, device=torch.device('cuda:2')), torch.tensor(target_hr, dtype=torch.float, device=torch.device('cuda:2'))
        else:    
            return torch.tensor(array, dtype=torch.float, device=torch.device('cuda:0')), torch.tensor(target_hr, dtype=torch.float, device=torch.device('cuda:0'))
