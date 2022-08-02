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
        # self.H = 180
        # self.W = 180
        # self.C = 3
        # self.video_path = data_path
        # self.st_maps_path = st_maps_path
        # # self.resize = resize
        # self.target_path = target_signal_path
        # self.maps = None

        # mean = (0.485, 0.456, 0.406)
        # std = (0.229, 0.224, 0.225)
        # Maybe add more augmentations
        # self.augmentation_pipeline = albumentations.Compose(
        #     [
        #         albumentations.Normalize(
        #             mean, std, max_pixel_value=255.0, always_apply=True
        #         )
        #     ]
        # )
        
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
        # # identify the name of the video file so as to get the ground truth signal
        # self.video_file_name = self.st_maps_path[index].split('/')[-1].split('.')[0]
        # # targets, timestamps = read_target_data(self.target_path, self.video_file_name)
        # # sampling rate is video fps (check)

        # # Load the maps for video at 'index'
        # self.maps = np.load(self.st_maps_path[index])
        # map_shape = self.maps.shape
        # self.maps = self.maps.reshape((-1, map_shape[3], map_shape[1], map_shape[2]))

        # # target_hr = calculate_hr(targets, timestamps=timestamp)
        # # target_hr = calculate_hr_clip_wise(map_shape[0], targets, timestamps=timestamps)
        # target_hr = get_hr_data(self.video_file_name)
        # # To check the fact that we dont have number of targets greater than the number of maps
        # # target_hr = target_hr[:map_shape[0]]
        # self.maps = self.maps[:target_hr.shape[0], :, :, :]
        # return {
        #     "st_maps": torch.tensor(self.maps, dtype=torch.float),
        #     "target": torch.tensor(target_hr, dtype=torch.float)
        # }
