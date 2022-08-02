import numpy as np
import pandas as pd
import glob
import os

def sortNum(text):
    split_arr = text.split('.')
    num = split_arr[0].split('_')[-1]
    return int(num)

if __name__ == '__main__':
        
    ndarray_datas = sorted(glob.glob(f"data/d1-99/*"), key=sortNum)
    # print(ndarray_datas)

    save_type = "train"
    train_num = 9600
    for num in range(train_num):
        print(num)
        data = ndarray_datas[num]
        save_data = np.load(data)
        file_name = data.split("/")[-1]
        np.save(f"data/{save_type}/{file_name}", save_data)

    # frame = ndarray_datas[0]
    # split_arr = frame.split("_")
    # p_path = split_arr[0].split("/")[-1]
    # hr_data_path = f"{p_path}/{split_arr[1]}/{split_arr[2]}"
    # hr_index = split_arr[-2]
    # # frame_file = f"{self.data_paths[index]}/frames.npy"
    
    # array = np.load(frame)
    # array = array.transpose(0, 3, 1, 2)
    
    
    # # target_arr = []

    # hr_file = f"data/{hr_data_path}/hr.csv"
    # df = pd.read_csv(hr_file, header=None)
    # hrs = df[0]
    # target_hr = hrs[int(hr_index)]
    # # for num in range(len(hrs)):
    # #     target_arr.append(hrs[num])
    
    # print("array")
    # print(array.shape)
    # print("hr")
    # print(target_hr)
