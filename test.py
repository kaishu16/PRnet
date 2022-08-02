# coding:utf-8

import dlib
from imutils import face_utils
import cv2
import glob
import warnings
import pandas as pd
import numpy as np
import os

def get_frames(frames, num, i, save_img_path):
    ret, frame = video.read()
    if not ret:
        print("can't read video")
        return []
        
    
    ret, bbox = tracker.update(frame)
    if not ret:
        print(f"can't find bbox {num}")
        return []            

    (x,y,w,h)=[int(v) for v in bbox]

    save_frame = frame[y:y+h, x:x+w]
    
    resize_frame = cv2.resize(save_frame, dsize=(256, 128))
    # print(resize_frame)
    # if num == 1:
    #     # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
    #     # cv2.imwrite(f"check/{save_img_path[-1]}/{i}.jpg", frame)
    #     cv2.imwrite(f"{save_img_path}/{i}.jpg", resize_frame)
        
        
    frames[i, :, :, :] = resize_frame
    
    return frames

def get_bbox(target_frame):
    # --------------------------------
    # 1.顔ランドマーク検出の前準備
    # --------------------------------
    # 顔検出ツールの呼び出し
    face_detector = dlib.get_frontal_face_detector()
    # 顔のランドマーク検出ツールの呼び出し
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    face_predictor = dlib.shape_predictor(predictor_path)
    
    # --------------------------------
    # 2.顔のランドマーク検出
    # --------------------------------
    # 顔検出
    # ※2番めの引数はupsampleの回数
    faces = face_detector(target_frame, 1)
    
    # 検出した全顔に対して処理
    for face in faces:
        # 顔のランドマーク検出
        landmark = face_predictor(target_frame, face)
        # 処理高速化のためランドマーク群をNumPy配列に変換(必須)
        landmark = face_utils.shape_to_np(landmark)
        
        # ランドマーク描画
        # for (i, (x, y)) in enumerate(landmark):
        #     cv2.circle(target_frame, (x, y), 1, (255, 0, 0), -1)

        # --------------------------------
        # 3.ランドマークから画像を切り取り
        # --------------------------------
        bbox = [0, 0, 0, 0]
        
        # 左のx
        bbox[0] = landmark[3][0]

        # 右のx
        bbox[2] = landmark[13][0] - bbox[0]

        # 上のy
        bbox[1] = min(landmark[40][1], landmark[41][1], landmark[46][1], landmark[47][1])

        # 下のy
        bbox[3] = max(landmark[3][1], landmark[13][1], landmark[50][1], landmark[52][1]) - bbox[1]
        
        # print(bbox)
    
        return bbox, target_frame
    
    # print("pass")
    return [], []

if __name__ == '__main__':
    file_paths = glob.glob('VIPL-HR/*/p[1-9][0-9][0-9]/v*/source3')
    
    data_index = 0
    save_type = "test"
    for path in file_paths:
        print(path)
        
        #動画・心拍ファイル
        video_file = f"{path}/video.avi"
        hr_file = f"{path}/gt_HR.csv"
        
        # #保存ファイルパスの指定
        # split_arr = path.split("/")
        # save_path = f"data/{split_arr[-3]}/{split_arr[-2]}/{split_arr[-1]}"
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # save_img_path = f"check/{split_arr[-3]}/{split_arr[-2]}/{split_arr[-1]}"
        # print(save_path)
        
        #保存ファイルパスの指定
        split_arr = path.split("/")
        save_path = f"data/{save_type}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_img_path = f"check/{split_arr[-3]}/{split_arr[-2]}/{split_arr[-1]}"
        # print(save_path)
        
        #動画の処理
        video = cv2.VideoCapture(video_file)
        if not video.isOpened():
            print("invalid video")
            continue
        
        fps = video.get(cv2.CAP_PROP_FPS)
        count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        times = count / fps
        # print(fps, count, times)
        # print(int(times))
        
        #心拍の処理
        df = pd.read_csv(hr_file)
        hrs = df["HR"]
        # print(len(hrs))
        
        #次元数設定
        dim = 0
        if len(hrs) > int(times):
            # print("video too short")
            #心拍の時間のほうが長い場合動画の最終秒数-1までやる
            dim = int(times)-3
        else:
            #動画の方が長いためhrがあるギリギリまでやる
            dim = len(hrs)-3
        
        
        save_hrs = []
        for num in range(1, dim):
            mean_hr = format(((hrs[num] + hrs[num+1] + hrs[num+2]) / 3), '.0f') 
            # print(mean_hr)
            save_hrs.append(mean_hr)
        # print(len(save_hrs))
        
        warnings.simplefilter('ignore')
        
        # save_frames = np.zeros((len(save_hrs), 50, 128, 256, 3))
        save_frames = []
        for num in range(1, dim):
            save_frame_path = f""
            # print(num)
            frames = np.zeros((50, 128, 256, 3))
            skip_video = False
            
            start_frame = 25 * num
            end_frame = 25 * num + 49
            video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            # print(video.get(cv2.CAP_PROP_POS_FRAMES))
            
            #ROIを設定するため
            if num == 1:
                ret, frame = video.read()
                if not ret:
                    print("can't read video")
                    skip_video = True
                    break
                
                #ROI部分取得
                bbox, marked_frame = get_bbox(frame)
                
                if bbox == []:
                    print("face not found")
                    skip_video = True
                    break
                
                # print("detect bbox")
                #trackingの初期化
                tracker = cv2.TrackerKCF_create() 
                ret = tracker.init(frame, bbox)
                
                # 画像の切り出し
                save_frame = frame[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                # print(save_frame.shape)
                
                # cv2.rectangle(marked_frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 1)
                
                # if not os.path.exists(save_img_path):
                #     os.makedirs(save_img_path)
                # cv2.imwrite(f"check/{save_img_path[-1]}/0.jpg", frame)
                
                
                #ROI部分配列に追加
                resize_frame = cv2.resize(save_frame, dsize=(256, 128))
                # print(resize_frame)
                # cv2.imwrite(f"{save_img_path}/0.jpg", resize_frame)
                
                frames[0, :, :, :] = resize_frame

                
                for i in range(1, 50):
                    frames = get_frames(frames, num, i, save_img_path)
                    
                    if frames == []:
                        skip_video = True
                        break
                    
                if skip_video == True:
                    break

                    
            else:
                for i in range(0, 50):
                    frames = get_frames(frames, num, i, save_img_path)
                    
                    if frames == []:
                        skip_video = True
                        break
                    
                if skip_video == True:
                    break
                
            # save_frames[num-1, :, :, :, :] = frames
            save_frames.append(frames)
        
        if skip_video == True:
            video.release()        
            print("skip this video")
            continue
        
        # print("save frame count")
        # print(len(save_frames))
        
        for num_index in range(len(save_frames)):
            print(save_frames[num_index].shape)
            np.save(f"data/{save_type}/{split_arr[-3]}_{split_arr[-2]}_{split_arr[-1]}_{num_index}_{data_index}.npy", save_frames[num_index])
            data_index = data_index + 1        

        
        # print(save_frames)
        # np.save(f"{save_path}/frames.npy", save_frames)
        
        # with open(f"{save_path}/hr.csv", mode='w') as f:
        #     f.write('\n'.join(save_hrs))
        
        video.release()

    print(data_index)

