from ultralytics import YOLO
import pandas as pd
from tqdm import tqdm
import os

model = YOLO(r"C:\Users\micha\Desktop\python_workspace\YOLO Mice Project\src\V6_pose\mouse_pose_yolo8l_v6\weights\best.pt")

# 2. 定義 keypoint 欄位名稱（假設你要用 kpt0_x…kpt7_y）
kpt_cols = []
for idx in range(8):
    kpt_cols += [f'kpt{idx}_x', f'kpt{idx}_y']

for i in tqdm(range(2,3), desc = "進度"):
    rows = []
    video_path = fr"C:\Users\micha\Desktop\python_workspace\YOLO Mice Project\data\mice{i}.mpg"    

    results = model.predict(source=video_path, show = False, save = False, conf =  0.5)
    for frame_idx, res in enumerate(tqdm(results)):
        if len(res.keypoints) == 0:
            row = [frame_idx] + [float('nan')]*8*2
        else:
            #取8點xy，轉成1*16
            kpt = res.keypoints[0].xy
            flat = kpt.reshape(-1).tolist()
            row = [frame_idx] + flat
        rows.append(row)

    cols = ["frame"] + kpt_cols
    df_kpt = pd.DataFrame(rows, columns= cols)
    out_root = r"C:\Users\micha\Desktop\python_workspace\YOLO Mice Project\src\GUI_tracking\prediction_results\1. keypoints"
    out_path  = os.path.join(out_root,f'kepyoints_{i}.csv' )
    df_kpt.to_csv(out_path,  index= False)
    print(f"已輸出kpt{i}")