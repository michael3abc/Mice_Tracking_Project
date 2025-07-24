def video_to_keypoint(video_path, out_csv):
    '''
    將影片透過yolo預測keypoints => 輸出成pred_keypoint
    ''' 
    from ultralytics import YOLO
    import pandas as pd
    from tqdm import tqdm
    import os
    model = YOLO(r"src\model\YOLO\YOLO_weights\mouse_pose_yolo11L_150\weights\best.pt")

    # 2. 定義 keypoint 欄位名稱（假設你要用 kpt0_x…kpt7_y）
    kpt_cols = []
    for idx in range(8):
        kpt_cols += [f'kpt{idx}_x', f'kpt{idx}_y']

    rows = []
    results = model.predict(source=video_path, show = False, save = False, conf = 0.3)    
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
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_kpt.to_csv(out_csv,  index= False)
    print(f"已輸出: {out_csv}")


import cv2
import numpy as np
import pandas as pd
import os
from collections import deque
from ultralytics import YOLO
from tqdm import tqdm
import yaml
from pathlib import Path

def video_to_keypoints_batch(
    video_path,
    out_csv = None,
    yolo_weight = None,
    yolo_conf = 0.3,
    batch_size=16,
    kp_history_len=3
):
    yolo = YOLO(yolo_weight)
    cap = cv2.VideoCapture(video_path)
    kpt_cols = [f'kpt{i}_{xy}' for i in range(8) for xy in 'xy']
    history = [deque(maxlen=kp_history_len) for _ in range(8)]
    rows = []
    frames, frame_ids = [], []
    frame_id = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="YOLO Keypoints", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        frames.append(frame)
        frame_ids.append(frame_id)
        pbar.update(1)
        
        # 若滿一個 batch 或到結尾
        if len(frames) == batch_size:
            results = yolo(frames, conf=yolo_conf)
            for idx, res in enumerate(results):
                if len(res.keypoints) == 0:
                    kp = np.zeros((8,2), dtype=np.float32)
                else:
                    kp = res.keypoints.xy[0].cpu().numpy().astype(np.float32)

                # 取得 bbox（預設 fallback）
                h, w = frames[idx].shape[:2]
                box_x, box_y = 0, 0
                box_w, box_h = w, h
                bbox_center = [w/2, h/2]
                if hasattr(res, 'boxes') and res.boxes is not None and len(res.boxes) > 0:
                    box = res.boxes.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = box
                    box_x, box_y = x1, y1
                    box_w, box_h = x2 - x1, y2 - y1
                    bbox_center = [(x1 + x2)/2, (y1 + y2)/2]

                # 補點滑動平均
                for k_idx, (x, y) in enumerate(kp):
                    if x > 0 and y > 0:
                        history[k_idx].append((x, y))
                valid = []
                for k_idx in range(8):
                    if history[k_idx]:
                        avg = np.mean(history[k_idx], axis=0)
                        valid.append(avg)
                    else:
                        valid.append(bbox_center)
                row = [frame_ids[idx], box_x, box_y, box_w, box_h] + np.array(valid).flatten().tolist()
                rows.append(row)
            frames, frame_ids = [], []

    # 處理最後不滿一個 batch 的 frames
    if frames:
        results = yolo(frames, conf=yolo_conf)
        for idx, res in enumerate(results):
            if len(res.keypoints) == 0:
                kp = np.zeros((8,2), dtype=np.float32)
            else:
                kp = res.keypoints.xy[0].cpu().numpy().astype(np.float32)

            h, w = frames[idx].shape[:2]
            box_x, box_y = 0, 0
            box_w, box_h = w, h
            bbox_center = [w/2, h/2]
            if hasattr(res, 'boxes') and res.boxes is not None and len(res.boxes) > 0:
                box = res.boxes.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = box
                box_x, box_y = x1, y1
                box_w, box_h = x2 - x1, y2 - y1
                bbox_center = [(x1 + x2)/2, (y1 + y2)/2]

            for k_idx, (x, y) in enumerate(kp):
                if x > 0 and y > 0:
                    history[k_idx].append((x, y))
            valid = []
            for k_idx in range(8):
                if history[k_idx]:
                    avg = np.mean(history[k_idx], axis=0)
                    valid.append(avg)
                else:
                    valid.append(bbox_center)
            row = [frame_ids[idx], box_x, box_y, box_w, box_h] + np.array(valid).flatten().tolist()
            rows.append(row)

    cap.release()
    df_kpt = pd.DataFrame(rows, columns=["frame", "box_x", "box_y", "box_w", "box_h"] + kpt_cols)
    if out_csv is not None:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df_kpt.to_csv(out_csv, index=False)
        print(f"[Keypoints-batch] 已輸出: {out_csv}")
    pbar.close()
    return df_kpt


# 使用範例
if __name__ == "__main__":
    # 依實際yaml載入參數    
    cfg_path = r"src\main\config.yaml"
    with open(cfg_path,'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    yolo_weight = cfg["yolo"]["weights"]
    kp_history_len = cfg["pose"]["kp_history_len"]
    yolo_conf = cfg["yolo"]["conf"]

    video_folder = Path(r"C:\Users\micha\Desktop\dataset_video")
    output_folder = Path(r"data_prediction\prediction_results\1_keypoints\yolov8")
    output_folder.mkdir(parents=True, exist_ok=True)


    for i in range(8,13):
        video_path = video_folder / f"mice{i}.mpg"
        output_csv = output_folder / f"keypoints_mice{i}.csv"

        video_to_keypoints_batch(
            video_path=str(video_path),
            out_csv=str(output_csv),
            yolo_weight=yolo_weight,
            yolo_conf=yolo_conf,
            batch_size=32,
            kp_history_len=kp_history_len
        )
