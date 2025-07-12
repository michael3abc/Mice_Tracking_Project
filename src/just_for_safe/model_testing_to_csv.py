import os
import cv2
import sys
import torch
import numpy as np
import pandas as pd
from collections import deque
from ultralytics import YOLO
from tqdm import tqdm

# 把專案根目錄加到 sys.path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# 載入行為模型
from Pose_to_behavior.model.Behaviors_Models import BehaviorBiLSTM

def get_unique_path(path: str) -> str:
    """
    如果 path 已存在，就在檔名後面加上 _1、_2… 直到找到一個不存在的路徑。
    例如：output.csv → output_1.csv → output_2.csv …
    """
    base, ext = os.path.splitext(path)
    counter = 1
    unique_path = path
    while os.path.exists(unique_path):
        unique_path = f"{base}_{counter}{ext}"
        counter += 1
    return unique_path

# === 參數設定 ===
video_path        = r"C:\Users\micha\Desktop\full_database\20080324115556.mpg"
yolo_weights      = r"C:\Users\micha\Desktop\python_workspace\YOLO Mice Project\src\V6_pose\mouse_pose_yolo8l_v6\weights\best.pt"
behavior_weights  = r"C:\Users\micha\Desktop\python_workspace\YOLO Mice Project\src\Pose_to_behavior\model\best_models\best_epoch100_BehaviorBiLSTM_1.pth"
minmax_npz        = r"C:\Users\micha\Desktop\python_workspace\YOLO Mice Project\src\Pose_to_behavior\model\minmax_values.npz"
output_csv        = r"C:\Users\micha\Desktop\python_workspace\YOLO Mice Project\src\GUI_tracking\prediction_results\keypoints_behavior.csv"

device            = "cuda" if torch.cuda.is_available() else "cpu"
pose_input_size   = 640
window_size       = 16
kp_history_len    = 3
rest_prob_margin  = 0.2
min_any_duration  = 96
behavior_names    = ['eat', 'groom', 'hang', 'micromovement', 'rear', 'rest', 'walk']
yolo_conf_thresh  = 0.3

# === 載入模型 & 參數 ===
yolo_model = YOLO(yolo_weights)
yolo_model.fuse()

behavior_model = BehaviorBiLSTM().to(device)
behavior_model.load_state_dict(torch.load(behavior_weights, map_location=device))
behavior_model.eval()

# min‐max normalization parameters
npz        = np.load(minmax_npz)
X_min      = npz["X_min"].reshape(-1)
X_max      = npz["X_max"].reshape(-1)

# === 工具函式 ===
def restore_and_normalize_keypoints(keypoints, orig_size, input_size=640):
    orig_w, orig_h = orig_size
    scale = min(input_size/orig_w, input_size/orig_h)
    pad_w = (input_size - orig_w*scale) / 2
    pad_h = (input_size - orig_h*scale) / 2
    restored = []
    for x_res, y_res in keypoints:
        x_orig = (x_res - pad_w) / scale
        y_orig = (y_res - pad_h) / scale
        x_norm = np.clip(x_orig/orig_w, 0, 1)
        y_norm = np.clip(y_orig/orig_h, 0, 1)
        restored.append((x_orig, y_orig, x_norm, y_norm))
    return restored

def calculate_velocity_acceleration(window):
    if len(window) < 7:
        return None
    arr   = np.stack(window, axis=0)  # shape (T,16)
    v_t   = arr[-1]   - arr[-4]
    v_t_1 = arr[-4]   - arr[-7]
    a_t   = v_t      - v_t_1
    return np.concatenate([v_t, a_t], axis=0)

def smooth_predictions(prob_window, rest_index):
    T, C = prob_window.shape
    raw  = np.argmax(prob_window, axis=1)
    out  = raw.copy()
    start = 0
    while start < T:
        lbl = raw[start]
        end = start + 1
        while end < T and raw[end] == lbl:
            end += 1
        length = end - start

        if lbl == rest_index:
            rest_probs = prob_window[start:end, lbl]
            low_conf   = rest_probs < rest_prob_margin
            if low_conf.any():
                top2 = np.argsort(prob_window[start:end], axis=1)[:, -2:]
                for i, (a, b) in enumerate(top2):
                    if low_conf[i]:
                        out[start+i] = a if b == lbl else b

        elif lbl != rest_index and length < min_any_duration:
            prev_lbl = out[start-1] if start>0 else None
            next_lbl = raw[end]     if end<T else None
            if prev_lbl is not None and prev_lbl == next_lbl:
                fill = prev_lbl
            else:
                left  = (raw[max(0,start-50):start][::-1] == prev_lbl).sum() if prev_lbl is not None else 0
                right = (raw[end:end+50] == next_lbl).sum() if next_lbl is not None else 0
                fill  = prev_lbl if left >= right else next_lbl
            if fill is not None:
                out[start:end] = fill

        start = end
    return out

def process_one_frame(frame, frame_idx, kps_xy, kps_conf,
                      kp_history, pose_window, model_window, behavior_probs, rows):
    """ 單幀做補點、特徵抽取、行為預測並存進 rows """
    orig_h, orig_w = frame.shape[:2]

    # — 補點（simple interp 或 last valid）—
    kps = []
    for idx, (x, y) in enumerate(kps_xy):
        if x > 0 and y > 0:
            pt = (x, y)
            kp_history[idx].append(pt)
        else:
            pt = kp_history[idx][-1] if kp_history[idx] else (np.nan, np.nan)
        kps.append(pt)

    # — 正規化 & 特徵向量 —
    restored = restore_and_normalize_keypoints(kps, (orig_w, orig_h), pose_input_size)
    flat_norm = np.array([v for (_,_,xn,yn) in restored for v in (xn,yn)], dtype=np.float32)
    flat_pt   = np.array([coord for pt in kps for coord in pt], dtype=np.float32)
    scaled    = (flat_pt - X_min) / (X_max - X_min + 1e-6)
    scaled    = np.clip(scaled, 0, 1)

    pose_window.append(scaled)
    v_a = calculate_velocity_acceleration(pose_window)
    if v_a is None:
        return  # 還不夠幀數
 
    X_full = np.concatenate([flat_norm, scaled, v_a], axis=0)
    model_window.append(X_full)

    # — 行為推論 & 平滑 —
    inp    = np.stack(model_window).T  # shape (64, window_size)
    with torch.no_grad():
        logits = behavior_model(torch.tensor(inp[None], dtype=torch.float32).to(device))
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    behavior_probs.append(probs)

    if len(behavior_probs) < window_size:
        return

    sm     = smooth_predictions(np.stack(behavior_probs), rest_index=behavior_names.index("rest"))
    final  = int(sm[-1])
    top3   = [(behavior_names[i], float(probs[i])) for i in np.argsort(probs)[-3:][::-1]]

    # — 儲存到 rows —
    row = [frame_idx] + [coord for pt in kps for coord in pt]
    for b, p in top3:
        row += [b, p]
    rows.append(row)

# === 主程式：Batch 推論 + 輸出 CSV ===
def main():
    cap = cv2.VideoCapture(video_path)

    # 緩存
    kp_history     = [deque(maxlen=kp_history_len) for _ in range(8)]
    pose_window    = deque(maxlen=window_size)
    model_window   = deque(maxlen=window_size)
    behavior_probs = deque(maxlen=window_size)
    rows           = []

    # Batch
    BATCH_SIZE  = 8
    frame_batch = []
    idx_batch   = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Processing frames")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1   

        frame_batch.append(frame)
        idx_batch.append(frame_idx)

        pbar.update(1)

        if len(frame_batch) == BATCH_SIZE:
            results = yolo_model(frame_batch, conf=yolo_conf_thresh)
            # 批次處理完之後再for 1~8處理各幀
            for res, fid, frm in zip(results, idx_batch, frame_batch):
                # 如果沒偵測到任何 keypoints，就給全 NaN／0，交給 process_one_frame 補點
                if res.keypoints is None or res.keypoints.xy.shape[0] == 0:
                    kps_xy   = [(np.nan, np.nan)] * 8
                    kps_conf = [0.0] * 8
                else:
                    kps_xy   = res.keypoints.xy.cpu().numpy()[0]
                    kps_conf = res.keypoints.conf.cpu().numpy()[0]
                process_one_frame(
                    frame=frm, frame_idx=fid,
                    kps_xy=kps_xy, kps_conf=kps_conf,
                    kp_history=kp_history,
                    pose_window=pose_window,
                    model_window=model_window,
                    behavior_probs=behavior_probs,
                    rows=rows
                )
            frame_batch.clear()
            idx_batch.clear()

    # 處理尾巴
    if frame_batch:
        results = yolo_model(frame_batch, conf=yolo_conf_thresh)
        for res, fid, frm in zip(results, idx_batch, frame_batch):
            pbar.update(1)            
            if res.keypoints is None or res.keypoints.xy.shape[0] == 0:
                kps_xy   = [(np.nan, np.nan)] * 8
                kps_conf = [0.0] * 8
            else:
                kps_xy   = res.keypoints.xy.cpu().numpy()[0]
                kps_conf = res.keypoints.conf.cpu().numpy()[0]

            process_one_frame(
                frame=frm, frame_idx=fid,
                kps_xy=kps_xy, kps_conf=kps_conf,
                kp_history=kp_history,
                pose_window=pose_window,
                model_window=model_window,
                behavior_probs=behavior_probs,
                rows=rows
            )

    cap.release()
    pbar.close() 

    # 輸出 CSV
    cols = ["frame"] + [f"kpt{i}_{a}" for i in range(8) for a in ('x','y')] \
           + [f"behavior{r}" for r in (1,2,3)] + [f"prob{r}" for r in (1,2,3)]
    df = pd.DataFrame(rows, columns=cols)

    save_path = get_unique_path(output_csv)
    df.to_csv(save_path, index=False, float_format="%.4f")
    print(f"已儲存到 {save_path}")

if __name__ == "__main__":
    main()
