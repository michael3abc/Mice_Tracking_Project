
import yaml, os, pandas as pd
import  numpy as np
from tqdm import tqdm
import sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) #前兩層
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from src.main.inference import InferenceEngine

cfg_path = r"src\main\config.yaml"
with open(cfg_path, 'r', encoding = 'utf-8') as f:
    cfg = yaml.safe_load(f)

engine = InferenceEngine(
            yolo_weights=cfg["yolo"]["weights"],
            yolo_conf=cfg["yolo"]["conf"],
            pose_input_size=cfg["yolo"]["input_size"],  
            orig_size = cfg["yolo"]["orig_size"],
            kp_history_len=cfg["pose"]["kp_history_len"],
            behavior_model=cfg["behavior"]["model"], 
            behavior_weights=cfg["behavior"]["weights"],
            window_size=cfg["behavior"]["window_size"],
            rest_prob_margin=cfg["behavior"]["rest_prob_margin"],
            min_any_duration=cfg["behavior"]["min_any_duration"],
            minmax_npz=cfg["paths"]["minmax_npz"],
        )

df = pd.read_csv(r"data_prediction\dataset\kpt_gt_behavior\kpt_gt_behavior.csv")
df = df.sort_values(['video_id', 'frame'])
# df = df[:1000]


window_size = engine.window_size
stride = 1
results = []
# 依影片編號分組
for vid, grp in tqdm(df.groupby("video_id"), desc="Processing videos"):
    kpts = grp[[f'kpt{i}_{a}' for i in range(8) for a in "xy"]].values
    frames = grp['frame'].values

    # 先建立一個暫存區，每一幀存預測
    frame_pred = [None] * len(frames)
    # orig_w, orig_h = engine.orig_size[0], engine.orig_size[1]

    # 每N幀滑動一次
    for i in range(0, len(kpts) - window_size + 1, stride):        
        model_input_window = []
        
        for j in range(window_size):
            kpt_frame = kpts[i+j]
            # === 1. 組成 valid dict 格式 ===
            valid = {idx: (kpt_frame[2*idx], kpt_frame[2*idx+1]) for idx in range(8)}
            # === 2. 用 build_feature_vector === 
            X_full = engine._build_feature_vector(valid, valid)
            if X_full is not None:
                model_input_window.append(X_full)
            else:
                # 缺失資料就放棄這個window
                model_input_window = []
                break

        # (B) window feature 組好就可以丟進行為預測
        if len(model_input_window) == window_size:
            engine.model_input_window.clear()
            for fv in model_input_window:
                engine.model_input_window.append(fv)
            behavior, top3 = engine._predict_behavior()  # 內部會自動平滑

            curr_frame_idx = i + window_size - 1 
            frame_pred[curr_frame_idx] = {
                "video_id": vid,
                "frame": int(frames[curr_frame_idx]),
                "behavior": behavior,
                "top3": str(top3)
            }
    for i in range(len(frames)):
        if frame_pred[i] is None:
            frame_pred[i] = {
                "video_id": vid,
                "frame": int(frames[i]),
                "behavior": "unknown",
                "top3": "[]"
            }
    results.extend(frame_pred)

def get_unique_path(path):
    base, ext = os.path.splitext(path)
    counter = 1
    new_path = path
    while os.path.exists(new_path):
        new_path = f"{base}_{counter}{ext}"
        counter += 1
    return new_path

# (6) 存成CSV
output_folder = r"data_prediction\prediction_results\2_behavios"
os.makedirs(output_folder, exist_ok=True)
csv_path = os.path.join(output_folder, "behavior_predictions.csv")
csv_path = get_unique_path(csv_path)
pd.DataFrame(results).to_csv(csv_path, index=False)
print(f"Done! 行為預測已輸出：{csv_path}")