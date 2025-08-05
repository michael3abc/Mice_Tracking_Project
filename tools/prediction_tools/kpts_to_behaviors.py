#  Keypoints CSV ➜ BiLSTM 行為預測 CSV

import os, sys, torch, pickle, yaml
import pandas as pd
import numpy as np
from pathlib import Path
from collections import deque
from tqdm import tqdm
from torch.cuda.amp import autocast




ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.gui.inference import make_engine, load_config


def kpts_to_behaviors(
        cfg_path:       str,
        input_csv:      str,
        output_folder:  str,
        output_name:    str,
        weights:        str | None = None
):
                    
    """
    讀入 kpts CSV，跑行為預測，把 behavior + top1/2/3、prob1/2/3
    寫回一份新的 CSV。完全重用 GUI 的 sliding-window & _predict_behavior 邏輯。
    """

    # 1. 建立engine
    cfg = load_config(cfg_path)
    engine = make_engine(cfg_path)

    if weights:
        engine.ycfg["weight"] = weights
        engine._init_behavior_model()

    T       = cfg["behavior"]["window_size"]
    stride  = cfg["behavior"]["predict_stride"]
    orig_w, orig_h = cfg["yolo"]["orig_size"]
    vel_d   = cfg["pose"]["vel_delta"]
    swl     = cfg["pose"]["smooth_window_length"]
    poly    = cfg["pose"]["polyorder"]
    stats   = None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    engine.device = device

    # 2. 讀kpts csv

    df = pd.read_csv(input_csv)
    # 欄名前 20 個就是 raw20
    raw20_cols = df.columns[1:21].tolist()  # 假設第一欄是 frame
    assert len(raw20_cols)==20, f"Expect 20 raw cols, got {len(raw20_cols)}"

    raw20_all  = df[raw20_cols].to_numpy(dtype=np.float32) 



    # 3. sliding windows
    valid_idxs = [i for i in range(T-1, len(raw20_all)) if i%stride == 0]
    windows = np.stack([raw20_all[i-T+1:i+1] for i in valid_idxs], axis=0)


    # 4. result containers
    N = len(df)
    behaviors = [""]*N
    t1, p1 = [np.nan]*N, [np.nan]*N
    t2, p2 = [np.nan]*N, [np.nan]*N
    t3, p3 = [np.nan]*N, [np.nan]*N


    X_full, _, _, _ = build_features(
        windows,                        # (1,T,20)
        ["dummy"]*len(windows),                  # y_dummy
        orig_size=(orig_w, orig_h),
        vel_delta=vel_d,
        smooth_window_length=swl,
        polyorder=poly,
        stats=stats
    )
    results = [engine._predict_behavior(x) for x in tqdm(X_full)]   # (T,F)

    # lbl, top3 = engine._predict_behavior(single_window)
    for idx, (lbl, top3) in zip(valid_idxs, results):
        behaviors[idx] = lbl
        names, probs = zip(*top3)           # top3 = [(name, prob), ...]
        t1[idx],p1[idx] = names[0],probs[0]
        t2[idx],p2[idx] = names[1],probs[1]
        t3[idx],p3[idx] = names[2],probs[2]

    # 6. 寫回df
    df["behavior"] = behaviors
    df["top1"], df["prob1"] = t1, p1
    df["top2"], df["prob2"] = t2, p2
    df["top3"], df["prob3"] = t3, p3

    # 6.2. 把stride中的空白值補成上次判斷
    df[["behavior","top1","prob1","top2","prob2","top3","prob3"]] = \
    df[["behavior","top1","prob1","top2","prob2","top3","prob3"]].ffill()


    # 7. 輸出成 csv
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{output_name}.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[kpts_to_behaviors] saved → {out_path}")

# if __name__ == "__main__":

#     kpts_to_behaviors(
#         cfg_path       = r"src\gui\gui_config.yaml",
#         input_csv      = r"data\prediction_results\1_keypoints\yolov11\keypoints_mice1.csv",
#         output_folder  = r"data\prediction_results\2_behavios\STTR_BiLSTM",
#         output_name    = "mics1_pred_behavior",
#         weights=None
#     )

from src.utils.build_feature import build_features
from src.utils.model_factory import build_model

# ==========================================================================================================================================================
def predict_behavior_from_kpts(
    cfg_path:      str,
    exp_folder:    str,
    input_csv:     str,
    output_folder: str,
    out_name:      str = None,
    out_put_csv:   bool=False,
    batch_size:    int = 512
):
    # ─────────────────── 1. 載入設定與模型結構 ───────────────────
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    model_params = cfg["model"]["params"]
    orig_size = cfg["data"]["orig_size"]
    device = torch.device(cfg["train"]["device"])

    model_path = os.path.join(exp_folder, "best_model.pth")
    encoder_path = os.path.join(exp_folder, "label_encoder.pkl")
    # minmax_path = os.path.join(exp_folder, "minmax_stats.npz")
    feat_dim = cfg["train"]["feature_dim"]

    with open(encoder_path, "rb") as f:
        le = pickle.load(f)
    behavior_names = le.classes_.tolist()

    # stats = np.load(minmax_path)
    # minmax_stats = {"mins": stats["X_min"], "maxs": stats["X_max"]}

    model = build_model(model_name, feat_dim, len(behavior_names), model_params)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # ─────────────────── 2. 讀取關鍵點 CSV → sliding windows ───────────────────
    df = pd.read_csv(input_csv)
    raw20_cols = df.columns[1:21].tolist()
    print(raw20_cols)
    raw20_all = df[raw20_cols].to_numpy(dtype=np.float32)
    print(len(raw20_all))

    T = cfg["sliding_window"]["window_size"]
    stride = cfg["sliding_window"]["stride"]
    valid_idxs = list(range(T-1, len(raw20_all), stride))

    windows = np.stack([raw20_all[i-T+1:i+1] for i in valid_idxs], axis=0)  # shape: (M, T, 20)

    # ─────────────────── 3. build_features → (M, T, F) ───────────────────
    X_feat, _, _, _ = build_features(
        windows,
        y = None,
        orig_size=orig_size,
        vel_delta=cfg["vel_delta"],
        smooth_window_length=cfg["smooth"]["window_length"],
        polyorder=cfg["smooth"]["polyorder"],
        le=le,
        # stats=minmax_stats
    )

    # ─────────────────── 4. 模型推論 (不平滑) ───────────────────

    all_probs = []
    with torch.no_grad():
        for start in range(0, X_feat.shape[0], batch_size):
            end = start + batch_size
            xb = X_feat[start:end]
            xb_tensor = torch.tensor(xb, dtype=torch.float32).to(device)

            with autocast():
                logits = model(xb_tensor)
            probs_batch  = torch.softmax(logits, dim=1).cpu().numpy()  # (M, C)
            all_probs.append(probs_batch)
    probs = np.concatenate(all_probs, axis = 0)

    # ─────────────────── 5. 解析 top1 + top3 ───────────────────
    N = len(df)
    behaviors = [np.nan] * N
    t1, p1 = [np.nan] * N, [np.nan] * N
    t2, p2 = [np.nan] * N, [np.nan] * N
    t3, p3 = [np.nan] * N, [np.nan] * N

    for idx, prob in zip(valid_idxs, probs):
        top3_idx = np.argsort(prob)[-3:][::-1]
        top3 = [(behavior_names[i], float(round(prob[i], 4))) for i in top3_idx]

        behaviors[idx] = top3[0][0]
        t1[idx], p1[idx] = top3[0]
        t2[idx], p2[idx] = top3[1]
        t3[idx], p3[idx] = top3[2]

    # ─────────────────── 6. 寫入 CSV ───────────────────
    df["behavior"] = behaviors
    df["top1"], df["prob1"] = t1, p1
    df["top2"], df["prob2"] = t2, p2
    df["top3"], df["prob3"] = t3, p3

    df[["behavior", "top1", "prob1", "top2", "prob2", "top3", "prob3"]] = \
        df[["behavior", "top1", "prob1", "top2", "prob2", "top3", "prob3"]].ffill()
    
    if out_put_csv:
        out_dir = Path(output_folder)
        out_dir.mkdir(parents=True, exist_ok=True)

        output_csv =  out_dir / f"{out_name}.csv"
        df.to_csv(output_csv, index=False, encoding="utf-8")
        print(f"✔ Done. Output saved to: {output_csv}")
    
    else:
        return df


if __name__ == "__main__":
    yaml_path = r"src\gui\gui_config.yaml"
    gui_cfg = yaml.safe_load(open(yaml_path, encoding="utf-8"))

    predict_behavior_from_kpts(
        cfg_path = r"model\behavior_models\train_config.yaml",
        exp_folder=gui_cfg["paths"]["experiment_root"],
        input_csv = r"data\prediction_results\3_combine\mice5_kpts.csv",
        output_folder = r"data\prediction_results\3_combine",
        out_name= "mice5_combimed",
        out_put_csv = True,
        batch_size=512
    )







    
