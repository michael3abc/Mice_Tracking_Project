# --------------------------------------------
#  Keypoints CSV ➜ BiLSTM 行為預測 CSV
#  支援命令列與 config 檔兩種設定方式
# --------------------------------------------
import os, sys, argparse
import numpy as np, pandas as pd, torch
from collections import deque
# 專案根目錄自動加入 sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from model.Behavior_models_old.Behaviors_Models import BehaviorBiLSTM, BehaviorBiLSTM_v3, BehaviorSTGCN_BiLSTM, SimpleSTTR
from model.Behavior_models_old.utils import (
    compute_velocity_acc,
    compute_cos_np,
    compute_space_distances,
    compute_direction_unit,
    compute_speed_std,
)

# 載入 config 檔案
import yaml
cfg_path = r"src\main\config.yaml"
with open(cfg_path, "r", encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

orig_w, orig_h = cfg["yolo"]["orig_size"]
window = cfg["behavior"]["window_size"]
kp_history_len = cfg["pose"]["kp_history_len"]
DEFAULT_MINMAX= cfg["paths"]["minmax_npz"]
DEFAULT_MODEL = cfg["behavior"]["weights"]
BEHAVIORS = ['eat','groom','hang','micromovement','rear','rest','walk']
REST_IDX  = BEHAVIORS.index('rest')

from collections import deque

history = [deque(maxlen=kp_history_len) for _ in range(8)]  # 假設跟 GUI 用的 kp_history_len=3


def smooth(prob, rest_margin=0.2, min_seg=32):
    """
    後處理機率序列，針對 rest 類別進行微調、短片段做 smoothing
    """
    raw, out = prob.argmax(1), prob.argmax(1).copy()
    t=0
    while t<len(raw):
        lbl, e = raw[t], t+1
        while e<len(raw) and raw[e]==lbl: e+=1
        seg=slice(t,e); L=e-t
        if lbl==REST_IDX:
            low = prob[seg,lbl] < rest_margin
            if low.any():
                top2=np.argsort(prob[seg],1)[:,-2:]
                for i,(a,b) in enumerate(top2):
                    if low[i]: out[t+i]=a if b==lbl else b
        elif L<min_seg:
            pl=out[t-1] if t>0 else None
            nl=raw[e] if e<len(raw) else None
            fill=pl if pl==nl else (pl if
                 (raw[max(0,t-50):t][::-1]==pl).sum() >=
                 (raw[e:e+50]==nl).sum() else nl)
            if fill is not None: out[seg]=fill
        t=e
    return out

def build70_via_utils(
    kp_px: np.ndarray,
    kp_rel: np.ndarray,
    win_mm: deque,
    win_rl: deque,
    vel_delta: int,
    Xmin: np.ndarray,
    Xmax: np.ndarray,
) -> np.ndarray | None:
    # 1) flat relative (16,)
    flat_rel = kp_rel.flatten().astype(np.float32)

    # 2) flat minmax (16,)
    flat_px = np.nan_to_num(kp_px.flatten(), nan=0.0).astype(np.float32)
    flat_mm = (flat_px - Xmin) / (Xmax - Xmin + 1e-6)

    # push into deques
    win_mm.append(flat_mm)
    win_rl.append(flat_rel)

    # 需要先累滿 2*vel_delta+1 幀
    if len(win_mm) < 2*vel_delta + 1:
        return None

    # stack → (1, T, 16)
    Xs = np.stack(win_mm, axis=0)[None, ...]  # (1, T, 16)

    # velocity & acceleration → each (1, T, 16)
    vel, acc = compute_velocity_acc(Xs, delta=vel_delta)
    # 把 batch 維度去掉 → (T,16)
    vel = vel[0]; acc = acc[0]

    # concat 最新一幀的 v_t, a_t → (32,)
    v_t = vel[-1]
    a_t = acc[-1]
    velacc = np.concatenate([v_t, a_t], axis=0)

    # cos(angle) → (T,1) 然後取最後一筆
    cos_seq = compute_cos_np(win_rl_to_kptseq(win_rl))  # see 下方輔助
    cos_val = float(cos_seq[-1,0])

    # space distances → (1,T,2)
    space_seq = compute_space_distances(Xs)
    space_val = space_seq[0, -1]   # (2,)

    # direction unit → (1,T,2)
    dir_seq = compute_direction_unit(vel[None, ...])
    dir_val = dir_seq[0, -1]       # (2,)

    # speed std → (1,T,1)
    std_seq = compute_speed_std(vel[None, ...])
    std_val = float(std_seq[0, -1, 0])

    # all together 16+16+32+1+2+2+1 = 70
    return np.concatenate([
        flat_rel, flat_mm, velacc,
        [cos_val], space_val, dir_val, [std_val]
    ], axis=0).astype(np.float32)

def win_rl_to_kptseq(win_rl: deque) -> np.ndarray:
    """
    把 win_rl (deque of 16-d rel) 轉成 (T,8,2)
    """
    arr = np.stack(win_rl, axis=0)       # (T,16)
    return arr.reshape(-1, 8, 2)         # (T,8,2)


''''''
# def predict_csv(kpt_csv,model_pth,minmax_npz,out_csv,window=32,device="cuda"):
#     """
#     核心流程：
#     1. 讀取 keypoint csv
#     2. 特徵轉換 + 時序窗
#     3. 行為模型推論
#     4. 行為結果與原始資料對齊並存檔
#     """

#     dev="cuda" if (device=="cuda" and torch.cuda.is_available()) else "cpu"
#     df=pd.read_csv(kpt_csv)
#     # kpts=df.drop(columns='frame').values.reshape(len(df),8,2).astype(np.float32)
#     kpt_cols = [f'kpt{i}_{xy}' for i in range(8) for xy in 'xy']
#     kpts = df[kpt_cols].values.reshape(len(df), 8, 2 ).astype(np.float32)

#     #box_centers的，目前不用
#     '''
#     # 如果有 bbox 也要特徵
#     bboxes = df[['box_x', 'box_y', 'box_w', 'box_h']].values.astype(np.float32)
#     # 展平成一維
#     flat_kpts = kpts.reshape(len(df), -1)  # shape: (N, 16)
#     # 合併進 model 特徵 (如果需要)
#     features = np.concatenate([flat_kpts, bboxes], axis=1)  # shape: (N, 20)
#     '''

#     net=BehaviorBiLSTM().to(dev)
#     net.load_state_dict(torch.load(model_pth,map_location=dev)); net.eval()

#     npz=np.load(minmax_npz)
#     Xmin,Xmax=npz["X_min"].reshape(-1),npz["X_max"].reshape(-1)

#     history = [deque(maxlen=kp_history_len) for _ in range(8)]
#     win_s = deque(maxlen=window)
#     win_f = deque(maxlen=window)
#     probs = []

#     # 每一幀做滑動視窗 feature
#     for frame_idx, kp_px in enumerate(kpts):

#         # Keypoint 補點平滑，避免瞬間抖動或遺失
#         for idx, (x, y) in enumerate(kp_px):
#             if x > 0 and y > 0:
#                 history[idx].append((x, y))
#         valid = []
#         for idx in range(8):
#             if history[idx]:
#                 avg = np.mean(history[idx], axis=0)
#                 valid.append(avg)
#             else:
#                 valid.append([orig_w/2, orig_h/2])

#         valid = np.array(valid)
#         norm = valid.copy()
#         norm[:, 0] /= orig_w
#         norm[:, 1] /= orig_h

#         feat = build64(valid, norm, win_s, Xmin, Xmax)

#         if feat is None:
#             win_f.append(np.zeros(64, np.float32))
#             continue
#         win_f.append(feat)
#         if len(win_f) < window:
#             continue

#         # 丟進 BiLSTM，拿到行為機率分布
#         X = np.stack(win_f).T
#         with torch.no_grad():
#             p = torch.softmax(net(torch.tensor(X[None], dtype=torch.float32, device=dev)), 1)[0].cpu().numpy()
#         probs.append(p)


#     if not probs:
#         print("序列不足，無法推論"); return
#     probs=np.vstack(probs)
#     labels=smooth(probs)

#     # 對齊原始 keypoint 長度，預測不到的（前 window-1 幀）直接補 None
#     num_empty = window - 1
#     total_len = len(df)
#     full_behavior = [None] * num_empty + [BEHAVIORS[i] for i in labels]
#     # 最後不夠的一樣補None
#     full_behavior += [None] * (total_len - len(full_behavior))
#     assert len(full_behavior) == len(df)


#     # 確保 probs 已經是 (有效幀數, num_behaviors) 的 numpy array
#     top3_idx = np.argsort(probs, axis=1)[:, -3:][:, ::-1]  # 每一行降冪排列 top3 index
#     top3_prob = np.take_along_axis(probs, top3_idx, axis=1)  # shape: (幀數, 3)
#     top3_label = np.array(BEHAVIORS)[top3_idx]  # shape: (幀數, 3)

#     full_top_label = {}
#     full_top_prob = {}
#     for i in range(3):
#         full_top_label[i] = [None] * num_empty + top3_label[:, i].tolist()
#         full_top_prob[i]  = [None] * num_empty + top3_prob[:, i].tolist()
#         full_top_label[i] += [None] * (total_len - len(full_top_label[i]))
#         full_top_prob[i]  += [None] * (total_len - len(full_top_prob[i]))

#     # 寫進 DataFrame
#     df['behavior'] = full_behavior
#     for i in range(3):
#         df[f"top{i+1}"] = full_top_label[i]
#         df[f"prob{i+1}"] = full_top_prob[i]

#     # 輸出單一檔案模式（behavior直接併到原本 keypoint csv）
#     if out_csv is not None: #i.e. single 輸出模式
#         os.makedirs(os.path.dirname(out_csv), exist_ok=True)
#         # 建議直接 append 回原 df，這樣欄位順序一致
#         df['behavior'] = full_behavior
#         df.to_csv(out_csv, index=False)
#         print(f"✓ Saved to {out_csv} | Device: {dev}")

#     # 回傳完整行為序列，讓 pipeline 可以彈性組合
#     return df


def predict_csv(model, kpt_csv, model_pth=DEFAULT_MODEL, minmax_npz=DEFAULT_MINMAX,
                out_csv=None, window=window, device="cuda"):
    dev = "cuda" if (device=="cuda" and torch.cuda.is_available()) else "cpu"
    df  = pd.read_csv(kpt_csv)
    kpt_cols = [f'kpt{i}_{xy}' for i in range(8) for xy in 'xy']
    kpts = df[kpt_cols].values.reshape(len(df), 8, 2).astype(np.float32)

    # 載模型
    net = model().to(dev)
    net.load_state_dict(torch.load(model_pth, map_location=dev))
    net.eval()

    npz = np.load(minmax_npz)
    Xmin, Xmax = npz["X_min"].reshape(-1), npz["X_max"].reshape(-1)

    history = [deque(maxlen=kp_history_len) for _ in range(8)]
    win_mm   = deque(maxlen=window)
    win_rl   = deque(maxlen=window)
    win_f    = deque(maxlen=window)  # 用來暫存最終特徵
    probs    = []

    for frame_idx, kp_px in enumerate(kpts):
        # 1) 平滑，計算 valid_px
        for i, (x,y) in enumerate(kp_px):
            if x>0 and y>0:
                history[i].append((x,y))
        valid_px = np.zeros((8,2), np.float32)
        for i in range(8):
            if history[i]:
                valid_px[i] = np.mean(history[i], axis=0)
            else:
                valid_px[i] = [orig_w/2, orig_h/2]

        # 2) 相對歸一化 valid_rel
        valid_rel = valid_px.copy()
        valid_rel[:,0] /= orig_w
        valid_rel[:,1] /= orig_h

        # 3) 呼叫 build70_via_utils
        feat70 = build70_via_utils(
            kp_px=valid_px,
            kp_rel=valid_rel,
            win_mm=win_mm,
            win_rl=win_rl,
            vel_delta=cfg["pose"]["vel_delta"],
            Xmin=Xmin,
            Xmax=Xmax
        )
        if feat70 is None:
            win_f.append(np.zeros(70, np.float32))
            continue

        win_f.append(feat70)
        if len(win_f) < window:
            continue

        # 4) 模型推論
        X_input = np.stack(win_f).T  # (70, window)
        X_tensor = torch.tensor(X_input[None], dtype=torch.float32, device=dev)  # (1, 70, window)
        X_tensor = X_tensor.unsqueeze(-1)  # (1, 70, window, 1)

        with torch.no_grad():
            p = torch.softmax(
                net(X_tensor),
                dim=1
            )[0].cpu().numpy()
        probs.append(p)

    if not probs:
        print("序列不足，無法推論")
        return

    probs = np.vstack(probs)
    labels = smooth(probs)

    # 對齊回原始長度
    num_empty   = window - 1
    total_len   = len(df)
    full_behav  = [None]*num_empty + [BEHAVIORS[i] for i in labels]
    full_behav += [None]*(total_len - len(full_behav))
    df['behavior'] = full_behav

    # top3
    top3_idx  = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
    top3_prob = np.take_along_axis(probs, top3_idx, axis=1)
    top3_lbl  = np.array(BEHAVIORS)[top3_idx]
    for i in range(3):
        col_lbl = [None]*num_empty + top3_lbl[:,i].tolist()
        col_prb = [None]*num_empty + top3_prob[:,i].tolist()
        col_lbl += [None]*(total_len - len(col_lbl))
        col_prb += [None]*(total_len - len(col_prb))
        df[f"top{i+1}"]  = col_lbl
        df[f"prob{i+1}"] = col_prb

    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"✓ Saved to {out_csv} | Device: {dev}")

    return df


if __name__=="__main__":
    for i in range(1,13):
        kpt_csv     = fr"data_prediction\prediction_results\1_keypoints\pass2\keypoints_mice{i}.csv"
        # 訓練好的模型檔
        model_pth   = r"src\model\Behavior_models\best_models\SimpleSTTR_best_epoch11_F10.7928.pth"
        # min–max npz
        minmax_npz  = r"minmax_GCN_LSTM.npz"
        # 想輸出的行為結果檔

        out_csv    = fr"data_prediction\prediction_results\2_behavios\ST-TR\behaviors_mice{i}.csv"
        
        # 確保輸出目錄存在
        out_dir = os.path.dirname(out_csv)
        os.makedirs(out_dir, exist_ok=True)

        # predict_csv(
        #     model = BehaviorBiLSTM_v3,
        #     kpt_csv    = kpt_csv,
        #     model_pth  = model_pth,
        #     minmax_npz = minmax_npz,
        #     out_csv    = out_csv,
        #     window     = 96,       # 你的 window_size
        #     device     = "cuda"    # or "cpu"
        # )

        predict_csv(
            model = lambda: SimpleSTTR(
                    in_channels= 70,   # 你輸入的70維 feature 是放在 channel 維度
                    d_model=128,      # 自訂特徵維度（注意力維度），可以自己調
                    num_heads=8,     # 多頭注意力數量
                    num_classes=7    # 你總共有 7 類行為               
            ),
            kpt_csv    = kpt_csv,
            model_pth  = model_pth,
            minmax_npz = minmax_npz,
            out_csv    = out_csv,
            window     = 96,       # 你的 window_size
            device     = "cuda"    # or "cpu"
        )



    
