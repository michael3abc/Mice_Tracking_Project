


#!/usr/bin/env python3
# --------------------------------------------
#  Keypoints CSV  ➜  BiLSTM 行為預測 CSV
#  不給參數就用下方 DEFAULT_* 路徑
# --------------------------------------------
import os, sys, argparse
import numpy as np, pandas as pd, torch
from collections import deque
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from model.Behaviors_Models import BehaviorBiLSTM

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

def build64(kp_px,kp_nm,win_scaled,Xmin,Xmax):
    Xmin=Xmin.reshape(-1); Xmax=Xmax.reshape(-1)
    flat_norm=kp_nm.flatten()
    flat_px=np.nan_to_num(kp_px.flatten(),nan=0.0)
    scaled=(flat_px-Xmin)/(Xmax-Xmin+1e-6)
    win_scaled.append(scaled)
    if len(win_scaled)<7: return None
    arr=np.stack(win_scaled)
    v=arr[-1]-arr[-4]
    a=v-(arr[-4]-arr[-7])
    return np.concatenate([flat_norm,scaled,v,a]).astype(np.float32)

def predict_csv(kpt_csv,model_pth,minmax_npz,out_csv,window=32,device="cuda"):


    dev="cuda" if (device=="cuda" and torch.cuda.is_available()) else "cpu"
    df=pd.read_csv(kpt_csv)
    kpts=df.drop(columns='frame').values.reshape(len(df),8,2).astype(np.float32)

    net=BehaviorBiLSTM().to(dev)
    net.load_state_dict(torch.load(model_pth,map_location=dev)); net.eval()

    npz=np.load(minmax_npz)
    Xmin,Xmax=npz["X_min"].reshape(-1),npz["X_max"].reshape(-1)

    history = [deque(maxlen=kp_history_len) for _ in range(8)]
    win_s = deque(maxlen=window)
    win_f = deque(maxlen=window)
    probs = []


    for frame_idx, kp_px in enumerate(kpts):
        # 滑動補點
        for idx, (x, y) in enumerate(kp_px):
            if x > 0 and y > 0:
                history[idx].append((x, y))
        valid = []
        for idx in range(8):
            if history[idx]:
                avg = np.mean(history[idx], axis=0)
                valid.append(avg)
            else:
                valid.append([orig_w/2, orig_h/2])
        valid = np.array(valid)
        norm = valid.copy()
        norm[:, 0] /= orig_w
        norm[:, 1] /= orig_h

        feat = build64(valid, norm, win_s, Xmin, Xmax)

        if feat is None:
            win_f.append(np.zeros(64, np.float32))
            continue
        win_f.append(feat)
        if len(win_f) < window:
            continue

        X = np.stack(win_f).T
        with torch.no_grad():
            p = torch.softmax(net(torch.tensor(X[None], dtype=torch.float32, device=dev)), 1)[0].cpu().numpy()
        probs.append(p)


    if not probs:
        print("❗ 序列不足，無法推論"); return
    probs=np.vstack(probs)
    labels=smooth(probs)
    frames=df['frame'].values[window-1:window-1+len(labels)]

    out=pd.DataFrame({
        "frame":frames,
        "behavior":[BEHAVIORS[i] for i in labels],
        **{f"prob_{b}":probs[:,i] for i,b in enumerate(BEHAVIORS)}
    })
    os.makedirs(os.path.dirname(out_csv),exist_ok=True)
    out.to_csv(out_csv,index=False,float_format="%.4f")
    print(f"✓ Saved to {out_csv} | Device: {dev}")

def batch_predict(n_files):
    for i in range(1, n_files+1):
        csv = fr"data_prediction/prediction_results/1_keypoints/keypoints_{i}.csv"
        out = fr"data_prediction\prediction_results\2_behavios\behavios_{i}.csv"
        if not os.path.exists(csv):
            print(f"[Warning] File not found, skip: {csv}")
            continue
        print(f"===== [{i}] {csv} =====")
        predict_csv(
            kpt_csv   = csv,
            model_pth = DEFAULT_MODEL,
            minmax_npz= DEFAULT_MINMAX,
            out_csv   = out,
            window    = window,
            device    = "cuda"
        )

if __name__=="__main__":
    batch_predict(6)


# # ---------------- CLI + 預設雙模式 ----------------
# if __name__=="__main__":
#     pa=argparse.ArgumentParser(
#         description="Keypoints CSV ➜ Behavior CSV (BiLSTM)")
#     pa.add_argument("--csv",   help="keypoints CSV 路徑")
#     pa.add_argument("--model", help="BehaviorBiLSTM .pth")
#     pa.add_argument("--minmax",help="minmax_values.npz")
#     pa.add_argument("--out",   help="輸出 CSV")
#     pa.add_argument("--window",type=int,default=32)
#     pa.add_argument("--cpu",action="store_true",help="強制用 CPU")
#     args=pa.parse_args()

#     # 若未提供參數 → 用上面 DEFAULT_* 路徑
#     csv   = args.csv   or DEFAULT_CSV
#     model = args.model or DEFAULT_MODEL
#     mm    = args.minmax or DEFAULT_MINMAX
#     out   = args.out   or DEFAULT_OUT
    
#     predict_csv(
#         kpt_csv   = csv,
#         model_pth = model,
#         minmax_npz= mm,
#         out_csv   = out,
#         window    = args.window,
#         device    = "cpu" if args.cpu else "cuda"
#     )
