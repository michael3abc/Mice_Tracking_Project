import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

def Check_Class(df):
    print("剩下的行為種類：", df['behavior'].unique())
    print("每種行為的樣本數：\n", df['behavior'].value_counts())
    
    '''
    剩下的行為種類： ['micromovement' 'walk' 'rear' 'eat' 'groom' 'drink' 'hang' 'rest']
    每種行為的樣本數：
    behavior
    groom            262666
    micromovement    251138
    rear             110212
    walk              91976
    hang              81625
    eat               80480
    rest              57224
    drink              2776

    由於drink太少 => 省略
    '''


    '''
    df(乾淨、未打亂的原始資料)
        ↓
    per-video sliding → 得到很多 window: (X, y)
        ↓
    再丟進 Balance_data() → 根據 y 來平衡 sample 數

    '''

def smooth_short_events(preds, min_duration):
    #避免模型過度敏感，在不同行為之間抖動

    smoothed = preds.copy()
    N = len(preds)
    start = 0
    while start < N:
        label = preds[start]
        end = start + 1
        while end < N and preds[end] == label:
            end += 1
        length = end - start

        if length < min_duration:
            prev_lbl = smoothed[start-1] if start>0 else None
            next_lbl = preds[end]       if end< N else None

            # 前後一致就填同一；不一致就比長度
            if prev_lbl is not None and prev_lbl == next_lbl:
                fill = prev_lbl
            else:
                # 計算前後段長度
                left_len = sum(1 for i in range(start-1,-1,-1) if preds[i]==prev_lbl) if prev_lbl is not None else 0
                right_len= sum(1 for i in range(end,N)     if preds[i]==next_lbl) if next_lbl is not None else 0
                fill = prev_lbl if left_len>=right_len else next_lbl

            if fill is not None:
                for i in range(start,end):
                    smoothed[i] = fill

        start = end
    return smoothed

def load_and_clean(data_path):
    """讀 CSV、過濾無用標籤、排序、做短事件平滑"""
    df = pd.read_csv(data_path)
    df = df[~df["behavior"].isin(["unknown", "drink"])]
    df = df.sort_values(["video_id", "frame"]).reset_index(drop=True)

    groups = []
    for vid, g in df.groupby("video_id"):
        labels = smooth_short_events(g["behavior"].tolist(), min_duration=0)
        tmp = g.copy()
        tmp["behavior"] = labels
        groups.append(tmp)
    return pd.concat(groups, ignore_index=True)

class FallbackWrapper(nn.Module):
    """
    在inference時，如果rest低於一定conf threshold => 輸出第二高的 
    """
    def __init__(self, base_model, rest_idx:int=6, tau:float=0.6):
        super().__init__()
        self.model    = base_model
        self.rest_idx = rest_idx
        self.tau      = tau

    def forward(self, x):
        logits = self.model(x)
        if self.training:
            return logits

        probs  = F.softmax(logits, dim=1)                 # (B, C)
        top2_p, top2_idx = probs.topk(2, dim=1)          # 取前兩名
        preds = top2_idx[:, 0].clone()                   # 先用最高分
        is_rest  = (preds == self.rest_idx)
        low_conf = probs[:, self.rest_idx] < self.tau

        n_fallback = int((is_rest & low_conf).sum().item())
        # if n_fallback > 0:
        #     print(f"[FallbackWrapper] {n_fallback} 個樣本 rest_conf < {self.tau:.2f}，執行 fallback")

        preds[is_rest & low_conf] = top2_idx[is_rest & low_conf, 1]

        return preds