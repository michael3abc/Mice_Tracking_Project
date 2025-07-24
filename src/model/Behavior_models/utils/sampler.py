import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import List, Literal, Tuple, Optional

def sample_windows_by_bout(
        X: np.ndarray,
        y: np.ndarray,
        bouts: np.ndarray,
        n_per_behavior: int          = 50,
        max_win_per_behavior: int | None = None,
        random_state: int            = 42,
)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:    
    """
    按行為(bout)重新抽測試資料：每種行為隨機抽 n_per_behavior 段 bout，
    並將對應所有窗口(samples)回傳。

    參數:
    - X: np.ndarray, shape=(N_windows, T, F)
    - y: np.ndarray, shape=(N_windows,)
    - bouts: np.ndarray, 每個 window 的 bout_id, shape=(N_windows,)
    - n_per_behavior: 每種行為欲抽的 bout 數目
    - random_state: 隨機種子

    回傳:
    - X_sel, y_sel, bouts_sel: 經 bout 重取樣後的子集
    """
    rng = np.random.default_rng(random_state)
    sel_idxs = []

    for beh in np.unique(y):
        # 1) 在 test 中，找出此行為所有的 bout_id
        all_bouts = np.unique(bouts[y == beh])
        # -- 抽 n_per_behavior 段 bout --不夠就全取
        chosen_b  = (
                rng.choice(all_bouts, n_per_behavior, replace=False)
                if len(all_bouts) > n_per_behavior else all_bouts
                     )
        
        # ── 2. 收集該行為的 window idx ────────
        beh_idxs = np.concatenate([
            np.where((bouts == b) & (y == beh))[0] for b in chosen_b
        ])
        # ── 3. window 上限───────────────
        cap = None
        if isinstance(max_win_per_behavior, dict):
            cap = max_win_per_behavior.get(beh, max_win_per_behavior.get("default"))
        else:
            cap = max_win_per_behavior  # int 或 None

        if cap is not None and len(beh_idxs) > cap:
            beh_idxs = rng.choice(beh_idxs, size=cap, replace=False)

        sel_idxs.extend(beh_idxs.tolist())

    sel_idxs = np.array(sel_idxs, dtype=int)
    return X[sel_idxs], y[sel_idxs], bouts[sel_idxs]

def assign_bout_ids(df):
    """
    給 df 加一個 bout_id：
    在同一 video_id 裡，每次 behavior 變化就 +1。
    """
    df = df.copy()
    # 如果上一row的行為不同 => start
    df['bout_change'] = (
        df['behavior'] != df.groupby('video_id')['behavior'].shift(1)
        ).astype(int)
    
    # cumulative sum（cumsum): 把一部影片總共換了幾次行為 => 可以切成幾個bout
    df['bout_id'] = df.groupby('video_id')['bout_change'].cumsum()

    df.drop(columns='bout_change', inplace=True)
    return df

def _pad_to_window(
        arr: np.array,
        window_size: int,
        mode: Literal['interp', 'edge', 'zero'] = 'interp'
        )-> np.array:
    """
    將長度 < window_size 的序列補齊到 window_size。
    arr  : 原始 (T_full, F) 陣列
    回傳: (window_size, F) 陣列
    """
    T_full, F = arr.shape #幀數、kpts
    pad_len = window_size - T_full

    # 全部補0
    if mode == 'zero':
        pad = np.zeros((pad_len, F), dtype=arr.dtype)
        return np.vstack([arr, pad])
    
    # 尾幀重複
    if mode == 'edge':
        edge = np.repeat(arr[-1:, :], pad_len, axis=0)
        return np.vstack([arr, edge])
    
    # 預設: 線性插值
    src_idx = np.linspace(0, T_full - 1, num=T_full)        # 產生一條根T_full一樣長的時間軸 ex.T = 60 => [0, 1, 2, …, 59]
    tgt_idx = np.linspace(0, T_full - 1, num=window_size)   # 目標時間軸: 以T_full長度拉伸，產生[0, 1, ..., 96-1]
    interp  = np.empty((window_size, F), dtype=arr.dtype)   # (96, F) 的空陣列放結果
    # 對每個kpts 在T上做線性插值
    for f in range(F):
        interp[:, f] = np.interp(tgt_idx, src_idx, arr[:,f])
    return interp

def sliding_windows(
    df: pd.DataFrame,
    window_size: int,
    min_pad_ratio: float,
    stride: int,
    kpt_cols: List[str],
    box_cols: List[str] = None, 
    pad_mode: Literal['interp', 'edge', 'zero'] = 'interp'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    把含「關鍵點 + 行為標籤」的 DataFrame，切成滑動視窗形式，
    並可同時提取 box 的中心座標與面積作為額外特徵。

    參數
    -------
    df : pd.DataFrame  
        必須至少包含：
        - 'video_id'：影片編號
        - 'behavior'：行為類別
        - 'bout_id' ：同一行為段編號
        - kpt_cols   ：所有關鍵點欄位（共 F 欄）
        - 若需 box 資訊，box_cols 應包含 ["box_x","box_y","box_w","box_h"]
    window_size : int  
        每個視窗含多少影格 (= T)。
    min_pad_ratio : float  
        若該 bout 幀數 < window_size * min_pad_ratio，則跳過。
    stride : int  
        視窗滑動步幅；stride < window_size 代表有重疊。
    kpt_cols : List[str]  
        關鍵點欄位名稱清單，共 F 欄。
    box_cols : Optional[List[str]]  
        若非 None，則從 df 擷取這四欄，
        並計算 (center_x, center_y, area) 三維特徵。
    pad_mode : {'interp','edge','zero'}  
        若 bout 幀數介於 min_pad_ratio 與 window_size 之間，採用何種方式補齊。


    回傳
    -------
    X  : np.ndarray, shape=(N_windows, T, F + (3 if box_cols else 0))
    y  : np.ndarray, shape=(N_windows,)
    vid: np.ndarray, shape=(N_windows,)
    bid: np.ndarray, shape=(N_windows,)
    """
    windows, labels, vids, bids = [], [], [], []

    # g對應的是 根據vid, bout_id: 切的到的sub_df (i.e. g = df[df.video_id == i][df.bid == j])
    for (vid, bid), g in df.groupby(['video_id', 'bout_id']):
        arr_kpt = g[kpt_cols].to_numpy(dtype = np.float32) # (T_frames, F): kpts的特徵矩陣 (kpts* T個frame)
        arr_box = g[box_cols].to_numpy(dtype = np.float32)

        T_full = arr_kpt.shape[0]                           # 該bout共有幾幀

        # 太短 → 跳過
        if T_full < window_size * min_pad_ratio:
            continue

        # 介於 min_pad_ratio 與 window_size → 補齊
        if T_full < window_size:
            arr_kpt = _pad_to_window(arr_kpt, window_size, mode=pad_mode)
            arr_box = _pad_to_window(arr_box, window_size, mode=pad_mode)
            T_full = window_size

        # slide
        for start in range(0, (T_full-window_size+1), stride):
            end = start + window_size
            win_kpt = arr_kpt[start:end]                      # (window_size, 16): 8kpts*2
            win_box = arr_box[start: end]                     # (window_size, 4 ): [x, y, w, h]

            # combine: kpts + box
            win_feat = np.concatenate([win_box, win_kpt], axis=1)
            windows.append(win_feat)

            label = g['behavior'].iloc[0]
            labels.append(label)

            vids.append(vid)       # 影片編號
            bids.append(bid)      # bout 編號

    return (np.stack(windows),   # X_windows
            np.array(labels),    # y_labels
            np.array(vids),      # vids
            np.array(bids) )     # bids


def print_bouts_per_behavior(split_name, y_arr, bouts_arr):
    """
    列出 split_name 這組資料中，
    每個行為字串對應到幾個獨立 bout_id
    """
    print(f"\n>>> [{split_name}] 各行為 Bout 數量")
    for beh in np.unique(y_arr):
        bout_ids = np.unique(bouts_arr[y_arr == beh])
        print(f"  {beh:>10s} : {len(bout_ids):4d} bouts")

def print_windows_per_behavior(split_name: str,
                               y_arr: np.ndarray):
    """
    列出 split_name 這組資料中，
    每個行為(label)對應到多少個 windows (samples)
    """
    print(f"\n>>> [{split_name}] 各行為 Window 數量")
    for beh in np.unique(y_arr):
        cnt = int((y_arr == beh).sum())
        print(f"  {beh:>10s} : {cnt:5d} windows")






# ========= random sampling: 隨機抽幀+前後n幀作為一個windows =========
def sample_center_windows(
        df: pd.DataFrame,
        center_per_class: int = 600,
        half_window: int = 15,
        random_state: int = 42
    ):
    """
    依 paper 方法抽樣：
      1. 先對每一種行為隨機抽 `center_per_class` 個中心幀。
      2. 以中心幀為基準，前後各取 half_window 幀 → window_size = 2*half_window+1。
      3. 若 window 觸及影片端點則捨棄該中心幀。
    參數
    ----
    df : 資料表，至少需有 ['video_id','frame','behavior', 所有 kpt col...]
    回傳
    ----
    X_windows : np.ndarray,  shape = (N, T, K) ，
                其中 T = 2*half_window+1, K = 16 (8 kpts × xy)
    y_labels  : np.ndarray,  shape = (N,)
    video_ids : np.ndarray,  shape = (N,)
    """
    rng    = np.random.default_rng(random_state)
    T      = 2 * half_window + 1
    kpt_cols = [c for c in df.columns if c.startswith("kpt")]
    
    # --- 1. 轉成 NumPy 陣列（一次性） ---
    vid_arr  = df["video_id"].to_numpy()
    frm_arr  = df["frame"].to_numpy()
    beh_arr  = df["behavior"].to_numpy()
    kpt_arr  = df[kpt_cols].to_numpy(dtype=np.float32)   # shape = (N, K)
    
    # --- 2. 建立「影片 → {frame: row_index}」快取 ---
    vid2map   = defaultdict(dict)   # (video_id, frame) -> row_index
    vid2min   = {}
    vid2max   = {}
    for idx, (v, f) in enumerate(zip(vid_arr, frm_arr)):
        vid2map[v][f] = idx
    for v, frames in vid2map.items():
        keys = frames.keys()
        vid2min[v], vid2max[v] = min(keys), max(keys)
    
    # --- 3. 開始依行為抽樣 ---
    windows, labels, vids = [], [], []
    for beh in np.unique(beh_arr):
        indices = np.where(beh_arr == beh)[0]
        pick    = rng.choice(
            indices, size=center_per_class,
            replace=len(indices) < center_per_class
        )
        for idx in pick:
            v   = vid_arr[idx]
            f   = frm_arr[idx]
            s, e = f - half_window, f + half_window
            if s < vid2min[v] or e > vid2max[v]:
                continue   # 邊界不夠
            try:
                rows = [vid2map[v][fr] for fr in range(s, e+1)]
            except KeyError:
                # 中間缺幀 → 捨棄
                continue
            windows.append(kpt_arr[rows])  # (T, K)
            labels.append(beh)
            vids.append(v)

    X = np.stack(windows).astype(np.float32)   # (N, T, K)
    y = np.asarray(labels)
    v = np.asarray(vids)
    return X, y, v

def sample_stratified_by_video(
    df: pd.DataFrame,
    center_per_class: int = 600,
    half_window: int = 15,
    random_state: int = 42
):
    rng      = np.random.default_rng(random_state)
    T        = 2 * half_window + 1
    kpt_cols = [c for c in df.columns if c.startswith("kpt")]

    # 快取 numpy 陣列
    vid_arr = df["video_id"].to_numpy()
    frm_arr = df["frame"].to_numpy()
    beh_arr = df["behavior"].to_numpy()
    kpt_arr = df[kpt_cols].to_numpy(dtype=np.float32)

    # 建立方便查的 map
    vid2idx = defaultdict(dict)
    for i,(v,f) in enumerate(zip(vid_arr,frm_arr)):
        vid2idx[v][f] = i
    vid2min = { v: min(frames) for v,frames in vid2idx.items() }
    vid2max = { v: max(frames) for v,frames in vid2idx.items() }

    windows, labels, vids = [], [], []

    for beh in np.unique(beh_arr):
        # 這個行為出現在哪些影片？
        vids_with_beh = np.unique(vid_arr[beh_arr == beh])
        M = len(vids_with_beh)
        if M == 0:
            continue

        # 每隻影片先分到的「均攤配額」
        base = center_per_class // M
        # 還有 remainder 個要整體補
        rem  = center_per_class - base * M

        # 先在每隻影片裡面取 base 筆
        for v in vids_with_beh:
            idxs = np.where((vid_arr == v) & (beh_arr == beh))[0]
            # 不夠就放回，不要就 replace=False
            n_pick = min(base, len(idxs))
            picks = rng.choice(idxs, size=n_pick, replace=False)
            for center_idx in picks:
                f = frm_arr[center_idx]
                s, e = f-half_window, f+half_window
                if s < vid2min[v] or e > vid2max[v]:
                    continue
                try:
                    rows = [vid2idx[v][ff] for ff in range(s,e+1)]
                except KeyError:
                    continue
                windows.append(kpt_arr[rows])
                labels.append(beh)
                vids.append(v)

        # 把餘數再從所有這個行為 frame 裡任意補足
        all_idxs = np.where(beh_arr == beh)[0]
        if rem>0 and len(all_idxs)>0:
            picks = rng.choice(all_idxs, size=rem, replace=len(all_idxs)<rem)
            for center_idx in picks:
                v = vid_arr[center_idx]
                f = frm_arr[center_idx]
                s, e = f-half_window, f+half_window
                if s < vid2min[v] or e > vid2max[v]:
                    continue
                try:
                    rows = [vid2idx[v][ff] for ff in range(s,e+1)]
                except KeyError:
                    continue
                windows.append(kpt_arr[rows])
                labels.append(beh)
                vids.append(v)

    X = np.stack(windows).astype(np.float32)
    y = np.asarray(labels)
    v = np.asarray(vids)
    return X, y, v

# def sliding_windows_videos(
#     df: pd.DataFrame,
#     window_size: int,
#     stride: int,
#     kpt_cols: List[str],
#     box_cols: List[str],
#     pad_mode: Literal['interp', 'edge', 'zero'] = 'interp'
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """
#     以「整支影片」為單位，對每支影片做滑動視窗。

#     參數
#     -------
#     df : pd.DataFrame  
#         必須至少包含：
#         - 'video_id'：影片編號
#         - 'frame_idx': 幀索引（用於排序）
#         - 'behavior'：行為標籤（字串）
#         - kpt_cols   ：關鍵點欄位清單 (16 維)
#         - box_cols   ：包圍盒欄位清單 (4 維)

#     window_size : int  
#         每個視窗含多少影格 (= T)。

#     stride : int  
#         視窗滑動步幅。

#     kpt_cols : list[str]  
#         所有關鍵點欄位名稱（共 16 維）。

#     box_cols : list[str]  
#         包圍盒欄位名稱（共 4 維）。

#     pad_mode : {'interp','edge','zero'}
#         若影片長度 < window_size，用何種方式補齊。

#     回傳
#     -------
#     X    : np.ndarray, shape=(N_windows, window_size, F)
#            F = len(box_cols) + len(kpt_cols)

#     y    : np.ndarray, shape=(N_windows,)
#            每個視窗的行為標籤 (依多數票決定)

#     vids : np.ndarray, shape=(N_windows,)
#            每個視窗所屬的 video_id
#     """
#     windows: List[np.ndarray] = []
#     labels:  List[str]       = []
#     vids:    List[int]       = []

#     # 依 video_id 分組，並按 frame_idx 排序
#     for vid, g in df.groupby('video_id'):
#         g = g.sort_values('frame')
#         # 取 keypoints / boxes 陣列
#         arr_kpt = g[kpt_cols].to_numpy(dtype=np.float32)  # (T_frames, F_kpt)
#         arr_box = g[box_cols].to_numpy(dtype=np.float32)  # (T_frames, F_box)
#         T_full  = arr_kpt.shape[0]

#         # 若小於一個 window，就補到 window_size
#         if T_full < window_size:
#             arr_kpt = _pad_to_window(arr_kpt, window_size, mode=pad_mode)
#             arr_box = _pad_to_window(arr_box, window_size, mode=pad_mode)
#             T_full = window_size

#         # 在每支影片上滑動切窗
#         for start in range(0, T_full - window_size + 1, stride):
#             end = start + window_size

#             win_kpt = arr_kpt[start:end]   # (window_size, F_kpt)
#             win_box = arr_box[start:end]   # (window_size, F_box)

#             # 合併特徵：先箱，再關鍵點 → (window_size, F_box+F_kpt)
#             win_feat = np.concatenate([win_box, win_kpt], axis=1)

#             # 擷取對應這個 window 的行為標籤，並以多數票決定
#             behaviors = g['behavior'].iloc[start:end].values
#             label = Counter(behaviors).most_common(1)[0][0]

#             windows.append(win_feat)
#             labels.append(label)
#             vids.append(vid)

#     X    = np.stack(windows, axis=0)   # (N, window_size, F)
#     y    = np.array(labels, dtype=object)
#     vids = np.array(vids,   dtype=int)
#     return X, y, vids
