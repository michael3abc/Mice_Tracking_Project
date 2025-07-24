# src/data_utils.py
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.signal import savgol_filter

def smooth_kpts(X: np.ndarray,
                window_length : int = 7,
                polyorder :     int = 2):
    """
    對8*(x, y)的keypoint平滑 => 輸入build_features
    """
    N, T, F = X.shape
    out = X.copy()

    for f0 in range(0, 16, 2):
        # x
        out[:, :, f0] = savgol_filter(
            X[:, :, f0],
            window_length=window_length,
            polyorder=polyorder,
            axis=1,     #在T: 時間上做
            mode='nearest'
        )
        # y
        out[:, :, f0+1] = savgol_filter(
            X[:, :, f0+1],
            window_length=window_length,
            polyorder=polyorder,
            axis=1,
            mode='nearest'
        )
    return out
    
def compute_velocity_acc(X_rel, delta=1):    
    """
    X: shape (N, T, D) → e.g. (樣本數, 幀數, 特徵維度*16 kpts)
    delta: 幀差距，例如 delta=3 表示三幀差分 (愈大表示一次看的幀數愈長)

    回傳: (vel, acc)，形狀 (N, T, D)，只是前面 delta 幀是 0
    """
    #v_t = x_t - x_(t-det)
    #acc同理

    vel = np.zeros_like(X_rel)
    acc = np.zeros_like(X_rel)
    vel[:, delta:, :] = X_rel[:, delta:, :] - X_rel[:, :-delta, :]
    acc[:, delta:, :] = vel[:, delta:, :] - vel[:, :-delta, :]
    return vel, acc

def compute_cos_np(kpts):
    """
    kpts: np.ndarray of shape [seq_len, 8, 2]
    回傳: np.ndarray of shape [seq_len, 1]
    """
    # 拆出 nose, body, tail_base
    nose = kpts[:, 0, :]    # [seq_len,2]
    body = kpts[:, 1, :]    # [seq_len,2]
    tail = kpts[:, 2, :]    # [seq_len,2]

    v1 = nose - body        # [seq_len,2]
    v2 = tail - body        # [seq_len,2]
    dot  = np.sum(v1 * v2, axis=1, keepdims=True)                  # [seq_len,1]
    norm = np.linalg.norm(v1, axis=1, keepdims=True) * np.linalg.norm(v2, axis=1, keepdims=True)  # [seq_len,1]
    return dot / (norm + 1e-6)  # [seq_len,1]

def compute_space_distances(X_rel):
    """
    X_rel: np.ndarray, shape = (N, seq_len, 16)   # 8 个 keypoint 的 x,y
    回传: np.ndarray, shape = (N, seq_len, 3)     # nose↔tail, front↔rear
    """
    N, T, _ = X_rel.shape
    kpt = X_rel.reshape(N, T, 8, 2)
    nose = kpt[:,:,0,:]
    tail = kpt[:,:,2,:]
    f_right   = kpt[:,:,3,:]
    f_left   = kpt[:,:,4,:]
    r_right   = kpt[:,:,5,:]
    r_left   = kpt[:,:,6,:]

    d_nt = np.linalg.norm(nose - tail, axis=-1, keepdims=True)   # (N,T,1)
    d_front_rear = np.linalg.norm((f_right + f_left)  -   (r_right + r_left),   axis=-1, keepdims=True)
    return np.concatenate([d_nt, d_front_rear], axis=2)             # (N,T,2)

def compute_direction_unit(vel):
    """
    vel: np.ndarray, shape = (N, seq_len, 16)  
         # 每个 window 有 seq_len 帧，每帧 16 维 velocity (8 个 keypoint × 2 维)
    返回:
    np.ndarray, shape = (N, seq_len, 2)  
      # 对每个 window、每帧，只保留 nose 关键点的速度方向单位向量 (dx, dy)
    """
    N, T, _ = vel.shape
    # 把最后一维 16 拆成 (8,2)：8 个 keypoint，每个都有 (dx,dy)
    vk = vel.reshape(N, T, 8, 2)
    # 取出 nose（index=0）的速度向量，shape → (N, T, 2)
    nose_v = vk[:,:,0,:]                        # (N,T,2)
    # 計算單位向量
    speed  = np.linalg.norm(nose_v, axis=-1, keepdims=True) + 1e-6
    return nose_v / speed # 鼻子方向向量         # (N,T,2)

def compute_speed_std(vel):
    """
    input:
    vel: np.ndarray, shape = (N, seq_len, 16)

    ouput: np.ndarray, shape = (N, seq_len, 1)
    各window 的kpts速度 magnitude 的全局 Std，再 tile 到 T 帧
    """
    N, T, _ = vel.shape
    vk = vel.reshape(N, T, 8, 2)
    mags = np.linalg.norm(vk, axis=-1)         # sqrt(vx^2 + vy^2) → (N, T, 8)
    stds = np.std(mags.reshape(N, -1), axis=1, keepdims=True)   # (N,1) 
    return np.repeat(stds[:, None, :], T, axis=1)               # (N,T,1)

def compute_std_movement(X):
    """
    X : np.ndarray, shape = (N, T, F)
    => 計算
    """
    N, T, _ = X.shape
    diffs = np.diff(X, axis=1)
    stds = np.std(diffs, axis=1)
    return np.repeat(stds[:, None, :], T, axis=1)



def det_rest(vel: np.array, speed_th = 0.05):
    """
    microment / rest 判斷:
    """
    # 1 取出 nose velocity：vel shape=(N,T,16)，前兩個欄位就是 nose 的 dx,dy
    nose_vel = vel[:, :, 0:2]               # (N, T, 2)
    # 2 算速度大小 magnitude
    nose_speed = np.linalg.norm(nose_vel, axis=2)  # (N, T)
    # 3 threshold（二值化）→ rest=0, micro=1
    rest_micro = (nose_speed > speed_th).astype(np.float32)  # (N, T)
    return rest_micro[..., None]   # 加一個 channel 維度，變成 (N, T, 1)   

def compute_box_features(box_arr: np.ndarray, orig_size: tuple[int,int]):
    """
    將box資訊轉成 center_x, center_y, area, 位移量平方, 面積變化量。

    參數
    -------
    box_arr : np.ndarray, shape = (N, T, 4)
        每格順序為 [box_x, box_y, box_w, box_h]

    回傳
    -------
    np.ndarray, shape = (N, T, 5)
        [wh_ratio, disp2, dc] #
    """
    
    N, T, F = box_arr.shape
    x  = box_arr[..., 0]
    y  = box_arr[..., 1]
    w  = box_arr[..., 2]
    h  = box_arr[..., 3]

    # 1. 中心座標與面積
    cx   = (x + w / 2.0) / orig_size[0]                   # (N, T)
    cy   = (y + h / 2.0)   / orig_size[1]                # (N, T)
    area = w * h         / (orig_size[0]*orig_size[1] + 1e-6)               # (N, T)   

    # 2. 計算寬高比 wh_ratio: (N,T) → (N,T,1)
    wh_ratio = np.expand_dims(w/(h + 1e-6), axis=2)         # (N, T, 1)



    # 3. 中心位移量  (N, T) → (N, T, 1)
    dx    = np.diff(cx,    axis=1, prepend=cx[:, :1])   # diff: A[i]-A[i-1]，在T上  prepend: 在前面補值 (跟本的shape一樣) e.g. [1,2,4].diff -> [1,2] -> prepend: [1,1,2]
    dy    = np.diff(cy,    axis=1, prepend=cy[:, :1])

    disp2 = np.expand_dims((dx**2 + dy**2), axis=2)                        # ‖Δc‖²: (N, T, 1) 逐幀位移總和

    dc    = np.abs(cx[:, -1] - cx[:, 0])           #  (N,) 起始結束中心點位移量
    dc    = np.repeat(dc[:,None], repeats=T, axis=1)
    dc    = dc[:,:,None]

    # da    = np.diff(area,  axis=1, prepend=area[:,:1])
    return np.concatenate([ disp2], axis=-1)  # (N, T, 3)


# for train batch inference
def build_features(X, y, orig_size, vel_delta, smooth_window_length, polyorder, le=None, stats=None):
    """
    inpit:
    X: (N, T, 20)
    20: [box_x,box_y,box_w,box_h, kpt0_x,kpt0_y,...,kpt7_x,kpt7_y]:  4 + 16 = 20

    output: 
    X_full  : feature_vector
    y_enc   : map到idx的class
    le      : sklearn.LabelEncoder，None→fit
    stats   : dict{'mins','maxs'} 來自 Train；None→自行計算

    """
    # 0) smooth
    X_kpts = X[:, :, 4:] 
    X_boxs = X[:, :, :4]
    # X_kpts = smooth_kpts(X_kpts, smooth_window_length, polyorder)

    # 1) 編碼 y
    # ---- 1. Label ----
    if le is None:
        le = LabelEncoder().fit(y)
    y_enc = le.transform(y)


    # 2) 原圖相對 + MinMax + velocity/acc + cos + space + dir + std
    orig_w, orig_h = orig_size
    X_rel = X_kpts.copy()
    X_rel[:, :, 0::2] /= orig_w
    X_rel[:, :, 1::2] /= orig_h

    if stats is None:
        mins = X_kpts.min((0,1));  maxs = X_kpts.max((0,1))
        stats = dict(mins=mins, maxs=maxs)
    # X_mM = (X_kpts - stats['mins']) / (stats['maxs'] - stats['mins'] + 1e-6)

    vel, acc = compute_velocity_acc(X_rel, delta=vel_delta)

    N, T, _ = X_rel.shape
    kpt_seq = X_rel.reshape(N, T, 8, 2)

    cos_feats = np.stack([compute_cos_np(kpt_seq[i]) for i in range(N)], axis=0)
    space_feats = compute_space_distances(X_rel) 
    std_move = compute_std_movement(X)

    dir_feats   = compute_direction_unit(vel)
    std_feats   = compute_speed_std(vel)
    rest_micro = det_rest(vel)
    # boxs
    box_feats = compute_box_features(X_boxs, orig_size)


    X_full = np.concatenate(
        [
            X_rel,    # 16
            # X_mM,     # 16
            vel,      # 16
            acc,      # 16
            cos_feats,# 1
            space_feats,# 2
            # std_move,  #20
            dir_feats,# 2
            std_feats,# 1
            # rest_micro,# 1
            box_feats  # 3
        ],
        axis=2
    )
    return X_full, y_enc, le, stats



def Encode_labels(y): 
    """
    把['行為'] maps 到 [0~7]
    """    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le #return le 之後才能做inverse: int -> str