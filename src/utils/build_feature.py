# src/data_utils.py
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.signal import savgol_filter
from joblib import Parallel, delayed
# from scipy.spatial import ConvexHull

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

class KinematicFeatures:
    """
    計算速度、加速度與第三階導數 (jerk)。
    """
    @staticmethod
    def compute( X_rel, delta):    
        """
        X: shape (N, T, D) → e.g. (樣本數, 幀數, 特徵維度*16 kpts)
        delta: 幀差距，例如 delta=3 表示三幀差分 (愈大表示一次看的幀數愈長)

        回傳: (vel, acc)，形狀 (N, T, D)，只是前面 delta 幀是 0
        """
        #v_t = x_t - x_(t-det)
        #acc同理
        vel  = np.zeros_like(X_rel)
        acc  = np.zeros_like(X_rel)
        jerk = np.zeros_like(X_rel)

        d = delta
        vel[:, d:, :]   = X_rel[:, d:, :]   - X_rel[:, :-d, :]
        acc[:, d:, :]   = vel[:, d:, :]     - vel[:, :-d, :]
        # jerk[:, d:, :]  = acc[:, d:, :]     - acc[:, :-d, :]
        return vel, acc
    


class GeometryFeatures:
    @staticmethod
    def compute_cos_np(kpts):
        """
        計算三種 cos 特徵：
        1. 鼻子→身體 與 身體→尾根 的向量夾角 cos 值
        2. 鼻子→身體 與 水平向量 (1,0) 的夾角 cos 值
        3. 身體→尾根 與 水平向量 (1,0) 的夾角 cos 值

        參數
        ----
        kpts : np.ndarray, shape (seq_len, 8, 2)
            每一 frame 的 8 個關鍵點 (x,y)

        回傳
        ----
        cos_feats : np.ndarray, shape (seq_len, 3)
            三個 cos 特徵，依序為 [cos1, cos2, cos3]
        """
        # 拆出 nose, body, tail_base
        nose = kpts[:, 0, :]    # [seq_len,2]
        body = kpts[:, 1, :]    # [seq_len,2]
        tail = kpts[:, 2, :]    # [seq_len,2]

        v1 = nose - body        # [seq_len,2]
        v2 = tail - body        # [seq_len,2]
        eps = 1e-6

        # 1. nose - body - tail:  cos
        dot12    = np.sum(v1 * v2, axis=1, keepdims=True)                  # [seq_len,1]
        norm1   = np.linalg.norm(v1, axis=1, keepdims=True) 
        norm2   = np.linalg.norm(v2, axis=1, keepdims=True)  # [seq_len,1]

        cos1 = dot12 / (norm1* norm2 + eps)  # [seq_len,1]

        # # 2. nose-body - ground (水平地面): cosθ = dx / ||v1||
        # cos2    = v1[:,:1] / (norm1 + eps)  # [seq_len,1]

        # # 3. body-tail - ground (水平地面)
        # cos3    = v2[:,:1] / (norm2 + eps)  # [seq_len,1]


        return np.concatenate([cos1,
                            #    cos2, 
                            #    cos3
                            ], axis = 1)

    @staticmethod
    def _convex_hull_area_chain(pts: np.ndarray):
        """
        Monotone Chain 凸包面積 (Shoelace 公式)
        pts: ndarray (K,2)
        回傳: area (float)
        """
        P = pts[np.lexsort((pts[:,1], pts[:, 0]))]
        if P.shape[0] < 3:
            return 0.0
        
        def _cross(o,a,b):
            """
            計算三點 O, A, B 的向量 OA 與 OB 的外積值：
            >0 表示逆時針左轉
            <0 表示順時針右轉
            0 表示三點共線
            """
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        
        lower = []
        for p in P:
            while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(tuple(p))

        upper = []
        for p in P[::-1]:
            while len(upper)>=2 and _cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(tuple(p))
        hull = lower[:-1] + upper[:-1]  # 去掉重複端點

        x, y = zip(*hull)   # e.g. hull = [(x0,y0),(x1,y1),(x2,y2)]  => zip(*hull) → x = (x0,x1,x2), y = (y0,y1,y2)
        x = np.array(x + x[:1])   # (x0, x1, x2, x0)
        y = np.array(y + y[:1])   # (y0, y1, y2, y0)

        #    A = 1/2 * | Σ ( x_i * y_{i+1} - x_{i+1} * y_i ) |
        area = 0.5 * np.abs(
            np.dot(x[:-1], y[1:])   # Σ x_i * y_{i+1}
        - np.dot(x[1:],  y[:-1])  # Σ x_{i+1} * y_i
        )

        return float(area)
    @staticmethod
    def batch_hull_for_one(sample_kpts):
        # sample_kpts: shape (T,8,2)
        T = sample_kpts.shape[0]
        out = np.zeros((T,1), dtype=np.float32)
        for t in range(T):
            out[t,0] = GeometryFeatures._convex_hull_area_chain(sample_kpts[t])
        return out


class SpaceFeature:
    @staticmethod
    def compute_space_distances(X_rel):
        """
        X_rel: np.ndarray, shape = (N, seq_len, 16)   # 8 個 keypoint 的 x,y
        回傳: np.ndarray, shape = (N, seq_len, 3)     # nose ↔ tail, front ↔ rear
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
    @staticmethod
    def compute_direction_unit(vel):
        """
        vel: np.ndarray, shape = (N, seq_len, 16)  
            # 每个 window 有 seq_len 帧，每帧 16 维 velocity (8 个 keypoint × 2 维)
        回傳:
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
    @staticmethod
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
    @staticmethod
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
    從 box 資訊計算中心位移平方特徵 (disp2)。

    參數
    -------
    box_arr : np.ndarray, shape = (N, T, 4)
        每格順序為 [box_x, box_y, box_w, box_h]

    回傳
    -------
    np.ndarray, shape = (N, T, 1)
        [disp2]  # 目前僅保留中心位移平方特徵
    """
    
    N, T, F = box_arr.shape
    x  = box_arr[..., 0]
    y  = box_arr[..., 1]
    w  = box_arr[..., 2]
    h  = box_arr[..., 3]

    # 1. 中心座標與面積
    cx   = (x + w / 2.0) / orig_size[0]                         # (N, T)
    cy   = (y + h / 2.0) / orig_size[1]                         # (N, T)
    # area = w * h         / (orig_size[0]*orig_size[1] + 1e-6)   # (N, T)   

    # 2. 計算寬高比 wh_ratio: (N,T) → (N,T,1)
    # wh_ratio = np.expand_dims(w/(h + 1e-6), axis=2)             # (N, T, 1)



    # 3. 中心位移量  (N, T) → (N, T, 1)
    dx    = np.diff(cx,    axis=1, prepend=cx[:, :1])   # diff: A[i]-A[i-1]，在T上  prepend: 在前面補值 (跟本的shape一樣) e.g. [1,2,4].diff -> [1,2] -> prepend: [1,1,2]
    dy    = np.diff(cy,    axis=1, prepend=cy[:, :1])

    disp2 = np.expand_dims((dx**2 + dy**2), axis=2)                        # ‖Δc‖²: (N, T, 1) 逐幀位移總和

    # dc    = np.abs(cx[:, -1] - cx[:, 0])           #  (N,) 起始結束中心點位移量
    # dc    = np.repeat(dc[:,None], repeats=T, axis=1)
    # dc    = dc[:,:,None]

    # da    = np.diff(area,  axis=1, prepend=area[:,:1])
    return np.concatenate([
                            # area,
                            # wh_ratio,
                            disp2,
                            # dc               
                           ], axis=-1)  # (N, T, 1)


# for train / inference
def build_features(X, y=None, orig_size=None, vel_delta=3, smooth_window_length=3, polyorder=2, le=None, stats=None):
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
    # 0. smooth
    X_kpts = X[:, :, 4:] 
    X_boxs = X[:, :, :4]
    # X_kpts = smooth_kpts(X_kpts, smooth_window_length, polyorder)

    # 1. 編碼 y
    if y is not None:
        if le is None:
            le = LabelEncoder().fit(y)
        y_enc = le.transform(y)
    else:
        y_enc = None



    # 2.1. relative kpts 
    orig_w, orig_h = orig_size
    X_rel = X_kpts.copy()
    X_rel[:, :, 0::2] /= orig_w
    X_rel[:, :, 1::2] /= orig_h

    # # 2.2. MinMax kpts
    # if stats is None:
    #     mins = X_kpts.min((0,1));  maxs = X_kpts.max((0,1))
    #     stats = dict(mins=mins, maxs=maxs)
    # X_mM = (X_kpts - stats['mins']) / (stats['maxs'] - stats['mins'] + 1e-6)

    # 2.3. velocity/acc
    vel, acc = KinematicFeatures.compute(X_rel, delta=vel_delta)

    N, T, _ = X_rel.shape
    kpt_seq = X_rel.reshape(N, T, 8, 2)

    # 2.4. cos
    cos_feats = np.stack([GeometryFeatures.compute_cos_np(kpt_seq[i]) for i in range(N)], axis=0)
    # 2.5. space distance
    space_feats = SpaceFeature.compute_space_distances(X_rel) 
    # 2.6. std movement
    # std_move = SpaceFeature.compute_std_movement(X_rel)
    # 2.7. nose direction
    dir_feats   = SpaceFeature.compute_direction_unit(vel)
    # 2.8. std speed
    speed_std   = SpaceFeature.compute_speed_std(vel)
    # rest_micro = det_rest(vel)
    # 2.9. bounding boxs
    box_feats = compute_box_features(X_boxs, orig_size)

    # # 2.10. convex area  
    # results = Parallel(n_jobs=-1, backend="loky")(
    #     delayed(GeometryFeatures.batch_hull_for_one)(kpt_seq[i]) for i in range(N)
    # )
    # convex_area = np.stack(results, axis=0)  # (N, T, 1)
    # delta_convex  = np.diff(convex_area, axis=1, prepend=convex_area[:,:1,:]) # 在T維度上做差分，補0


    X_full = np.concatenate(
        [
            X_rel,        # 16
            vel,          # 16
            acc,          # 16
            cos_feats,    # 1
            space_feats,  # 2
            # std_move,     #20
            dir_feats,    # 2
            speed_std,    # 1
            # rest_micro,   # 1
            box_feats,    # 1
            # convex_area,   # 1 
            # delta_convex
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
