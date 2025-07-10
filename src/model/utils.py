import pandas as pd
import numpy as np
from sklearn.utils import resample

def Check_Class(df):
    print("剩下的行為種類：", df['behavior'].unique())
    print("每種行為的樣本數：\n", df['behavior'].value_counts())
    
'''
剩下的行為種類： ['micromovement' 'walk' 'rear' 'eat' 'groom' 'drink' 'hang' 'rest']
每種行為的樣本數：
 behavior
micromovement    116967
groom            115687
walk              61074
rear              57113
eat               43393
hang              36178
rest              19042
drink              1203

由於drink太少 => 省略
'''


'''
df(乾淨、未打亂的原始資料)
     ↓
per-video sliding → 得到很多 window: (X, y)
     ↓
再丟進 Balance_data() → 根據 y 來平衡 sample 數

'''

def Balance_data(df):
    from sklearn.utils import resample #用來取樣
    min_count = df["behavior"].value_counts().min()

    balanced_dfs = []
    for behavior in df['behavior'].unique():
        df_behavior = df[df['behavior'] == behavior]
        df_down = resample(df_behavior,
                               replace= False, #不可重複
                               n_samples= min_count, #下取樣
                               random_state = 42)       
        balanced_dfs.append(df_down)       
    
    df_balanced = pd.concat(balanced_dfs).sort_values(by = ['video_id', 'frame']).reset_index(drop=True)
    print("✅ 平衡後行為數量：")
    print(df_balanced['behavior'].value_counts())                 
    return df_balanced


# def Sliding_windows(df):  
    """
    讀 keypoints & 行為，同時蒐集每段 window 的 bbox center。
    參數:
      df: 包含 kpt*_x, kpt*_y, behavior, video_id, frame
      box_df: DataFrame，必須和 df 同索引，裡面要有 ['center_x','center_y']
              如果不傳，就會填 0
    回傳:
      X: np.ndarray, (N_windows, window_size, n_features)
      y: np.ndarray, (N_windows,)
      centers: np.ndarray, (N_windows, n_features) – 每 window 用同一 center_x,center_y 重複 fill
    """
    window_size = 32   #一次要看多少幀
    stride = 4         #一次要跳過幾幀
                     
    feature_cols = [c for c in df.columns if c.startswith('kpt')] 
    # e.g. ['kpt0_x','kpt0_y',...,'kpt7_x','kpt7_y']

    X_list = [] #kpts
    y_list = [] #行為

    for vid in df['video_id'].unique():
        df_vid = df[df['video_id'] == vid].reset_index(drop = True)       

        for start_idx in range(0, len(df_vid)-window_size+1, stride):
            window = df_vid.iloc[start_idx : start_idx + window_size]

            # 取出特徵矩陣 (32,16)
            X = window[feature_cols].values  #只取kpts => 把他轉array 
            #取window最後一幀的行為
            y = window['behavior'].iloc[-1]
            X_list.append(X)
            y_list.append(y)

    X = np.stack(X_list) # shape = (N, 32, 16): 把X_list裡的矩陣"堆疊"起來
    y = np.array(y_list) # shape = (num_samples, labels)
    return X, y

def Balance_windows(X, y, random_state = 42):

    #1. 找出class跟最小樣本數
    classes, counts = np.unique(y, return_counts=True)
    # ex. classes = ['eat','rear','walk']
    #     counts  = [2,    1,     2]

    min_count = counts.min()

    idxs_balanced = []

    #2. 針對每個類別 => resample
    for cls in classes:
        cls_idxs = np.where(y == cls)[0] #True會在第一個
        '''
        y = np.array(['eat','walk','eat','rear'])
        mask = (y == 'eat')           # array([True,False,True,False])
        idxs = np.where(mask)[0]      # array([0,2])
        '''
        sel = resample(cls_idxs,
                       replace=False,
                       n_samples=min_count,
                       random_state=random_state)
        
        idxs_balanced.extend(sel.tolist()) #轉回python list 
        
    # 3. 排序（可選，讓結果更一致）
    idxs_balanced = sorted(idxs_balanced)

    #4. 根據平衡結果切出X, y
    X_balanced = X[idxs_balanced]
    y_balanced = y[idxs_balanced]

    # print(" 平衡後每個類別樣本數：")
    # for cls in classes:
    #     print(f"  {cls}: {sum(y_balanced == cls)}")
    return X_balanced, y_balanced


def Sliding_windows(df, window_size = 32, stride = 4):  
    window_size = window_size   #一次要看多少幀
    stride = stride         #一次要跳過幾幀
                     
    feature_cols = [c for c in df.columns if c.startswith('kpt')] 
    # e.g. ['kpt0_x','kpt0_y',...,'kpt7_x','kpt7_y']

    X_list = [] #kpts
    y_list = [] #行為
    for vid in df['video_id'].unique():
        df_vid = df[df['video_id'] == vid].reset_index(drop = True)

        for start_idx in range(0, len(df_vid)-window_size+1, stride):
            window = df_vid.iloc[start_idx : start_idx + window_size]

            # 取出特徵矩陣 (32,16)
            X = window[feature_cols].values  #只取kpts => 把他轉array 
            #取window最後一幀的行為
            y = window['behavior'].iloc[-1]

            X_list.append(X)
            y_list.append(y)

    X = np.stack(X_list) # shape = (N, 32, 16): 把X_list裡的矩陣"堆疊"起來
    y = np.array(y_list) # shape = (num_samples, labels)
    # print("總樣本數:", X.shape[0])
    # print("每個樣本維度:", X.shape[1:], "＝32 幀 × 16 維")
    # print("Labels shape:", y.shape)
    # print("行為分佈:\n", pd.Series(y).value_counts())
    return X, y



from sklearn.preprocessing import LabelEncoder
def Encode_labels(y): #把['行為'] maps 到 [0~7]    

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le #return le 之後才能做inverse: int -> str



import numpy as np

def impute_windows(X):
    """
    X: np.ndarray, shape = (N_windows, window_size, n_features)
    回傳同樣 shape，但 NaN 已經補好的 X_imp
    """
    N, W, F = X.shape
    X_imp = np.empty_like(X)
    
    for i in range(N):
        Xi = X[i]  # (W, F)
        for j in range(F):
            col = Xi[:, j]
            nans = np.isnan(col)
            if nans.all():
                # 整個序列都沒值，就填 0
                col[:] = 0.0
            elif nans.any():
                # 用 np.interp 在時間維度做線性插值
                idx = np.arange(W)
                valid = ~nans
                col[nans] = np.interp(idx[nans], idx[valid], col[valid])
            # else: 沒 NaN，不動
            X_imp[i, :, j] = col

    return X_imp

def compute_velocity_acc(X, delta=1):
    """
    X: shape (N, T, D) → e.g. (樣本數, 幀數, 特徵維度*16 kpts)
    delta: 幀差距，例如 delta=3 表示三幀差分 (愈大表示一次看的幀數愈長)

    回傳: (vel, acc)，形狀都一樣 (N, T, D)，只是前面 delta 幀是 0
    """
    #v_t = x_t - x_(t-det)
    #acc同理
    vel = np.zeros_like(X)
    acc = np.zeros_like(X)
    vel[:, delta:, :] = X[:, delta:, :] - X[:, :-delta, :]
    acc[:, delta:, :] = vel[:, delta:, :] - vel[:, :-delta, :]
    return vel, acc

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




if __name__ == '__main__':
    # 只有在你直接執行 data_processing.py 才會跑這裡
    df = pd.read_csv(r"C:\Users\micha\Desktop\python_workspace\YOLO Mice Project\src\Pose_to_behavior\dataset\keypoints_with_behavior.csv")
    df = df[(df.behavior!='unknown')&(df.behavior!='drink')]
    df = df.sort_values(['video_id','frame']).reset_index(drop=True)
    Check_Class(df)
    X, y = Sliding_windows(df)
    print("Sliding:", X.shape, y.shape)
    Xb, yb = Balance_windows(X, y)
    print("Balanced:", Xb.shape, pd.Series(yb).value_counts())
    ye, le = Encode_labels(yb)
    print("Encode sample:", yb[:5], ye[:5], dict(zip(le.classes_, le.transform(le.classes_))))

    print(Xb, yb)