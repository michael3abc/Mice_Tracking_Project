import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# === 工具函式 ===

def add_bout_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    標記每支影片內的連續事件 (bout) 編號
    輸入 df 需包含 ['video_id','behavior'] 兩欄
    回傳時會新增欄位 ['bout_trigger','bout_id','group_id']
    """
    df = df.copy()
    df["bout_trigger"] = (df["behavior"] != df["behavior"].shift()).astype(int)
    df["bout_id"] = df.groupby("video_id")["bout_trigger"].cumsum()
    df["group_id"] = df["video_id"].astype(str) + "_" + df["bout_id"].astype(str)
    return df

def sample_bouts_by_behavior(df: pd.DataFrame,
                             n_bouts: int = 50,
                             random_state: int = 42) -> list:
    """
    隨機對每種行為抽取固定數量的 bouts（依 group_id 分組）。

    參數:
    - df: 需包含 ['video_id','frame','behavior'] 的 DataFrame
    - n_bouts: 每種行為欲抽樣的 bout 數量
    - random_state: 隨機種子

    回傳:
    - selected_group_ids: 被抽中的 group_id 清單
    """
    # 先標記 bout_ids
    df_b = add_bout_ids(df)[['group_id', 'behavior']].drop_duplicates()

    np.random.seed(random_state)
    sampled = []
    for beh, grp in tqdm(df_b.groupby('behavior'),
                         desc="抽樣各行為 bouts"):
        cnt = len(grp)
        if cnt >= n_bouts:
            chosen = grp.sample(n=n_bouts,
                                replace=False,
                                random_state=random_state)
        else:
            print(f"警告：行為 '{beh}' 只有 {cnt} 段，全部保留。")
            chosen = grp
        sampled.append(chosen['group_id'])
    return pd.concat(sampled).tolist()

def evaluate_sampled_bouts_df(df_cmp: pd.DataFrame,
                              selected_group_ids: list) -> None:
    """
    在 df_cmp 上加上 bout 標籤，過濾出 selected_group_ids，
    並計算 classification_report & confusion_matrix。

    參數:
    - df_cmp: 包含 ['video_id','frame','gt_behavior','pred_behavior'] 的 DataFrame
    - selected_group_ids: 欲評估的 group_id 清單
    """
    df = df_cmp.copy()
    # 新增行為欄位以供 add_bout_ids
    df['behavior'] = df['gt_behavior']
    df = add_bout_ids(df)

    df_sel = df[df['group_id'].isin(selected_group_ids)]
    if df_sel.empty:
        raise ValueError("沒有符合條件的 frames，請檢查 selected_group_ids！")

    y_true = df_sel['gt_behavior'].astype(str)
    y_pred = df_sel['pred_behavior'].astype(str)
    labels = sorted(y_true.unique())

    print("=== [Sampled Bouts 評估] Classification Report ===")
    print(classification_report(y_true, y_pred,
                                labels=labels,
                                target_names=labels,
                                digits=4,
                                zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    print("=== [Sampled Bouts 評估] Confusion Matrix ===")
    print(df_cm)


# === 主程式 ===

def main(start_video=1, end_video=12,
         window=96, sample_per_behavior=50):
    """
    完整流程：
    1. 讀取 GT & 預測結果，對齊成 df_all_compare
    2. 抽樣每種行為固定數量 bouts
    3. 只對抽樣後的 bouts 做評估
    """
    all_cmp = []

    # 1. 讀取並對齊 frame-level
    for vid in range(start_video, end_video+1):
        gt_path   = fr"data_prediction\dataset\gt_behaviors\gt_behavior_mice{vid}.csv"
        pred_path = fr"data_prediction\prediction_results\2_behavios\ST-TR\behaviors_mice{vid}.csv"

        # 1.1 讀 GT
        df_gt = pd.read_csv(gt_path)[["frame","behavior"]].rename(
            columns={"behavior":"gt_behavior"})
        df_gt["frame"] = df_gt["frame"].astype(int)
        df_gt["video_id"] = vid

        # 1.2 讀 Pred，並補上 video_id
        df_pred = pd.read_csv(pred_path)[["frame","behavior"]].rename(
            columns={"behavior":"pred_behavior"})
        df_pred["frame"] = df_pred["frame"].astype(int)
        df_pred["video_id"] = vid
        df_pred = df_pred.sort_values("frame").reset_index(drop=True)

        # 1.3 處理延遲：前 window-1 幀無預測
        df_pred.loc[:window-2, "pred_behavior"] = None

        # 1.4 merge on video_id & frame
        df_cmp = pd.merge(df_gt, df_pred,
                          on=["video_id","frame"],
                          how="inner")
        all_cmp.append(df_cmp)

    df_all = pd.concat(all_cmp, ignore_index=True)
    print(f"✅ 總共比對到 {len(df_all)} 幀")

    # 2. 抽樣 bouts（以 GT 行為作分群）
    df_for_sampling = df_all[['video_id','frame','gt_behavior']].rename(
        columns={"gt_behavior":"behavior"})
    selected_ids = sample_bouts_by_behavior(
        df_for_sampling,
        n_bouts=sample_per_behavior,
        random_state=42
    )
    print(f"✅ 抽樣後共 {len(selected_ids)} 段 bouts")

    # 3. 評估抽樣後的結果
    evaluate_sampled_bouts_df(df_all, selected_ids)


if __name__ == "__main__":
    main(start_video=1,
         end_video=12,
         window=96,
         sample_per_behavior=50)
