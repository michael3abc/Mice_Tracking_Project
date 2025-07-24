import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



def main(start_video=1, end_video=12, window_size=96, stride=8):
    all_true = []
    all_pred = []

    for i in range(start_video, end_video + 1):
        video_id = i
        gt_csv   = fr"data_prediction\dataset\gt_behaviors\gt_behavior_mice{video_id}.csv"
        pred_csv = fr"data_prediction\prediction_results\2_behavios\ST-TR\behaviors_mice{video_id}.csv"

        # 1. 讀入 frame-level GT 與 Pred
        df_gt = pd.read_csv(gt_csv)[["frame", "behavior"]].rename(columns={"behavior": "gt_behavior"})
        df_pred = pd.read_csv(pred_csv)[["frame", "behavior"]].rename(columns={"behavior": "pred_behavior"})
        df_pred = df_pred.sort_values("frame").reset_index(drop=True)

        # 建立 frame → pred_behavior 的對照 dict
        pred_map = dict(zip(df_pred["frame"], df_pred["pred_behavior"]))

        # 2. 生成滑動視窗
        y_seq = df_gt["gt_behavior"].values
        n_frames = len(y_seq)
        starts = list(range(0, n_frames - window_size + 1, stride))

        # 每個視窗的 GT 取最後一幀；Pred 也取最後一幀的預測
        for s in starts:
            end_idx = s + window_size - 1
            gt_lbl = y_seq[end_idx]
            frame_num = df_gt["frame"].iat[end_idx]
            pred_lbl = pred_map.get(frame_num, None)
            
            all_true.append(gt_lbl)
            all_pred.append(pred_lbl)

        print(f"✅ Video {video_id}: 共 {len(starts)} 個視窗")

    # 3. 合併所有視窗後，計算指標
    y_true = np.array(all_true, dtype=object)
    y_pred = np.array(all_pred, dtype=object)

    # Window-level Accuracy
    acc = np.mean(y_true == y_pred)
    print(f"\n=== Window-level Accuracy: {acc:.4%}")

    # Classification Report
    labels = pd.unique(y_true)
    print("\n=== Classification Report ===")
    print(classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=labels,
        digits=4,
        zero_division=0
    ))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_ratio = cm.astype(float)
    row_sums = cm_ratio.sum(axis=1, keepdims=True)
    # 避免除以 0
    mask = row_sums.squeeze() != 0
    cm_ratio[mask] /= row_sums[mask]

    # 4. 畫出比例版 (normalized) Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_ratio, annot=True, fmt=".2f", cmap="YlGnBu",
        xticklabels=labels, yticklabels=labels
    )
    plt.title(f"Normalized Confusion Matrix (Windows {start_video}–{end_video})")
    plt.xlabel("Predicted Behavior")
    plt.ylabel("Ground-Truth Behavior")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(start_video=1, end_video=12, window_size=96, stride=24)
