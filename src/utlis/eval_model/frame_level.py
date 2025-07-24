import os
import re
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



# === 主程式 ===
def main(start_video=1, end_video = 12):
    all_compare = []
    window = 96
    for i in range(start_video, end_video+1):
        video_id = i
        gt_csv = fr"data_prediction\dataset\gt_behaviors\gt_behavior_mice{i}.csv"
        pred_csv = fr"data_prediction\prediction_results\2_behavios\ST-TR\behaviors_mice{i}.csv"    
         


        # 1. 讀入 ground truth & 預測
        df_gt = pd.read_csv(gt_csv)[["frame", "behavior"]].rename(columns={"behavior": "gt_behavior"})
        df_gt["frame"] = df_gt["frame"].astype(int)

        df_pred = pd.read_csv(pred_csv)[["frame", "behavior"]].rename(columns={"behavior": "pred_behavior"})
        df_pred = df_pred.sort_values("frame").reset_index(drop=True)

        # 將前 window-1 幀的預測都設成 NaN（或 None）: 因為預測延遲
        df_pred.loc[:window-2, ["behavior","top1","prob1","top2","prob2","top3","prob3"]] = None

        # 2. 對齊
        df_compare = pd.merge(df_gt, df_pred, on="frame", how="inner")
        df_compare['video_id'] = video_id
        all_compare.append(df_compare)

        print(f"✅ 成功比對 {len(df_compare)} 幀")

    
    df_all = pd.concat(all_compare, ignore_index=True)

    # 3. 指標
    y_true = df_all["gt_behavior"].astype(str)
    y_pred = df_all["pred_behavior"].astype(str)

    acc = (y_true == y_pred).mean()
    print(f"\n=== [全部影片綜合] Frame Accuracy: {acc:.4%}")

    print("\n[全部影片綜合] Classification Report:")
    labels = df_all["gt_behavior"].unique().tolist()
    print(classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=labels,
        digits=4,
        zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\n[全部影片綜合] Confusion Matrix:")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    # 4. 比例版 Confusion Matrix
    cm_ratio = cm.astype(float)
    row_sums = cm_ratio.sum(axis=1, keepdims=True)
    cm_ratio[row_sums.squeeze()==0, :] = 0
    cm_ratio[row_sums.squeeze()!=0, :] /= row_sums[row_sums.squeeze()!=0]

    # 5. 畫圖
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_ratio, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Normalized Confusion Matrix (Video {start_video}~{end_video})")
    plt.xlabel("Predicted Behavior")
    plt.ylabel("Ground Truth Behavior")
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main(start_video=1, end_video=12)
