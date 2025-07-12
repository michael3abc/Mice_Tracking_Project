import os
import re
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# === 設定路徑 ===


# === 讀取 ground truth 範圍 → 展開成每幀 ===
def load_gt_frame_labels(txt_path):
    gt_rows = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            m = re.match(r'frame:\s*(\d+)-(\d+)\s+(\w+)', line)
            if m:
                start, end, label = int(m.group(1)), int(m.group(2)), m.group(3)
                for fid in range(start, end + 1):
                    gt_rows.append((fid, label))
    return pd.DataFrame(gt_rows, columns=["frame", "gt_behavior"])

# === 主程式 ===
def main():
    all_compare = []
    for i in range(1,6):
        video_id = i
        txt_path = fr"data_prediction\dataset\behaviors\mice{i}.txt"
        pred_csv = fr"data_prediction\prediction_results\2_behavios\behavios_{i}.csv"

        # 1. 讀入 ground truth & 預測
        df_gt = load_gt_frame_labels(txt_path)
        df_pred = pd.read_csv(pred_csv)[["frame", "behavior"]].rename(columns={"behavior": "pred_behavior"})
        df_pred["frame"] = df_pred["frame"].astype(int)

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

    print("\n [全部影片綜合] Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    labels = sorted(set(y_true) | set(y_pred))  # union 所有出現過的 label
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\n [全部影片綜合] Confusion Matrix:")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    # 4. 比例版 Confusion Matrix
    cm_ratio = cm.astype(np.float32)
    row_sums = cm_ratio.sum(axis=1, keepdims=True)

    zero_rows = (row_sums == 0).flatten()
    cm_ratio[zero_rows, :] = 0
    cm_ratio[~zero_rows, :] = cm_ratio[~zero_rows, :] / row_sums[~zero_rows]

    # 5. 畫圖
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_ratio, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Normalized Confusion Matrix (Video {1}~{5})")
    plt.xlabel("Predicted Behavior")
    plt.ylabel("Ground Truth Behavior")
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()
