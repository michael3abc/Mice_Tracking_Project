import os
import re
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# === 設定路徑 ===
video_id = 1
gt_behavior_path = fr"data_prediction\dataset\kpt_gt_behavior\kpt_gt_behavior.csv"
pred_csv = fr"data_prediction\prediction_results\2_behavios\behavior_predictions.csv"

# === 讀取 ground truth 範圍 → 展開成每幀 ===
def load_gt_frame_labels(gt_behavior_path):
    gt_rows = []
    with open(gt_behavior_path, "r", encoding="utf-8") as f:
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
    # 1. 讀入 ground truth & 預測
    df_gt = pd.read_csv(gt_behavior_path)[["frame", "behavior"]].rename(columns={"behavior" : "gt_behavior"})

    df_pred = pd.read_csv(pred_csv)[["frame", "behavior"]].rename(columns={"behavior": "pred_behavior"})
    df_pred["frame"] = df_pred["frame"].astype(int)

    # 2. 對齊
    df_compare = pd.merge(df_gt, df_pred, on="frame", how="inner")
    df_compare = df_compare.fillna("unknown")  # 防止 NaN

    df_compare["gt_behavior"] = df_compare["gt_behavior"].astype(str)
    df_compare["pred_behavior"] = df_compare["pred_behavior"].astype(str)

    print(f"✅ 成功比對 {len(df_compare)} 幀")

    # 3. 指標
    y_true = df_compare["gt_behavior"]
    y_pred = df_compare["pred_behavior"]

    acc = (y_true == y_pred).mean()
    print(f"🎯 Frame Accuracy: {acc:.4%}")

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, digits=4))

    labels = sorted(set(df_compare["gt_behavior"]) | set(df_compare["pred_behavior"]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    print("\n🧩 Confusion Matrix:")
    print(pd.DataFrame(cm, index=labels, columns=labels))

    # 4. 比例版混淆矩陣
    cm_ratio = cm.astype(np.float32)
    row_sums = cm_ratio.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).flatten()
    cm_ratio[zero_rows, :] = 0
    cm_ratio[~zero_rows, :] = cm_ratio[~zero_rows, :] / row_sums[~zero_rows]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_ratio, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Normalized Confusion Matrix (Video {video_id})")
    plt.xlabel("Predicted Behavior")
    plt.ylabel("Ground Truth Behavior")
    plt.tight_layout()
    plt.show()

    # # 儲存圖片
    # out_img = os.path.join(os.path.dirname(pred_csv), f"confusion_matrix_{video_id}.png")
    # plt.savefig(out_img)
    # print(f"🖼️ 混淆矩陣圖片已儲存：{out_img}")
    # plt.close()


    # # 4. 輸出對照表（可選）
    # out_csv = os.path.join(os.path.dirname(pred_csv), f"compare_behavior_only_{video_id}.csv")
    # df_compare.to_csv(out_csv, index=False)
    # print(f"\n📁 輸出：{out_csv}")

if __name__ == "__main__":
    main()
