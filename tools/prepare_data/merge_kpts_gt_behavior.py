import re #正規表達
import pandas as pd
import os
# 用來把pred_kpts + gt_behavior讓模型學習

start = 1
end = 12

if __name__ == '__main__':
    # 1. 讀行為標籤
    gt_csv_folder = r"C:\Users\micha\Desktop\Mice_tracking_project\data_prediction\dataset\gt_behaviors"
    # 2. 讀keypoints
    kpt_folder = r"data_prediction\prediction_results\1_keypoints\yolov11"

    df_kpt_with_labeled = []
    for i in range(start, end+1):
        df_gt_behavior = pd.read_csv(os.path.join(gt_csv_folder, f"gt_behavior_mice{start}.csv"))
        df_kpt_pred = pd.read_csv(os.path.join(kpt_folder, f"keypoints_mice{end}.csv"))
       
        if "video" not in df_kpt_pred.columns:
            df_kpt_pred["video"] = i
        # 合併：根據 video + frame
        df_labeled = pd.merge(df_kpt_pred,df_gt_behavior, on=["video", "frame"], how='inner')
        # 儲存到 list
        df_kpt_with_labeled.append(df_labeled)
        print(f"✅ 合併完成：mice{start}~{end}.csv，樣本數 = {len(df_labeled)}")

    # 3. 合併所有影片成一份總表
    df_all = pd.concat(df_kpt_with_labeled, ignore_index=True)

    # 4. 輸出
    out_folder = r'data\gt_dataset\3_pred_kpt_gt_behavior'
    os.makedirs(out_folder, exist_ok=True)
    out_csv = os.path.join(out_folder, 'kpts_with_gt_behavior_yolov11.csv') # your file name
    df_all.to_csv(out_csv, index=False)
    print(f"\n 已完成：已輸出 {out_csv}，總樣本數 = {len(df_all)}")

