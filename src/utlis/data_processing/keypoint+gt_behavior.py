import re #正規表達
import pandas as pd
import os
# 用來把pred_kpts + gt_behavior讓模型學習

def convert_behavior_txt_to_csv(n, txt_folder, save_folder):
    """
    將 mice1~n.txt 行為標註轉換成 CSV，每一列對應一個 frame。
    欄位為：video, frame, behavior
    """
    os.makedirs(save_folder, exist_ok=True)

    for i in range(1, n+1):
        txt_path = os.path.join(txt_folder, f'mice{i}.txt')
        if not os.path.exists(txt_path):
            print(f"❌ 找不到檔案: {txt_path}")
            continue
        rows = []
        with open(txt_path,'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line.startswith('frame:'):
                    continue
                m = re.match(r'frame:\s*(\d+)\s*-\s*(\d+)\s+(\w+)', line)
                if m:
                    start = int(m.group(1))
                    end = int(m.group(2))
                    label = str(m.group(3))
                    for frame in range(start, end + 1):
                        rows.append({"video": i, "frame": frame, "behavior": label})
                else:
                    print(f"⚠️ 無法解析此行: {line}")

        df = pd.DataFrame(rows)
        save_path = os.path.join(save_folder, f"gt_behavior_mice{i}.csv")
        df.to_csv(save_path, index=False)
        print(f"mice{i}.txt → {save_path}，共 {len(df)} 筆行為標註")
    print("for all finish!")

# if __name__ == '__main__':
#     txt_folder = r"C:\Users\micha\Desktop\dataset_gt_behavior"
#     save_folder = r"C:\Users\micha\Desktop\Mice_tracking_project\data_prediction\dataset\gt_behaviors"
#     convert_behavior_txt_to_csv(n = 12, txt_folder=txt_folder, save_folder=save_folder)
    



if __name__ == '__main__':
    # 1. 讀行為標籤
    gt_csv_folder = r"C:\Users\micha\Desktop\Mice_tracking_project\data_prediction\dataset\gt_behaviors"
    # 2. 讀keypoints
    kpt_folder = r"data_prediction\prediction_results\1_keypoints\yolov11"

    df_kpt_with_labeled = []
    for i in range(1, 13):
        df_gt_behavior = pd.read_csv(os.path.join(gt_csv_folder, f"gt_behavior_mice{i}.csv"))
        df_kpt_pred = pd.read_csv(os.path.join(kpt_folder, f"keypoints_mice{i}.csv"))
       
        if "video" not in df_kpt_pred.columns:
            df_kpt_pred["video"] = i
        # 合併：根據 video + frame
        df_labeled = pd.merge(df_kpt_pred,df_gt_behavior, on=["video", "frame"], how='inner')
        # 儲存到 list
        df_kpt_with_labeled.append(df_labeled)
        print(f"✅ 合併完成：mice{i}.csv，樣本數 = {len(df_labeled)}")

    # 3. 合併所有影片成一份總表
    df_all = pd.concat(df_kpt_with_labeled, ignore_index=True)

    # 4. 輸出
    out_folder = r'data_prediction\dataset\kpt_gt_behavior'
    os.makedirs(out_folder, exist_ok=True)
    out_csv = os.path.join(out_folder, 'kpts_with_gt_behavior_yolov11.csv')
    df_all.to_csv(out_csv, index=False)
    print(f"\n 已完成：已輸出 {out_csv}，總樣本數 = {len(df_all)}")

