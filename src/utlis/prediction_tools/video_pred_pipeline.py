import sys
sys.path.append(".")
from video_to_keypoints import  video_to_keypoints_batch
from kpts_Predict_Behaviors import predict_csv
import yaml
yaml_path = r"src\main\config.yaml"
import pandas as pd



def main(
    video_path,
    kpt_csv,
    out_csv,
    model_path,
    minmax_npz,
    window,
    yolo_weight,
    yolo_conf,
    batch_size=16,
    kp_history_len=3,
    output_mode="single",   # 支援 "single"（合併一檔）或 "split"（分開兩檔）模式
    device="cuda"
):
    # Step 1: 使用 YOLO 模型從影片中預測 keypoints 並輸出至 csv
    df_kpt = video_to_keypoints_batch(
            video_path=video_path,
            out_csv=kpt_csv,
            yolo_weight=yolo_weight,
            yolo_conf=yolo_conf,
            batch_size=batch_size,
            kp_history_len=kp_history_len
            )

    # Step 2: 行為預測模型推論
    # 這裡只回傳預測結果，不直接寫檔案，讓 pipeline 有彈性處理資料合併或格式輸出
    df_pred = predict_csv(
            kpt_csv=kpt_csv,
            model_pth=model_path,
            minmax_npz=minmax_npz,
            out_csv=None,
            window=window,
            device=device
            )
    
    # 根據 output_mode 決定輸出格式
    if output_mode == "single":
        new_cols = [col for col in df_pred.columns if col not in df_kpt.columns or col == "frame"]
        df_merged = pd.merge(df_kpt, df_pred[new_cols], on="frame", how="right")
        df_merged.to_csv(out_csv, index=False, float_format="%.4f")

    
    elif output_mode == "split":
        # 分開輸出模式
        keypoints_cols = [col for col in df_kpt.columns if col.startswith('kpt') or col == 'frame']
        behav_cols = [col for col in df_pred.columns if col not in keypoints_cols]
        df_kpt = df_kpt[keypoints_cols]
        df_beh = df_pred[behav_cols]
        kpt_csv_path = kpt_csv if kpt_csv.endswith('.csv') else kpt_csv + '.csv'
        beh_csv_path = out_csv if out_csv.endswith('.csv') else out_csv + '_behavior.csv'
        df_kpt.to_csv(kpt_csv_path, index=False)
        df_beh.to_csv(beh_csv_path, index=False)
        return kpt_csv_path, beh_csv_path
    
    else:
        raise ValueError("output_mode 必須是 'single' 或 'split'")

if __name__ == "__main__":   
    output_mode="single"
    video_path = r"C:\Users\micha\Desktop\full_database\20080324115556.mpg"

    # 根據 output_mode 決定輸出路徑配置，確保資料流向清晰
    if output_mode == "single":    
        kpt_csv = r"data_prediction\prediction_results\3_combine\kpts_mice1.csv"
        out_csv = kpt_csv    
    else:
        kpt_csv = r"data_prediction\prediction_results\3_combine\kpts_mice1.csv"
        out_csv = r"data_prediction\prediction_results\3_combine\pred_behavior_mice1.csv"
        
    # 讀取所有設定參數
    with open(yaml_path, "r", encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    model_path = cfg["behavior"]["weights"]
    minmax_npz = cfg["paths"]["minmax_npz"]
    window = cfg["behavior"]["window_size"]
    yolo_weight = cfg["yolo"]["weights"]
    pose_input_size = cfg["yolo"]["input_size"]
    yolo_conf = cfg["yolo"]["conf"]
    orig_w, orig_h = cfg["yolo"]["orig_size"][0], cfg["yolo"]["orig_size"][1]
    batch_size = 32 #依顯卡效能決定
    kp_history_len = cfg["pose"]["kp_history_len"]

    # 啟動完整 pipeline，執行關鍵點預測與行為辨識
    main(
        video_path=video_path,
        kpt_csv=kpt_csv,
        out_csv=out_csv,
        model_path=model_path,
        minmax_npz=minmax_npz,
        window=window,
        yolo_weight=yolo_weight,
        yolo_conf=yolo_conf,
        batch_size=batch_size,
        kp_history_len=kp_history_len,
        output_mode=output_mode
    )


