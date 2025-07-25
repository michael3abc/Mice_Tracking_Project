import sys
sys.path.append(".")
from video_to_keypoints import  video_to_keypoints_batch
from kpts_to_behaviors import predict_behavior_from_kpts
import yaml, os
import pandas as pd
from pathlib import Path



def run_pipeline(
    video_path      : str,
    # ------ YOLO → kpts ------
    yolo_weight     : str,
    yolo_conf       : float,
    kp_history_len  : int,
    # ------ kpts → behavior ------
    cfg_path        : str,
    exp_folder      : str,
    # ------ I/O ------
    out_dir         : str,
    base_name       : str,        # 不帶 .csv 的檔名
    output_mode     : str = "single"     # "single" or "split"
):
    """
    1. 影片 → keypoints csv
    2. keypoints → behaviors
    3. 依 output_mode 整合輸出
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    kpt_csv  = out_dir / f"{base_name}_kpts.csv"
    beh_csv  = out_dir / f"{base_name}_behavior.csv"
    single_csv = out_dir / f"{base_name}_merged.csv"

    # Step 1: 使用 YOLO 模型從影片中預測 keypoints 並輸出至 csv
    df_kpt = video_to_keypoints_batch(
            video_path  = video_path,
            out_csv     = str(kpt_csv),
            yolo_weight = yolo_weight,
            yolo_conf   = yolo_conf,
            batch_size  = 32,
            kp_history_len = kp_history_len
            )
    

    df_kpt = pd.read_csv(kpt_csv) 

    # Step 2: 行為預測模型推論
    # 這裡只回傳預測結果，不直接寫檔案，讓 pipeline 有彈性處理資料合併或格式輸出
    df_beh  = predict_behavior_from_kpts(
        cfg_path      = cfg_path,
        exp_folder    = exp_folder,
        input_csv     = str(kpt_csv),
        output_folder = str(out_dir),
        out_name      = base_name,
        out_put_csv   = False    
    )
    
    # 根據 output_mode 決定輸出格式
    if output_mode == "single":
        # 1. 合併欄位
        new_cols = [c for c in df_beh.columns
                    if c not in df_kpt.columns or c == "frame"]
        df_merged = df_kpt.merge(df_beh[new_cols], on="frame", how="right")

        df_merged.to_csv(single_csv, index=False, float_format="%.4f")
        print(f"✔ merged csv saved to: {single_csv}")

        # 3. 如果不想留下原本的 *_kpts.csv，可加一行刪除：
        kpt_csv.unlink()      # 或 os.remove(kpt_csv)

    elif output_mode == "split":

        df_beh.to_csv(beh_csv, index=False)
        print(f"✓ behavior csv saved to {beh_csv}")
    
    else:
        raise ValueError("output_mode 必須是 'single' 或 'split'")

if __name__ == "__main__":
    # === 讀 gui_config ===
    yaml_path = r"src\gui\gui_config.yaml"
    gui_cfg = yaml.safe_load(open(yaml_path, encoding="utf-8"))

    run_pipeline(
        # video
        video_path     = r"C:\Users\micha\Desktop\dataset_video\mice5.mpg",
        # YOLO
        yolo_weight    = gui_cfg["yolo"]["weights"],
        yolo_conf      = gui_cfg["yolo"]["conf"],
        kp_history_len = gui_cfg["pose"]["kp_history_len"],
        # 行為模型
        cfg_path       = r"model\behavior_models\train_config.yaml",
        exp_folder     = r"C:\Users\micha\Desktop\behaviors_model_metrics\STTR_BiLSTM_best_epoch28_test_F1_0.7622",
        # output
        out_dir        = r"data\prediction_results\3_combine",
        base_name      = "mice5",
        output_mode    = "single"
    )
