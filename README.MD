專案根目錄
├── src
│   ├── main
│   │   ├── gui.py                   實時影片播放 GUI 播放器
│   │   ├── inference.py             YOLO → DeepSort → Behavior 模型串接推論
│   │   ├── main.py                  專案主程式（載入 GUI + 推論）
│   │   └── config.yaml              超參數設定檔
│   │
│   └── utils
│       ├── frame_cap.py             影片擷取成圖片（for 手動標記）
│       ├── video_to_keypoints.py    影片 → YOLO Pose → 關鍵點輸出
│       ├── Kpts_Predict_Behaviors.py    輸入 kpts → 行為預測 csv
│       ├── video_pred_pipeline.py       串接 YOLO + 行為預測（for API）
│       └── keypoint+gt_behavior.py      合併 YOLO 關鍵點 + 行為 ground truth（供訓練用）
│
├── model
│   ├── YOLO
│   │   ├── train_YOLO.py            YOLO 模型訓練腳本
│   │   └── YOLO_weight/             訓練後模型權重與 results
│   │
│   └── behavior_model
│       ├── Behavior_Model.py        行為分類模型架構
│       ├── train_behavior_models.py     訓練行為模型（用合併後 csv）
│       └── utils/                   行為模型所需的資料處理函式
│
├── data_prediction
│   ├── dataset
│   │   └── gt_behaviors/            Ground Truth 標註的行為 CSV
│   │
│   └── prediction_results
│       ├── keypoints/              YOLO Pose 預測出來的關鍵點序列
│       ├── behaviors/              根據關鍵點預測的行為結果
│       └── combine/                YOLO + 行為 → 組合成完整預測結果（for API or analysis）



model training 流程:
1. Read parameters from hyp_behavior.yaml

2. Load & clean the CSV: 刪除unknown、drink，以video_id, frame排序，針對過短暫行為做平滑

3. Extract sliding windows of keypoints:

4. Split into train/val/test by continuous “bout” groups: 將一次連續行為視為一個bout，避免一個window有same behavior from different segment

5. Balance the train set
6. Build per‐window features
7. Train a BiLSTM (or Transformer) model
8. Evaluate on the held‐out test set and plot a confusion matrix