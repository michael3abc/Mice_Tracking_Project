📦 專案根目錄
```
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
```

GPT推薦管理方式:
```
📦 Mice_tracking_project/
├── src/
│   ├── main/
│   │   ├── gui.py                  📺 GUI 播放器（實時播放）
│   │   ├── inference.py            🧠 YOLO → DeepSort → 行為推論
│   │   ├── main.py                 🚀 主程式（啟動 GUI + 推論）
│   │   └── config.yaml             🔗 🔄 (deprecated: 移入 config/)
│   │
│   ├── utils/
│   │   ├── frame_cap.py            🎞️ 擷取影片 frame（for 手工標記）
│   │   ├── video_to_keypoints.py   🎯 YOLO pose 預測關鍵點
│   │   ├── Kpts_Predict_Behaviors.py   🧠 透過 kpt → 行為預測 csv
│   │   ├── video_pred_pipeline.py      🔗 串接 YOLO + 行為 → 推論 pipeline
│   │   └── keypoint+gt_behavior.py     🧾 加上真實行為 label（供訓練用）
│
├── model/
│   ├── YOLO/
│   │   └── train_YOLO.py           🏋️ 訓練 YOLO Pose 模型
│   │
│   └── behavior_model/
│       ├── Behavior_Model.py       🧠 行為模型架構（BiLSTM / CNN）
│       ├── train_behavior_models.py    🏋️ 行為模型訓練主程式
│       └── utils/                  🧰 訓練前資料前處理與分析
│
├── weights/                        💾 所有模型的權重統一管理
│   ├── yolo/
│   │   ├── yolov8x-pose.pt
│   │   └── yolov8l-pose.pt
│   └── behavior/
│       ├── BiLSTM_best.pt
│       └── CNN_drink_model.pt
│
├── logs/                           📝 訓練過程 log 與 TensorBoard
│   ├── yolo/
│   └── behavior/
│
├── experiments/                    🧪 不同實驗組合輸出資料夾
│   ├── exp1_CNN_groom/
│   └── exp2_BiLSTM_drink/
│
├── scripts/                        🖥️ 主要訓練/測試流程腳本
│   ├── run_train_yolo.py
│   ├── run_train_behavior.py
│   ├── run_inference_pipeline.py
│   └── export_results_to_csv.py
│
├── config/                         ⚙️ 拆分 YAML 設定檔
│   ├── yolo_config.yaml
│   ├── behavior_config.yaml
│   └── gui_config.yaml
│
├── data_prediction/
│   ├── dataset/
│   │   ├── gt_behaviors/           ✅ Ground Truth 標註的行為 CSV
│   │   └── kpt_gt_behavior/        ✅ kpt + behavior 合併訓練資料
│   │
│   └── prediction_results/
│       ├── keypoints/             🎯 YOLO pose 預測結果
│       ├── behaviors/             🧠 預測的行為結果（來自 kpt）
│       └── combine/               📦 YOLO + 行為 → 合併完整預測
```
