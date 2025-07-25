# Mice Tracking Project
```text
├── src
│   ├── gui
│   │   ├── gui_display.py              # GUI 播放器設定
│   │   ├── inference.py                # 模型推論架構
│   │   ├── main.py                     # 專案主程式（載入 GUI + 推論）
│   │   └── gui_config.yaml             # GUI 相關超參數
│   │
│   └── utils
│       ├── build_feature.py            # 建立 feature_vector (train/inference)
│       └── model_factory.py            # 將輸入轉成各模型所需格式
│
├── model
│   ├── YOLO
│   │   ├── train_YOLO.py               # YOLO 模型訓練腳本
│   │   ├── yolo_config.yaml            # YOLO 訓練超參數
│   │   └── YOLO_weights/               # 訓練後權重與 results
│   │
│   └── behavior_model
│       ├── models/                     # 不同行為模型的 class 定義
│       ├── train.py                    # 行為模型訓練入口
│       ├── trainer.py                  # 訓練流程 & checkpoint 儲存
│       ├── dataset_loader.py           # Dataset 與 DataLoader 建立
│       └── train_utils/                # 行為模型所需資料處理函式
│
├── data
│   ├── gt_dataset
│   │   ├── 0_frames/                   # video 擷取出之圖片
│   │   ├── 1_keypoints/                # Roboflow 標註關鍵點資料
│   │   ├── 2_gt_behaviors/             # Ground‐Truth 行為標註 CSV
│   │   └── 3_pred_kpt_gt_behavior/     # YOLO Kpts + GT 行為合併 (訓練用)
│   │
│   └── prediction_results
│       ├── 1_keypoints/                # YOLO Pose 預測輸出關鍵點
│       ├── 2_behaviors/                # 關鍵點 → 行為預測結果
│       └── 3_combine/                  # Kpts + 行為 → 完整預測結果
│
└── tools
    ├── prepare_data
    │   ├── frame_cap.py                # 影片截圖 (手動標記用)
    │   └── merge_kpts_gt_behavior.py   # 合併 Kpts 與 行為 GT
    │
    └── prediction_tools
        ├── video_to_keypoints.py       # 影片 → YOLO Pose → 關鍵點
        ├── kpts_to_behaviors.py   # Kpts → 行為預測 CSV
        └── video_pred_pipeline.py      # 影片→Kpts→行為 csv 整合流程
