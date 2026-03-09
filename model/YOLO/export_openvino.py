import shutil
from pathlib import Path
from ultralytics import YOLO
# 你的模型名稱／路徑
POSE_MODEL_NAME = "yolov11L_pose"     # or "yolov8n-pose"
PT_WEIGHT_PATH  = fr"model\YOLO\YOLO_weights\mouse_pose_yolo11s_80\weights\best.pt"

# Ultralytics Model 物件
pose_model = YOLO(PT_WEIGHT_PATH)

# OpenVINO IR 輸出目錄
output_dir = Path(f"{POSE_MODEL_NAME}_openvino_model")
xml_path   = output_dir / f"{POSE_MODEL_NAME}.xml"

# 1. 若想強制重轉儲，先刪除整個 output_dir
if output_dir.exists():
    print(f"[Export] 刪除舊資料夾：{output_dir}")
    shutil.rmtree(output_dir)

# 2. 呼叫 export() 進行轉檔
print(f"[Export] 開始匯出模型到 OpenVINO IR → {output_dir}")
pose_model.export(
    format="openvino",   # 指定輸出格式
    dynamic=True,        # 保留動態 Shape
    half=True            # 產生 FP16 模型
)

# # 3. 確認輸出結果
# if xml_path.exists():
#     print(f"[Export] 完成，IR 檔案位置：{xml_path}")
# else:
#     raise FileNotFoundError(f"匯出失敗，找不到 {xml_path}")