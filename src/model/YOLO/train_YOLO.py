from ultralytics import YOLO
from pathlib import Path
import torch

def main():
    data_yaml_path = Path(r"C:\Users\micha\Desktop\frame_datas\Mice tracking II.v6i.yolov8\data.yaml")
    assert data_yaml_path.exists(), f"data.yaml 不存在！{data_yaml_path}"
    size = "m"
    model = YOLO(f"yolo11{size}-pose.pt")
    model.train(
        data=str(data_yaml_path),
        cfg=r"C:\Users\micha\Desktop\Mice_tracking_project\src\model\YOLO\hyp_yolo.yaml",   # ← 直接指定 cfg/hyp 檔
        project=r"C:\Users\micha\Desktop\Mice_tracking_project\src\model\YOLO\YOLO_weights",
        name=fr"mouse_pose_yolo11{size}_{150}",
        verbose=True,
        resume=False,                # 新訓練
        exist_ok = True,
        save = True
    )




if __name__ == "__main__":
    main()
