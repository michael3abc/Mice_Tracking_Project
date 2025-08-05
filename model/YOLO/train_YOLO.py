from ultralytics import YOLO
from pathlib import Path
import yaml

def main():
    data_yaml_path = Path(r"data\gt_dataset\1_keypoints\Mice tracking II.v6i.yolov8\data.yaml")
    assert data_yaml_path.exists(), f"data.yaml 不存在！{data_yaml_path}"

    cfg_path = r"model\YOLO\yolo_config.yaml"
    with open(cfg_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    size = 'n'  # 選擇yolo模型尺寸    
    model = YOLO(f"yolo11{size}-pose.pt")

    model.train(
        data=str(data_yaml_path),
        cfg= str(cfg_path),   # ← 直接指定 cfg/hyp 檔
        project=r"model\YOLO\YOLO_weights",
        name=fr"mouse_pose_yolo11{size}_{cfg['epochs']}",
        verbose=True,
        resume=False,                # 新訓練
        exist_ok = True,
        save = True
    )




if __name__ == "__main__":
    main()
