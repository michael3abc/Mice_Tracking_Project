from ultralytics import YOLO
from pathlib import Path
import torch

def main():
    data_yaml_path = Path(r"C:\Users\micha\Desktop\frame_datas\Mice tracking II.v6i.yolov8\data.yaml")
    assert data_yaml_path.exists(), f"data.yaml 不存在！{data_yaml_path}"

    model = YOLO("yolo11l-pose.pt")
    model.train(
        data=str(data_yaml_path),
        cfg=r"C:\Users\micha\Desktop\Mice_tracking_project\src\model\YOLO\hyp_yolo.yaml",   # ← 直接指定 cfg/hyp 檔
        project=r"C:\Users\micha\Desktop\Mice_tracking_project\src\model\YOLO\YOLO_weights",
        name=fr"mouse_pose_yolo11L_{150}",
        verbose=True,
        resume=False,                # 新訓練
        exist_ok = True,
        save = True
    )
    # model.train(
    #     data=str(data_yaml_path),
    #     epochs=20,
    #     imgsz=640,
    #     batch=8,        
    #     lr0=5e-3,
    #     lrf=0.01,
    #     warmup_epochs=3,
    #     hyp=r"C:\Users\micha\Desktop\Mice_tracking_project\src\model\YOLO\hyp_yolo.yaml",
    #     project=r"src\model\YOLO\YOLO_weights",
    #     name="mouse_pose_yolo8x_200_epo",
    #     device=0 if torch.cuda.is_available() else "cpu",
    #     amp=True,            # 混合精度
    #     # close_mosaic=10,     # 後 10 epoch 關閉 Mosaic 收斂
    #     verbose=True,
    #     save=True,
    # )



if __name__ == "__main__":
    main()
