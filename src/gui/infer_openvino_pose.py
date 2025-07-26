# infer_openvino_pose.py
from openvino.runtime import Core
import cv2, numpy as np
from ultralytics.yolo.utils.ops import letterbox  # 如果你已安裝 ultralytics

# 1. 載入 IR
core = Core()
model = core.read_model("yolov8_openvino/yolov8_pose.xml")
score_layer = model.get_output_op(0)   # 取第一個輸出
compiled = core.compile_model(model, device_name="GPU")  # 若有Intel GPU/NPU可改成"GPU"或"AUTO"

# 2. 前處理
img = cv2.imread("test.jpg")
h0, w0 = img.shape[:2]
img_p = letterbox(img, (640,640), auto=False)[0]            # letterbox to 640×640
img_p = cv2.cvtColor(img_p, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
img_p = np.transpose(img_p, (2,0,1))[None]                  # (1,3,640,640)

# 3. 推論
res = compiled([img_p])[score_layer]    # e.g. shape (1, 29, 8400) for YOLO11l-pose

# 4. 後處理：解析 keypoint
# 每個 prediction 含 [x_center, y_center, w, h, conf, key1_x, key1_y, key2_x, key2_y, …]
pred = res[0].reshape(-1, res.shape[1])  # (8400, 29)
# 篩掉置信度過低的框
conf_thresh = 0.25
mask = pred[:,4] > conf_thresh
pred = pred[mask]

# 將相對座標轉回原圖
boxes = pred[:, :4]
kpts  = pred[:, 5:].reshape(-1, 4, 2)  # 4 個關鍵點
# 反 letterbox：把 0~1 正規化座標映射回原圖
for xy in (boxes, kpts.reshape(-1,2))[0:]:  
    xy[:,0] = (xy[:,0] * 640 - (640-w0)/2) * (w0/ (640 - (640-w0)))  
    xy[:,1] = (xy[:,1] * 640 - (640-h0)/2) * (h0/ (640 - (640-h0)))

# 5. 繪圖檢查
out_img = img.copy()
for box, k in zip(boxes, kpts):
    x1,y1,w,h = box
    cv2.rectangle(out_img, (int(x1),int(y1)), (int(x1+w),int(y1+h)), (0,255,0),2)
    for (kx,ky) in k:
        cv2.circle(out_img, (int(kx),int(ky)), 3, (0,0,255), -1)
cv2.imwrite("result.jpg", out_img)
