import cv2
import os
# 將影片擷取圖片 => 拿去手工label
video_id = 6
cap = cv2.VideoCapture(fr"C:\Users\micha\Desktop\dataset_video\mice{video_id}")  # your video path
output_path = fr"data\gt_dataset\0_frames\mice{video_id}"                        # or your output path
os.makedirs(output_path, exist_ok=True)

count = 0
save_count = 0
step = 1000

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if count%step ==0:
        cv2.imwrite(os.path.join(output_path, f'frame_{count}.jpg'), frame)
        save_count += 1
    count += 1

cap.release()
