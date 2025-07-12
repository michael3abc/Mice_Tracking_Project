import cv2
import os
# 將影片擷取圖片 => 拿去手工label
cap = cv2.VideoCapture(r"C:\Users\micha\Desktop\full_database\20080421153635.mpg")
output_path = r"c:\Users\micha\Desktop\frame_datas\frames_6"
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
