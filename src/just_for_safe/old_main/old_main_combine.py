import sys, os, json
import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import pandas as pd
from collections import deque
from typing import Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QSlider, QLabel)
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtGui import QFont 

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) #前兩層
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from model.Behaviors_Models import Behavior1D_CNN, BehaviorBiLSTM, BehaviorBiLSTM_v3


class Main_program(QWidget):
    def __init__(
        self, 
        yolo_weights: str,
        yolo_conf: float,       # 只保留高於信心門檻的
        pose_input_size: int,   # YOLOv8‑Pose 模型輸入圖像邊長
        kp_history_len: int,    # 紀錄幾幀關鍵點 => 做「滑動平均」
        behavior_weights: str, 
        window_size: int,       # 模型一次要看幾幀才能下判斷
        rest_prob_margin: float,# rest特別處理: 當 rest 預測機率低於這個 margin，就會替換成第二高的類別
        min_any_duration: int,  # 用於smoothing 行為預測 (避免判定抖動)
        minmax_npz: str,        # 存 keypoint min/max 的 .npz 檔案路徑
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
        ):
                
        super().__init__()

        #1. 初始化參數
        self._init_params(
            yolo_weights, yolo_conf,
            pose_input_size, kp_history_len,
            behavior_weights, window_size,
            rest_prob_margin, min_any_duration,
            minmax_npz, device
        )

        # 2. 模型
        self._init_models()

        # 3. 運行維護
        self._init_state()

        # 4. GUI
        self._init_ui()

    def _init_models(self):
        # YOLOv8-Pose
        self.model    = YOLO(self.yolo_weights)
        self.model.fuse()  # optional 加速

        # DeepSort
        self.tracker  = DeepSort(max_age=10, max_iou_distance=0.7)

        # 行为预测模型
        self.behavior_model = BehaviorBiLSTM_v3().to(self.device)
        state = torch.load(self.behavior_weights, map_location="cpu")
        self.behavior_model.load_state_dict(state)
        self.behavior_model.eval()

        # min-max 参数
        npz = np.load(self.minmax_npz)
        self.X_min, self.X_max = npz["X_min"], npz["X_max"]

    def _init_params(self,
                     yolo_weights, yolo_conf,
                     pose_input_size, kp_history_len,
                     behavior_weights, window_size,
                     rest_prob_margin, min_any_duration,
                     minmax_npz, device):
        # 统一保存所有超参数
        self.device            = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_weights      = yolo_weights
        self.yolo_conf         = yolo_conf
        self.pose_input_size   = pose_input_size
        self.kp_history_len    = kp_history_len
        self.behavior_weights  = behavior_weights
        self.window_size       = window_size
        self.rest_prob_margin  = rest_prob_margin
        self.min_any_duration  = min_any_duration
        self.minmax_npz        = minmax_npz

        self.keypoint_names = ['tail_base','body','nose','rear_right', 'rear_left', 'front_left',  'front_right', 'hip']
        # 骨架連線
        self.skeleton = [
            (0, 1),  # tail → hip
            (1, 7),  # body → hip
            (1, 3),  # hip → rear_right
            (1, 4),  # hip → rear_left
            (7, 5),  # body -> front_left
            (7, 6),  # body -> front_right
            (2, 7),  # nose → body
        ]
        self.line_color = (180, 180, 180)
        # 顏色樣式
        self.point_colors = [
            (255, 0, 0),     # 🔴 紅色 → index 0 tail
            (0, 255, 0),     # 🟢 綠色 → index 1 body
            (0, 0, 255),     # 🔵 藍色 → index 2 head
            (128, 128, 0),   # 🟡 橄欖黃 → index 3 rear_right
            (255, 128, 0),   # 🟠 橘色 → index 4 rear_left
            (0, 255, 255),   # 🟦 青色 → index 5  front_left
            (255, 0, 255),   # 🟣 粉紫 → index 6 front_right
            (128, 0, 255)    # 🔮 紫藍 → index 7 hip
        ]
        self.behavior_names = ['eat', 'groom', 'hang', 'micromovement', 'rear', 'rest', 'walk']   

    def _init_state(self):
        # Keypoint 歷史 => 補幀
        self.kp_history = [deque(maxlen=self.kp_history_len) for _ in range(8)]

        # Pose → 特征 窗口
        self.pose_window        = deque(maxlen=self.window_size)
        self.model_input_window = deque(maxlen=self.window_size)

        # Behavior 机率 窗口
        self.behavior_probs_window = deque(maxlen=self.window_size)

        # Video 播放控制
        self.cap          = None
        self.total_frames = 0
        self.timer        = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

    def _init_ui(self):
        # 主視窗
        self.setWindowTitle("Mouse Pose Tracking & Behavior")
        self.resize(1920, 1080)
        self.setAcceptDrops(True)

        open_btn = QPushButton("Open Video") #自訂按鈕
        open_btn.clicked.connect(self.open_file)

        # 顯示區
        self.video_label = QLabel()
        self.video_label.setFixedSize(1600, 900)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)

        # 控制按钮
        self.play_btn = QPushButton("Play");   self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.play_pause)

        self.slider   = QSlider(Qt.Horizontal); self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self.seek)

        self.time_label  = QLabel("00:00 / 00:00")
        f = QFont("Arial Rounded MT Bold", 18); f.setBold(True)
        self.time_label .setFont(f); self.time_label .setAlignment(Qt.AlignCenter)

        self.speedBox = QComboBox()
        self.speedBox.addItems(["0.25x","0.5x","1.0x","1.5x","2.0x"])
        self.speedBox.setCurrentText("1.0x")
        self.speedBox.currentIndexChanged.connect(self.change_speed)

        # 排版
        ctrl = QHBoxLayout()
        ctrl.addWidget(open_btn)
        ctrl.addWidget(self.play_btn)
        ctrl.addWidget(self.speedBox)
        ctrl.addWidget(self.slider)

        main = QVBoxLayout()
        main.addWidget(self.video_label)
        main.addWidget(self.time_label )
        main.addLayout(ctrl)

        self.setLayout(main)

#--------影片撥放function-------------------------------
    def dragEnterEvent(self, event): #拖移放置影片
        if event.mimeData().hasUrls(): #檢查使用者拖進來的是不是一個「檔案路徑」
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        file_path = urls[0].toLocalFile() #如果丟進來folder打開第一個，轉成本地路徑
        self.load_video(file_path) #用load video開啟 (由CV2撥放)

    def open_file(self):
        fileName, _ = QFileDialog.getOpenFileName( #彈出一個系統的「開啟檔案視窗」，讓使用者選影片檔案，且限制(檔案)影片類型
            self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv *.mpg)"
        )
        if fileName:
            self.load_video(fileName)

    def load_video(self, path): #載入影片
        # Release previous
        if self.cap: #檢查是否已經有物件，有的化把他釋出 (才能接新的)
            self.cap.release()
            self.timer.stop()

        # 用OpenCV 打開 => capture
        self.cap = cv2.VideoCapture(path) 
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) #讀取總幀數
        self.slider.setMaximum(self.total_frames) #滑桿最大值為影片總幀數
        self.slider.setEnabled(True) #啟用滑桿
        self.play_btn.setEnabled(True) #可以play了
        self.play_btn.setText("Play")

    def play_pause(self): #暫停撥放
        if not self.cap: #還沒有影片 => 沒反應
            return
        
        if self.timer.isActive(): #如果正在撥放 => 按一下暫停，btn顯示play
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            speed_text = self.speedBox.currentText()
            speed = float(speed_text.replace("x", ""))
            normal_interval = int(1000/fps)
            interval = int(normal_interval/speed)
            self.timer.start(interval)     #設定Qtimer的週期 
            self.play_btn.setText("Pause")

    def seek(self, frame_no): #找滑桿移動到的位置給CV
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    
    def format_time(self, ms):
        seconds = int(ms // 1000)
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02}:{seconds:02}"
   
    def change_speed(self):
        speed_text = self.speedBox.currentText()
        speed = float(speed_text.replace("x", "")) #從speedBox取得str後轉成float
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        if hasattr(self, 'mediaPlayer'):
            self.mediaPlayer.setPlaybackRate(speed)
        normal_interval = int(1000 / fps)
        interval = int(normal_interval / speed)
        self.timer.setInterval(interval)

#--------行為相關function-------------------------------
    def restore_and_normalize_keypoints(self, keypoints, orig_size, input_size = 640): #補關鍵點 + 正規化
        """
        keypoints: List of (x_resized, y_resized) from YOLOv8 Pose 輸出
        orig_size: (orig_w, orig_h) 原圖寬高
        input_size: 模型輸入大小，預設 640
        回傳: List of tuples (x_orig, y_orig, x_norm, y_norm)
        """
        orig_w, orig_h = orig_size
        inp = input_size
        #縮放比例:
        scale = min(inp / orig_w, inp / orig_h)
        #計算填充的像素(比例)
        pad_w = (inp - orig_w*scale) / 2 #左右各補一半
        pad_h = (inp - orig_h*scale) / 2

        restored = []
        for x_res, y_res in keypoints:
            x_orig = (x_res - pad_w) / scale
            y_orig = (y_res - pad_h) / scale
            x_norm = min(max(x_orig / orig_w, 0.0), 1.0) #限制在0~1 避免出錯
            y_norm = min(max(y_orig / orig_h, 0.0), 1.0)
            restored.append((x_orig, y_orig, x_norm, y_norm))
        return restored

    def calculate_velocity_acceleration(self, window): #計算速度、加速度 => 輸出24 dim 特徵向量
        #輸入16維的kpt計算速度、加速度
        if len(window) < 7:
            return None

        p_t = window[-1]
        p_t_1 = window[-4]
        p_t_2 = window[-7]
        
        v_t = p_t - p_t_1
        v_t_1 = p_t_1 - p_t_2
        a_t = v_t - v_t_1
        return np.concatenate([v_t, a_t]) #16 + 16 = 32維

    def keypoint_interp(self, idx, curr_frame, max_disp = 50, ): #線性插值補keypoints
        hist = self.kp_history[idx]     #deque存: (frame_idx, (x,y))

        pts = [p for p in hist if isinstance(p, tuple) and len(p) == 2]
        if len(pts) < 2: 
            return None
      
        # 「最靠前」與「最靠後」的兩個真實點
        (f0, (x0, y0)), (f1, (x1,y1)) = pts[0], pts[-1]
        if not (f0 < curr_frame < f1):
            return None
        
        #做線性插值
        alpha = (curr_frame - f0)/(f1 - f0)                
        x_interp, y_interp = x0 + alpha* (x1 - x0), y0 + alpha* (y1 - y0)
        
        #如果誤差太大 => pass (允許位移的pix)
        dx, dy = x_interp - x1, y_interp - y1
        if (dx*dx + dy*dy)**0.5 > max_disp:
            return None
        return (int(x_interp), int(y_interp))

    def smooth_predictions(self, prob_window, rest_index: int): #避免行為預測抖動
        """
        prob_window: (T, C) 每幀的機率分布
        rest_index: rest 在 classes 裡的 index
        min_rest_duration: 最少要多少幀才算是真正的 rest
        min_any_duration: 其他類別的最小段落門檻
        回傳: 長度 T 的平滑後 label list
        """
        T, C = prob_window.shape #時間長度 * 幾個class； ex. prob_window[0] = [0.1, 0.2, 0.7] 表示第 0 幀的預測結果
        raw = np.argmax(prob_window, axis=1)
        out = raw.copy()

        start = 0
        while start < T:
            lbl = raw[start]
            # 找這段段落 [start, end)
            end = start+1
            while end<T and raw[end]==lbl:
                end += 1
            length = end - start

            # 2️. rest 段落特殊處理
            if lbl == rest_index:
                rest_probs =  prob_window[start:end, rest_index]
                low_conf_mask = rest_probs < self.rest_prob_margin
                if np.any(low_conf_mask):
                    top2 = np.argsort(prob_window[start:end], axis=1)[:, -2:]
                    for i, (a, b) in enumerate(top2):
                        if low_conf_mask[i]:
                            out[start + i] = a if b == rest_index else b

            # 3️. 其他類別的抖動過短
            elif lbl != rest_index and length < self.min_any_duration:
                prev_lbl = out[start-1] if start > 0 else None
                next_lbl = raw[end]     if end < T   else None

                # 如果前後同，就補同；否則比長度
                if prev_lbl is not None and prev_lbl == next_lbl:
                    fill = prev_lbl
                else:
                    # 計算前段/後段延續長度
                    # left_len = sum(1 for i in range(start-1,-1,-1) if raw[i]==prev_lbl) if prev_lbl is not None else 0
                    # right_len= sum(1 for i in range(end,T)     if raw[i]==next_lbl) if next_lbl is not None else 0
                    # fill = prev_lbl if left_len>=right_len else next_lbl

                    left = np.sum(raw[max(0, start - 50):start][::-1] == prev_lbl) if prev_lbl is not None else 0
                    right = np.sum(raw[end:end + 50] == next_lbl) if next_lbl is not None else 0
                    fill = prev_lbl if left >= right else next_lbl

                if fill is not None:
                    out[start:end] = fill

            start = end

        return out
    
    def process_track(self, frame: np.ndarray, track, kps_xy, kps_conf, curr_frame: int): #處理每格YOLO + deepsort輸出的Track
        """
        處理一個track:
        1. 框出box => 補缺kpts => 畫skeleton => 呼叫normalize_and_flatten_to_Xfull => 把X_full丟到model_input_window => 讓predict_behavior model預測行為
        """
        x, y, w, h = map(int, track.to_ltwh())
        cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 1)       
        bbox_center = (x + w // 2, y + h // 2)
        valid = {}
        conf_thresh  = self.yolo_conf
        
        for idx,(kx, ky) in enumerate(kps_xy):
            if kx > 0 and ky > 0:
                pt = (int(kx), int(ky))
                self.kp_history[idx].append(pt)

            if self.kp_history[idx]:
                avg = np.mean(self.kp_history[idx], axis=0)
                valid[idx] = tuple(map(int, avg))
            else:
                pt = bbox_center
                valid[idx] = pt
        # #計算kpt
        # for idx, (kx, ky) in enumerate(kps_xy):
        #     conf = float(kps_conf[idx])
        #     if conf >= conf_thresh:
        #         # 真實點
        #         pt_real = (int(kx), int(ky))
        #         self.kp_history[idx].append((curr_frame, pt_real))
        #         valid[idx] = pt_real
        #     else:
        #         # 嘗試插值
        #         pt = self.keypoint_interp(idx, curr_frame)
        #         if pt is not None:
        #             valid[idx] = pt
        #         else:
        #             # 插值 & 信心都失敗 → 優先用最後一次真實點
        #             if len(self.kp_history[idx]) > 0:
        #                 _, last_pt = self.kp_history[idx][-1]
        #                 valid[idx] = last_pt
        #             else:
        #                 # 連歷史都沒有才 fallback 用 bbox_center
        #                 valid[idx] = bbox_center
                
                    
                    

        for a, b in self.skeleton:
            if a in valid and b in valid:
                cv2.line(frame, valid[a], valid[b], self.line_color, 1, lineType=cv2.LINE_AA)

        for idx, pt in valid.items():
            cv2.circle(frame, pt, 3, self.point_colors[idx], -1, lineType=cv2.LINE_AA)     # 彩色點

        X_full = self.build_feature_vector(frame, valid)

        if X_full is None:
            return
        self.model_input_window.append(X_full)
        behavior, top3 = self.predict_behavior()
        if behavior:
            cv2.putText(frame, behavior, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    
    def build_feature_vector (self, frame, valid): #補帧 + 正規化 + [速度, 加速度] => 輸出成X_full 64維
        '''
        呼叫補帧 + normalize => 輸出flat
        '''
        if len(valid) != 8:
            return None

        restored = self.restore_and_normalize_keypoints(
            [valid[j] for j in range(8)],
            orig_size=(frame.shape[1], frame.shape[0]),  #(w, h)
            input_size=640
        )

        
        #用相對位置
        flat_norm = []
        for _, _, x_n, y_n in restored:
            flat_norm.extend([x_n, y_n])
        flat_norm = np.array(flat_norm, dtype= np.float32) # shape = (16,)
        # self.pose_window.append(flat_norm)

        #用min-Max
        ordered_kps = [valid[i] for i in range(8)]
        flat_kpt = np.array([coord for pt in ordered_kps for coord in pt], dtype=np.float32)# 轉成 numpy 陣列 (16,)

        # 從npz讀取X_min和X_max，是 (1,1,16)，先reshape
        X_min = self.X_min.reshape(-1)  # (16,)
        X_max = self.X_max.reshape(-1)  # (16,)

        # 做 min-max 正規化
        flat_kps_scaled = (flat_kpt - X_min) / (X_max - X_min + 1e-6)
        self.pose_window.append(flat_kps_scaled)

        if len(self.pose_window) >= 3:
            input_vec = self.calculate_velocity_acceleration(self.pose_window)
            if input_vec is not None:
                # 組合成64 維向量，
                X_full = np.concatenate([flat_norm, flat_kps_scaled, input_vec], axis=0)  # shape: (64,)
                return X_full
        return None
             
    def predict_behavior(self): #當window_np滿的時候，去抓來給行為model預測              
        window_np = np.array(self.model_input_window).T  # 累積32張1*16 ，把它transpose => shape: (16, 32)
        input_tensor = torch.tensor(window_np, dtype=torch.float32).unsqueeze(0).to(self.device) # 轉成 (batch_size=1, channels=16, sequence_length=32)   
        with torch.no_grad():
            logits = self.behavior_model(input_tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()  # (C,)
            self.behavior_probs_window.append(probs)

             # 取得 top-3 行為與機率
            top3_idx = np.argsort(probs)[-3:][::-1]  # 從高到低
            top3 = [(self.behavior_names[i], round(probs[i], 4)) for i in top3_idx]

            if len(self.behavior_probs_window) == self.window_size:
                prob_window = np.array(self.behavior_probs_window)
                smoothed = self.smooth_predictions(prob_window, rest_index=self.behavior_names.index("rest"))
                final_idx = int(smoothed[-1])
                behavior = self.behavior_names[final_idx]
            else:
                behavior = ""  
                top3 = []       
        return behavior, top3

#------------------------------------------------------------------
    def next_frame(self): #抓每一幀給YOLO跟其他model
        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            return
        curr_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1 #目前是第幾幀
        frame = cv2.resize(frame, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

        # ---------- YOLOv8-Pose 推論 ----------
        results = self.model(frame, conf=self.yolo_conf)[0]
        all_kps_conf = results.keypoints.conf.cpu().numpy()
        all_kps_xy   = results.keypoints.xy.cpu().numpy()

        # ---------- DeepSort 偵測 ----------
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            scores = results.boxes.conf.cpu().numpy()
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = box
                detections.append(([x1, y1, x2 - x1, y2 - y1], score, 'mouse'))

        tracks = self.tracker.update_tracks(detections, frame=frame)

        # ---------- 繪製關鍵點與預測行為 ----------
        for i, track in enumerate(tracks):
            if not track.is_confirmed():
                continue
            if i < len(all_kps_xy):
                single_kps_xy   = all_kps_xy[i]    # shape (8,2)
                single_kps_conf = all_kps_conf[i]  # shape (8,)
                self.process_track(
                    frame, track,
                    single_kps_xy,
                    single_kps_conf,
                    curr_frame
                )

        # ---------- 更新時間與滑桿 ----------
        curr = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) #目前到哪一幀
        total = self.total_frames
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        curr_ms = curr * 1000 / fps
        total_ms = total * 1000 / fps
        self.time_label.setText(f"{self.format_time(curr_ms)} / {self.format_time(total_ms)}    Frame: {curr}/{total}")

        self.slider.blockSignals(True)
        self.slider.setValue(curr) #更新滑桿位置
        self.slider.blockSignals(False)

        # 把 OpenCV 畫面 → Qt 畫面: BGR → RGB → QImage → QPixmap
        # CV 是用numpy 轉成Qt要的QPixmap
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape #(高度, 寬度, 色彩通道數) → 通常是 (720, 1280, 3)
        bytes_per_line = ch * w #每一行的像素資料
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888) #每個 pixel 有紅、綠、藍 8-bit 共 24 bit。
        pix = QPixmap.fromImage(qimg).scaled(self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(pix)


    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()


if __name__ == "__main__":   
    from pathlib import Path
    current_dir = Path(__file__).parent
    json_path = current_dir / "config.json"

    with open(json_path,"r", encoding="utf-8") as f:
        
        cfg = json.load(f)

    app = QApplication(sys.argv)
    player = Main_program(
    yolo_weights=cfg["yolo"]["weights"],
    yolo_conf=cfg["yolo"]["conf"],
    pose_input_size=cfg["yolo"]["input_size"],
    kp_history_len=cfg["pose"]["kp_history_len"],
    behavior_weights=cfg["behavior"]["weights"],
    window_size=cfg["behavior"]["window_size"],
    rest_prob_margin=cfg["behavior"]["rest_prob_margin"],
    min_any_duration=cfg["behavior"]["min_any_duration"],
    minmax_npz=cfg["paths"]["minmax_npz"]
)

    player.show()
    sys.exit(app.exec_())
