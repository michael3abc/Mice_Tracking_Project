# gui.py
import sys
import os
import json
import yaml
import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QSlider, QLabel, QComboBox
)
from inference import InferenceEngine
from collections import deque

class MainWindow(QWidget):
    def __init__(self, cfg: str):
        super().__init__()
        # self._load_config(cfg_path)
        self.cfg = cfg    
        self.display_scale = self.cfg.get("display_scale", 2.0)
        self._init_ui()
        self._init_video()
        # 啟用拖放
        self.setAcceptDrops(True)

        # 建立推論引擎
        self.engine = InferenceEngine(
            yolo_weights=self.cfg["yolo"]["weights"],
            yolo_conf=self.cfg["yolo"]["conf"],
            pose_input_size=self.cfg["yolo"]["input_size"],  
            orig_size = self.cfg["yolo"]["orig_size"],

            kp_history_len=self.cfg["pose"]["kp_history_len"],
            vel_delta = self.cfg["pose"]["vel_delta"],
            pose_class_path= self.cfg["pose"]["pose_class_path"],

            behavior_model=self.cfg["behavior"]["model"], 
            behavior_weights=self.cfg["behavior"]["weights"],
            window_size=self.cfg["behavior"]["window_size"],

            rest_prob_margin=self.cfg["behavior"]["rest_prob_margin"],
            min_any_duration=self.cfg["behavior"]["min_any_duration"],
            minmax_npz=self.cfg["paths"]["minmax_npz"],
        )
        self.frame_buf = deque(maxlen=self.engine.window_size)

    def _init_ui(self):
        self.setWindowTitle("Mouse Pose Tracking & Behavior")
        self.resize(1920, 1080)

        # 顯示區
        self.video_label = QLabel()
        self.video_label.setFixedSize(1600, 900)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)

        # 按鈕＆控制元件
        open_btn = QPushButton("Open Video")
        open_btn.clicked.connect(self.open_file)

        self.play_btn = QPushButton("Play")
        self.play_btn.setEnabled(False)
        self.play_btn.clicked.connect(self.play_pause)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderMoved.connect(self.seek)

        self.time_label = QLabel("00:00 / 00:00")
        font = QFont("Arial Rounded MT Bold", 18)
        font.setBold(True)
        self.time_label.setFont(font)
        self.time_label.setAlignment(Qt.AlignCenter)

        self.speedBox = QComboBox()
        self.speedBox.addItems(["0.25x","0.5x","1.0x","1.5x","2.0x"])
        self.speedBox.setCurrentText("1.0x")
        self.speedBox.currentIndexChanged.connect(self.change_speed)

        ctrl = QHBoxLayout()
        ctrl.addWidget(open_btn)
        ctrl.addWidget(self.play_btn)
        ctrl.addWidget(self.speedBox)
        ctrl.addWidget(self.slider)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.time_label)
        layout.addLayout(ctrl)
        self.setLayout(layout)

    def _init_video(self):
        self.cap = None
        self.total_frames = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.next_frame)

    #-------- 影片拖放 & 開檔 -------------------------------
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        file_path = urls[0].toLocalFile()
        self.load_video(file_path)

    def open_file(self):
        fn, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.mpg)"
        )
        if fn:
            self.load_video(fn)

    def load_video(self, path):
        # 釋放前一支
        if self.cap:
            self.cap.release()
            self.timer.stop()
            self.engine.reset()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            print("影片開啟失敗")
            return
        # 讀取並印出 FPS
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[INFO] Loaded video '{path}' with FPS = {fps}")

        self.cap = cv2.VideoCapture(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.slider.setMaximum(self.total_frames)
        self.slider.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.play_btn.setText("Play")

        


    #-------- 播放／暫停 & 進度條 -------------------------------
    def play_pause(self):
        if not self.cap:
            return

        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("Play")
        else:
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            speed = float(self.speedBox.currentText().replace("x", ""))
            interval = int(1000 / fps / speed)
            self.timer.start(interval)
            self.play_btn.setText("Pause")

    def seek(self, frame_no):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    def format_time(self, ms):
        seconds = int(ms // 1000)
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes:02}:{seconds:02}"

    def change_speed(self):
        if not self.cap:
            return
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        speed = float(self.speedBox.currentText().replace("x", ""))
        interval = int(1000 / fps / speed)
        self.timer.setInterval(interval)

    #-------- 每幀更新 & 呼叫 InferenceEngine -------------------------------
    def next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            while self.frame_buf:
                self._show_frame(self.frame_buf.popleft())
            self.timer.stop()
            return

        curr = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

        # 呼叫 inference，回傳帶標註的 frame
        annotated = self.engine.process_frame(frame=frame,curr_frame= curr)

        self.frame_buf.append((annotated, curr))
        # if len(self.frame_buf) == self.engine.window_size:
        oldest_frame, oldest_idx = self.frame_buf.popleft()
        self._show_frame(oldest_frame, oldest_idx)

    def _show_frame(self, annotated, curr):
        # 更新時間與 slider
        total = self.total_frames
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        curr_ms  = curr * 1000 / fps
        total_ms = total * 1000 / fps
        self.time_label.setText(
            f"{self.format_time(curr_ms)} / {self.format_time(total_ms)}    "
            f"Frame: {curr}/{total}"
        )
        self.slider.blockSignals(True)
        self.slider.setValue(curr)
        self.slider.blockSignals(False)

        # OpenCV BGR → Qt RGB 顯示
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio
        )
        self.video_label.setPixmap(pix)

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()

# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
#     window = MainWindow(cfg_path)
#     window.show()
#     sys.exit(app.exec_())
