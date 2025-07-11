# inference.py
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import Optional

import sys, os
import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from collections import deque
from typing import Optional

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) #前兩層
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from model.Behaviors_Models import Behavior1D_CNN, BehaviorBiLSTM, BehaviorBiLSTM_v3
from model import Behaviors_Models

class InferenceEngine:
    def __init__(
        self,
        yolo_weights: str,
        yolo_conf: float,
        pose_input_size: int,
        kp_history_len: int,
        behavior_model : str,
        behavior_weights: str,
        window_size: int,
        rest_prob_margin: float,
        min_any_duration: int,
        minmax_npz: str,
        device: Optional[str] = None
    ):
        # 1. 儲存超參數
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_conf = yolo_conf
        self.pose_input_size = pose_input_size  
        self.kp_history_len = kp_history_len
        self.window_size = window_size
        self.rest_prob_margin = rest_prob_margin
        self.min_any_duration = min_any_duration

        # 2. 載入模型
        self.behavior_weights = behavior_weights
        self.behavior_model_name = behavior_model
        self._init_models(yolo_weights, behavior_weights, minmax_npz)

        # 3. 狀態維護
        self.kp_history = [deque(maxlen=kp_history_len) for _ in range(8)]
        self.pose_window = deque(maxlen=window_size)
        self.model_input_window = deque(maxlen=window_size)
        self.behavior_probs_window = deque(maxlen=window_size)

        # 4. 類別／顏色設定（畫 skeleton 時用）
        self._init_visuals()

    def _init_models(self, yolo_w, beh_w, minmax_npz):
        # YOLOv8-Pose
        self.yolo = YOLO(yolo_w)
        self.yolo.fuse()

        # DeepSort
        self.tracker = DeepSort(max_age=10, max_iou_distance=0.7)

        # 行為模型（BiLSTM v3）
        try:
            ModelClass = getattr(Behaviors_Models, self.behavior_model_name)
        except AttributeError:
            raise ValueError(f"Unknown behavior model: {self.behavior_model_name}")
        
        self.behavior_model = ModelClass().to(self.device)
        state = torch.load(beh_w, map_location="cpu")
        self.behavior_model.load_state_dict(state)
        self.behavior_model.eval()

        # MinMax normalization parameters
        npz = np.load(minmax_npz)
        self.X_min = npz["X_min"].reshape(-1)
        self.X_max = npz["X_max"].reshape(-1)

    def _init_visuals(self):
        self.skeleton = [(0,1),(1,7),(1,3),(1,4),(7,5),(7,6),(2,7)]
        self.point_colors = [
            (255,0,0),(0,255,0),(0,0,255),(128,128,0),
            (255,128,0),(0,255,255),(255,0,255),(128,0,255)
        ]
        self.line_color = (180,180,180)
        self.behavior_names = ['eat','groom','hang','micromovement','rear','rest','walk']

    def process_frame(self, frame: np.ndarray, curr_frame: int):
        """
        1) 呼叫 YOLOv8-Pose，取出 keypoints + boxes  
        2) DeepSort 更新追蹤  
        3) 逐 track 呼叫 _process_track 加上關鍵點 & 行為  
        回傳：帶標註的 frame
        """
        # (A) YOLO 推論
        results = self.yolo(frame, conf=self.yolo_conf)[0]
        kps_xy   = results.keypoints.xy.cpu().numpy()
        kps_conf = results.keypoints.conf.cpu().numpy()

        # (B) 準備 deepsort 偵測輸入
        dets = []
        for box,score in zip(results.boxes.xyxy.cpu().numpy(), results.boxes.conf.cpu().numpy()):
            x1,y1,x2,y2 = box
            dets.append(([x1,y1,x2-x1,y2-y1], float(score), 'mouse'))
        tracks = self.tracker.update_tracks(dets, frame=frame)

        # (C) 處理每個 track
        for i, track in enumerate(tracks):
            if not track.is_confirmed() or i >= len(kps_xy):
                continue
            frame = self._process_track(
                frame, track, kps_xy[i], kps_conf[i], curr_frame
            )
        return frame

    def _process_track(self, frame, track, kps_xy, kps_conf, curr_frame):
        # draw bbox
        x,y,w,h = map(int, track.to_ltwh())
        cv2.rectangle(frame, (x,y), (x+w,y+h), self.line_color, 1)

        # 平滑 keypoints + skeleton
        valid = self._get_valid_kps(track, kps_xy, curr_frame)
        for a,b in self.skeleton:
            if a in valid and b in valid:
                cv2.line(frame, valid[a], valid[b], self.line_color, 1, cv2.LINE_AA)
        for idx,pt in valid.items():
            cv2.circle(frame, pt, 3, self.point_colors[idx], -1, cv2.LINE_AA)

        # 行為預測
        X = self._build_feature_vector(frame, valid)
        if X is not None:
            self.model_input_window.append(X)
            behavior, _ = self._predict_behavior()
            if behavior:
                cv2.putText(frame, behavior, (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1,cv2.LINE_AA)
        return frame


    def _get_valid_kps(self, track, kps_xy, curr_frame):
        """
        根據 history 做平滑：
        如果當前 keypoint 有值就存入 history，
        否則用 history 的平均值，
        如果 history 也沒值，則用 bbox center 當 fallback。
        """
        x, y, w, h = map(int, track.to_ltwh())
        bbox_center = (x + w // 2, y + h // 2)
        valid = {}
        for idx, (kx, ky) in enumerate(kps_xy):
            if kx > 0 and ky > 0:
                pt = (int(kx), int(ky))
                self.kp_history[idx].append(pt)
            if self.kp_history[idx]:
                avg = np.mean(self.kp_history[idx], axis=0)
                valid[idx] = (int(avg[0]), int(avg[1]))
            else:
                valid[idx] = bbox_center
        return valid

    def restore_and_normalize_keypoints(self, keypoints, orig_size, input_size=640):
        """
        keypoints: List of (x_resized, y_resized)
        orig_size: (w, h) of original frame
        回傳 List of (x_orig, y_orig, x_norm, y_norm)
        """
        orig_w, orig_h = orig_size
        inp = input_size
        scale = min(inp / orig_w, inp / orig_h)
        pad_w = (inp - orig_w * scale) / 2
        pad_h = (inp - orig_h * scale) / 2

        restored = []
        for x_res, y_res in keypoints:
            x_orig = (x_res - pad_w) / scale
            y_orig = (y_res - pad_h) / scale
            x_norm = min(max(x_orig / orig_w, 0.0), 1.0)
            y_norm = min(max(y_orig / orig_h, 0.0), 1.0)
            restored.append((x_orig, y_orig, x_norm, y_norm))
        return restored

    def calculate_velocity_acceleration(self, window):
        """
        window: deque of flat_kps_scaled vectors (每個 shape=(16,))
        回傳 shape=(32,) 的速度+加速度特徵，或 None
        """
        if len(window) < 7:
            return None
        p_t   = window[-1]
        p_t_1 = window[-4]
        p_t_2 = window[-7]
        v_t   = p_t   - p_t_1
        v_t1  = p_t_1 - p_t_2
        a_t   = v_t   - v_t1
        return np.concatenate([v_t, a_t], axis=0)

    def _build_feature_vector(self, frame, valid):
        """
        根據 valid 的 8 個 keypoints：
        1) restore & normalize → flat_norm (16,)
        2) min-max normalize → flat_kps_scaled (16,)
        3) velocity+acceleration → (32,)
        最後 concat 成 (64,) 的 X_full
        """
        if len(valid) != 8:
            return None

        # 1) restore & normalize
        pts = [valid[i] for i in range(8)]
        restored = self.restore_and_normalize_keypoints(
            pts, orig_size=(frame.shape[1], frame.shape[0]), input_size=self.pose_input_size
        )
        flat_norm = np.array([coord for _, _, x_n, y_n in restored for coord in (x_n, y_n)],
                             dtype=np.float32)

        # 2) min-max normalize
        flat_kpt = np.array([coord for pt in pts for coord in pt], dtype=np.float32)
        flat_scaled = (flat_kpt - self.X_min) / (self.X_max - self.X_min + 1e-6)
        self.pose_window.append(flat_scaled)

        # 3) velocity & acceleration
        if len(self.pose_window) >= 3:
            velacc = self.calculate_velocity_acceleration(self.pose_window)
            if velacc is not None:
                return np.concatenate([flat_norm, flat_scaled, velacc], axis=0)
        return None

    def _predict_behavior(self):
        """
        當 model_input_window 滿後，送進 BiLSTM 預測，並做平滑
        回傳 (final_label: str, top3: List[(label, prob)])
        """
        # shape → (channels=64, seq_len)
        window_np = np.array(self.model_input_window).T
        tensor = torch.tensor(window_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.behavior_model(tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
        self.behavior_probs_window.append(probs)

        # top3
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3 = [(self.behavior_names[i], float(round(probs[i], 4))) for i in top3_idx]

        # 平滑後取最後一幀
        if len(self.behavior_probs_window) == self.window_size:
            prob_window = np.array(self.behavior_probs_window)
            sm = self.smooth_predictions(
                prob_window, rest_index=self.behavior_names.index("rest")
            )
            final = self.behavior_names[int(sm[-1])]
        else:
            final = ""
        return final, top3

    def smooth_predictions(self, prob_window, rest_index: int):
        """
        prob_window: (T, C)
        回傳長度 T 的 label array (np.ndarray)
        """
        T, C = prob_window.shape
        raw = np.argmax(prob_window, axis=1)
        out = raw.copy()
        start = 0

        while start < T:
            lbl = raw[start]
            end = start + 1
            while end < T and raw[end] == lbl:
                end += 1
            length = end - start

            # rest 特殊處理
            if lbl == rest_index:
                rest_probs = prob_window[start:end, rest_index]
                low = rest_probs < self.rest_prob_margin
                if np.any(low):
                    top2 = np.argsort(prob_window[start:end], axis=1)[:, -2:]
                    for i, (a, b) in enumerate(top2):
                        if low[i]:
                            out[start + i] = a if b == rest_index else b

            # 其他的 duration 太短
            elif length < self.min_any_duration:
                prev_lbl = out[start-1] if start>0 else None
                next_lbl = raw[end]    if end<T   else None
                if prev_lbl is not None and prev_lbl == next_lbl:
                    fill = prev_lbl
                else:
                    left  = np.sum(raw[max(0, start-50):start] == prev_lbl) if prev_lbl is not None else 0
                    right = np.sum(raw[end:end+50] == next_lbl) if next_lbl is not None else 0
                    fill = prev_lbl if left>= right else next_lbl
                if fill is not None:
                    out[start:end] = fill

            start = end

        return out

