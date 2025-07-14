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

from model.Behaviors_Models import Behavior1D_CNN, BehaviorBiLSTM, BehaviorBiLSTM_v3, BehaviorTransformer
from model.utils import compute_space_distances, compute_direction_unit, compute_speed_std, compute_velocity_acc, compute_cos_np
from model import Behaviors_Models
os.chdir(r"C:\Users\micha\Desktop\Mice_tracking_project")


class InferenceEngine:
    def __init__(
        self,
        yolo_weights: str,
        yolo_conf: float,
        
        pose_input_size: int,  
        orig_size : list,    

        kp_history_len: int,
        vel_delta : int,
        pose_class_path : str,

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
        self.display_scale = 2.0

        self.yolo_input_size = pose_input_size

        self.pose_window_rel = deque(maxlen=window_size)   # 存 rel-norm 序列
        self.pose_window_minMax = deque(maxlen=window_size)   # 存 min-max 序列

        self.model_input_window   = deque(maxlen=window_size)
        self.behavior_probs_window = []

        self.frame_size = tuple(orig_size)     # (width, height)

        self.kp_history_len = kp_history_len
        self.vel_delta = vel_delta
        self.pose_class = np.load(pose_class_path, allow_pickle=True).tolist()

        self.window_size = window_size
        self.rest_prob_margin = rest_prob_margin
        self.min_any_duration = min_any_duration
        

        # 2. 載入模型
        self.behavior_weights = behavior_weights
        self.behavior_model_name = behavior_model
        self._init_models(yolo_weights, behavior_weights, minmax_npz)

        # 3. 狀態維護
        self.kp_history = [deque(maxlen=kp_history_len) for _ in range(8)]
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
        if ModelClass.__name__ == "BehaviorTransformer":
            num_classes = 7
            self.behavior_model = ModelClass(
                feature_dim=70,
                d_model=64,
                nhead=8,
                num_layers=1,
                num_classes=num_classes,
                dropout=0.1
            ).to(self.device)
        elif ModelClass.__name__ == "BehaviorBiLSTM_v3":
            # BiLSTM_v3 要指定 input_dim
            # 这里 71 = 16(norm)+16(scale)+32(vel/acc)+1(cos)+3(space)+2(dir)+1(std)
            self.behavior_model = ModelClass(
                input_dim=70,
                hidden_dim=64,
                num_layers=3,
                num_classes=7,
                se_ratio=16,
                dropout_p=0.2
            ).to(self.device)
        else:
            self.behavior_model = ModelClass().to(self.device)

        state = torch.load(beh_w, map_location="cpu")
        self.behavior_model.load_state_dict(state)
        self.behavior_model.eval()

        # MinMax normalization parameters
        npz = np.load(minmax_npz)
        self.X_min = npz["X_min"].reshape(-1)
        self.X_max = npz["X_max"].reshape(-1)

    def _init_visuals(self):
        #小鼠骨架(關鍵點)連線

        '''
        keypoint_to_name = {
        0: 'nose',
        1: 'body',
        2: 'tail_base',
        3: 'front_right',
        4: 'front_left',
        5: 'rear_right',
        6: 'rear_left',
        7: 'hip'
        }

        flip_idx: [0, 1, 2, 4, 3, 6, 5, 7]
        '''
        self.skeleton = [
                    (0,1),  # nose ↔ body
                    (3,1),  # front_right ↔ body
                    (5,7),  # rear_right ↔ hip
                    (7,6),  # hip ↔ rear_left
                    (4,1),  # front_left ↔ body
                    (2,7),  # tail_base ↔ hip
                    (7,1) ]  # hip ↔ body

        # self.skeleton = [(0,1),(1,7),(1,3),(1,4),(7,5),(7,6),(2,7)]
        #關鍵點顏色
        self.point_colors = [
            (255,0,0),(0,255,0),(0,0,255),(128,128,0),
            (255,128,0),(0,255,255),(255,0,255),(128,0,255)
        ]
        self.line_color = (180,180,180)
        #行為種類
        self.behavior_names = ['eat','groom','hang','micromovement','rear','rest','walk']
        # self.behavior_names = self.pose_class
        self.rest_index = self.behavior_names.index("rest")


    '''
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

        # (B) 準備 deepsort 偵測輸入 (於一隻鼠沒差，但未來可用在預測多隻鼠)
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
    '''
    def process_frame(self, frame: np.ndarray, curr_frame: int):
        # (A) YOLO 推論
        results = self.yolo(frame, conf=self.yolo_conf)[0]
        if len(results.boxes) == 0:
            return frame

        # 只取第一隻偵測到的老鼠
        bbox = results.boxes.xyxy[0].cpu().numpy()   # shape (4,)
        kps_xy   = results.keypoints.xy[0].cpu().numpy()
        kps_conf = results.keypoints.conf[0].cpu().numpy()

        annotated = self._process_single_mouse(frame,bbox, kps_xy,kps_conf, curr_frame)

        annotated_disp = cv2.resize(annotated, None, fx=self.display_scale, fy=self.display_scale, interpolation=cv2.INTER_LINEAR)


        # # 把框跟關鍵點一起送去處理
        # frame = self._process_single_mouse(frame, bbox, kps_xy, kps_conf, curr_frame)
        return annotated_disp
    
    def _process_single_mouse(self,
                            frame: np.ndarray,
                            bbox: np.ndarray,
                            kps_xy: np.ndarray,
                            kps_conf: np.ndarray,
                            curr_frame: int) -> np.ndarray:
        """
        frame      : 原始 (未放大) 影格，BGR
        bbox       : ndarray [x1, y1, x2, y2]，YOLO 輸出之原尺寸座標
        kps_xy     : ndarray (8, 2)，YOLO 關鍵點   ── 原尺寸
        kps_conf   : ndarray (8,)   ，關鍵點置信度 ── (目前未用，可之後做 keypoint mask)
        curr_frame : 目前影格索引 (for history)
        回傳       : 已放大、且重畫骨架/框/行為文字的影格，BGR
        """
        # ─────────────────────────────────────────────────────────────
        # 1. 以「原尺寸」計算 bbox 與關鍵點
        # ─────────────────────────────────────────────────────────────
        x1o, y1o, x2o, y2o = map(int, bbox)           # original bbox
        w0, h0 = x2o - x1o, y2o - y1o

        # 更新 / 平滑 keypoints，存 history
        valid = self._get_valid_kps((x1o, y1o, w0, h0), kps_xy, curr_frame)

        # ─────────────────────────────────────────────────────────────
        # 2. 行為特徵 & 預測 (仍用原尺寸座標)
        # ─────────────────────────────────────────────────────────────
        feat = self._build_feature_vector(frame, valid)
        if feat is not None:
            self.model_input_window.append(feat)
            behavior, _ = self._predict_behavior()
        else:                                  # 尚未收滿窗口時也補 0，保持時序長度
            self.model_input_window.append(np.zeros(70, dtype=np.float32))
            behavior = ""

        # ─────────────────────────────────────────────────────────────
        # 3. 放大影格供顯示；所有待畫座標 × scale
        # ─────────────────────────────────────────────────────────────
        s = float(self.display_scale)          # e.g. 2.0
        disp = (cv2.resize(frame, None, fx=s, fy=s,
                        interpolation=cv2.INTER_LINEAR)
                if s != 1.0 else frame.copy())

        # 放大後的 bbox 與線寬
        x1, y1, x2, y2 = [int(v * s) for v in (x1o, y1o, x2o, y2o)]
        # line_w = max(1, int(round(1 * s)))     # 線條寬度隨 scale 放大
        cv2.rectangle(disp, (x1, y1), (x2, y2),
                    self.line_color, 1, cv2.LINE_AA)

        # 放大後的 keypoint 座標
        scaled_valid = {i: (int(pt[0] * s), int(pt[1] * s))
                        for i, pt in valid.items()}

        # 畫骨架
        for a, b in self.skeleton:
            if a in scaled_valid and b in scaled_valid:
                cv2.line(disp, scaled_valid[a], scaled_valid[b],
                        self.line_color, 1, cv2.LINE_AA)

        # 畫 keypoints        
        for idx, pt in scaled_valid.items():
            cv2.circle(disp, pt, 3, self.point_colors[idx], -1, cv2.LINE_AA)

        # 畫行為文字（若已預測出行為）
        if behavior:
            cv2.putText(disp, behavior,
                        (x1, y1 - int(round(10 * s))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255),
                        1, cv2.LINE_AA)

        # 回傳「放大 + 已重新繪製」的影格，給 GUI 顯示
        return disp

    
    def _get_valid_kps(self, bbox, kps_xy, curr_frame):
        """
        bbox: (x, y, w, h)
        kps_xy: np.ndarray (8,2) 
        """
        x, y, w, h = bbox
        bbox_center = (x + w//2, y + h//2)
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




    def restore_and_normalize_keypoints(self, keypoints, input_size=640):
        """
        orig_size: 影片原始長寬
        input_size: YOLO 輸入邊長
        keypoints: List of (x_resized, y_resized)
        orig_size: (w, h) of original frame
        回傳 List of (x_orig, y_orig, x_norm, y_norm)
        """
        orig_w, orig_h = self.frame_size
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

    # def calculate_velocity_acceleration(self, window):
    #     """
    #     window: deque of flat_kps_scaled vectors (每個 shape=(16,))
    #     回傳 shape=(32,) 的速度+加速度特徵，或 None
    #     """
    #     if len(window) < 7:
    #         return None
    #     p_t   = window[-1]
    #     p_t_1 = window[-self.vel_delta]
    #     p_t_2 = window[-2*self.vel_delta]
    #     v_t   = p_t   - p_t_1
    #     v_t1  = p_t_1 - p_t_2
    #     a_t   = v_t   - v_t1
    #     return np.concatenate([v_t, a_t], axis=0)

    # def compute_head_body_tail_cos(kpts):
    #     """
    #     kpts: Tensor of shape [seq_len, 8, 2]，第 8 個點順序為:
    #     0:nose, 1:body, 2:tail_base, ... 其餘點可以忽略
    #     回傳: Tensor of shape [seq_len, 1]，每個時間步的 cos(angle)
    #     """
        
    #     nose, body, tail_base = kpts[0], kpts[1], kpts[2]
    #     v1, v2 = (body - nose), (body - tail_base)
    #     # 計算餘弦： (v1·v2) / (||v1|| * ||v2||)
    #     dot = torch.sum(v1 * v2, dim=1, keepdim=True)  # [seq_len, 1]
    #     norm = torch.norm(v1, dim=1, keepdim=True) * torch.norm(v2, dim=1, keepdim=True)  # [seq_len,1]
    #     cos_angle = dot / (norm + 1e-6)  # 避免除以 0

    #     return cos_angle  # [seq_len, 1]

    def _build_feature_vector(self, frame, valid):
        if len(valid) != 8:
            return None

        # 1) restore & normalize → flat_norm (16,)
        pts = [valid[i] for i in range(8)]
        restored = self.restore_and_normalize_keypoints(pts, input_size=self.yolo_input_size)
        flat_rel   = np.array([coord for *_, x_n, y_n in restored 
                                for coord in (x_n, y_n)], dtype=np.float32)  # (16,)


        # 2) min-max normalize → flat_scaled (16,)
        flat_kpt = np.array([coord for pt in pts for coord in pt], dtype=np.float32)
        flat_minMax = (flat_kpt - self.X_min) / (self.X_max - self.X_min + 1e-6)
        
        self.pose_window_rel.append(flat_rel)
        self.pose_window_minMax.append(flat_minMax)
 
        if len(self.pose_window_minMax) < 2*self.vel_delta+1:
            return None
        
        # 調用utils function
        # 3) velocity & acceleration (32,) 
        X_minMax = np.stack(self.pose_window_minMax, axis=0)[None,...]  # (1, T, 16)
        vel, acc = compute_velocity_acc(X_minMax, delta=self.vel_delta)
        vel, acc = vel[0], acc[0]
        velacc   = np.concatenate([vel[-1], acc[-1]], axis=0)     # (32,)

        # 4) cos(angle) on relative coords
        X_rel = np.stack(self.pose_window_rel, axis=0)               # (T,16)
        kpt_seq = X_rel.reshape(-1, 8, 2)                            # (T,8,2)
        cos_seq = compute_cos_np(kpt_seq)                         # (T,1)
        cos_val = float(cos_seq[-1,0])                            # scalar

        # 5) 空間距離 on relative coords
        space_seq = compute_space_distances(X_rel[None,...])         # (1, T, 2)
        space_val = space_seq[0,-1]                               # (2,)

        # 6) 方向單位向量
        dir_seq = compute_direction_unit(vel[None, ...])                        # → (1,T,2)
        dir_val = dir_seq[0, -1]                                                # → (2,)

        # ———— 7) 速度全局 std ————
        std_seq = compute_speed_std(vel[None, ...])                             # → (1,T,1)
        std_val = float(std_seq[0, -1, 0])                     

        return np.concatenate([
            flat_rel,           # 16
            flat_minMax,         # 16
            velacc,              # 32
            [cos_val],           # 1
            space_val,           # 2
            dir_val,             # 2
            [std_val],           # 1
        ], axis=0)               # → 16+16+32+1+2+2+1 = 70
    


    def _predict_behavior(self):
        """
        當 model_input_window 滿後，送進 BiLSTM 預測，並做平滑
        回傳 (final_label: str, top3: List[(label, prob)])
        """
        if len(self.model_input_window) < self.window_size:
            return "", []          # ← 還沒滿就什麼都不顯示
        
        # # shape → (channels=64, seq_len)
        # window_np = np.array(self.model_input_window).T
        # tensor = torch.tensor(window_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        # Transformer 需要 (batch, seq_len, feature_dim=65)

        window_np = np.stack(self.model_input_window, axis=0)      # (T, 70)
        tensor = torch.tensor(window_np, dtype=torch.float32).unsqueeze(0)  # (1,T,70)
        if isinstance(self.behavior_model, BehaviorBiLSTM_v3):
            tensor = tensor.permute(0, 2, 1)                        # (1,70,T)
        tensor = tensor.to(self.device)
        
        # ③ forward + softmax
        with torch.no_grad():
            probs  = torch.softmax(self.behavior_model(tensor), dim=1)[0].cpu().numpy()
        self.behavior_probs_window.append(probs)

        prob_seq  = np.array(self.behavior_probs_window)
        smoothed = self.smooth_predictions(prob_seq)
        final = self.behavior_names[int(smoothed[-1])]

        # top3
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3 = [(self.behavior_names[i], float(round(probs[i], 4))) for i in top3_idx]

        return final, top3

    # def smooth_predictions(self, prob_window, rest_index: int):
    #     """
    #     prob_window: (T, C)
    #     回傳長度 T 的 label array (np.ndarray)
    #     """
    #     T, C = prob_window.shape
    #     raw = np.argmax(prob_window, axis=1)
    #     out = raw.copy()
    #     start = 0

    #     while start < T:
    #         lbl = raw[start]
    #         end = start + 1
    #         while end < T and raw[end] == lbl:
    #             end += 1
    #         length = end - start

    #         # rest 特殊處理
    #         if lbl == rest_index:
    #             rest_probs = prob_window[start:end, rest_index]
    #             low = rest_probs < self.rest_prob_margin
    #             if np.any(low):
    #                 top2 = np.argsort(prob_window[start:end], axis=1)[:, -2:]
    #                 for i, (a, b) in enumerate(top2):
    #                     if low[i]:
    #                         out[start + i] = a if b == rest_index else b

    #         # 其他的 duration 太短
    #         elif length < self.min_any_duration:
    #             prev_lbl = out[start-1] if start>0 else None
    #             next_lbl = raw[end]    if end<T   else None
    #             if prev_lbl is not None and prev_lbl == next_lbl:
    #                 fill = prev_lbl
    #             else:
    #                 left  = np.sum(raw[max(0, start-50):start] == prev_lbl) if prev_lbl is not None else 0
    #                 right = np.sum(raw[end:end+50] == next_lbl) if next_lbl is not None else 0
    #                 fill = prev_lbl if left>= right else next_lbl
    #             if fill is not None:
    #                 out[start:end] = fill

    #         start = end

    #     return out

    # inference.py ── 放在 InferenceEngine 類別內（取代舊 smooth_predictions）

    def smooth_predictions(self, prob_arr: np.ndarray) -> np.ndarray:
        """
        prob_arr : shape (T, C)
        回傳     : 長度 T 的 label array
        """
        REST_IDX  = self.rest_index          # ← 在 __init__ 設 self.rest_index
        raw  = prob_arr.argmax(1)
        out  = raw.copy()
        t = 0
        while t < len(raw):
            lbl = raw[t]
            e   = t + 1
            while e < len(raw) and raw[e] == lbl:
                e += 1
            seg = slice(t, e)
            L   = e - t

            # ── 1) rest 類別特殊門檻 ───────────────────────
            if lbl == REST_IDX:
                low = prob_arr[seg, lbl] < self.rest_prob_margin
                if low.any():
                    top2 = np.argsort(prob_arr[seg], axis=1)[:, -2:]   # 每幀後兩高
                    for i, (a, b) in enumerate(top2):
                        if low[i]:
                            out[t+i] = a if b == lbl else b

            # ── 2) 片段太短 → 依兩側多數決補值 ─────────────
            elif L < self.min_any_duration:
                pl = out[t-1]  if t > 0          else None   # previous label
                nl = raw[e]    if e < len(raw)   else None   # next label
                if pl == nl and pl is not None:
                    fill = pl
                else:
                    left  = (raw[max(0, t-50):t]   == pl).sum() if pl is not None else 0
                    right = (raw[e:e+50]           == nl).sum() if nl is not None else 0
                    fill  = pl if left >= right else nl
                if fill is not None:
                    out[seg] = fill
            # ──────────────────────────────────────────────
            t = e
        return out


    def reset(self):
        # 清空所有狀態 buffer
        for dq in self.kp_history:
            dq.clear()
        # 如果你還有額外存 flat_scaled / flat_norm 的 deque
        if hasattr(self, 'pose_window_minMax'):
            self.pose_window_minMax.clear()
        if hasattr(self, 'pose_window_rel'):
            self.pose_window_rel.clear()
        self.model_input_window.clear()
        self.behavior_probs_window = []