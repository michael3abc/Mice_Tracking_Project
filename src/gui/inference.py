# inference.py
import cv2
import numpy as np
import sys, os
# import openvino as ov
import torch, pickle
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from collections import deque



src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) #前兩層
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from utils.build_feature import build_features
from utils.model_factory import build_model, SHAPE_SWITCH




class InferenceEngine:
    def __init__(self, cfg: dict, device: str | None = None):
        self.cfg    = cfg
        # self.device = device or ("cuda" if torch.cuda.is_available() else "xpu")
        auto_dev = None
        if device:
            auto_dev = device
        else:
            if torch.cuda.is_available():
                auto_dev = "cuda"
            elif torch.xpu.is_available():
                auto_dev = "xpu"            
            else:
                auto_dev = "cpu"
        self.device = auto_dev
        print(f"[InferenceEngine] 使用裝置：{self.device}")



            # --- 把區塊存在屬性，供其他 init 用 ---
        self.ycfg, self.pcfg, self.bcfg, self.paths = (
            cfg["yolo"], cfg["pose"], cfg["behavior"], cfg["paths"]
        )

        # 1. 影像檢測 + 追蹤
        self._init_yolo_tracker()

        # 2. 行為模型
        self._init_behavior_model()

        # 3. 其他超參與 deque
        self.vel_delta        = self.pcfg["vel_delta"]
        self.pose_history_len = self.pcfg["kp_history_len"]
        self.window_size      = self.bcfg["window_size"]
        self.predict_stride   = self.bcfg["predict_stride"]
        self.rest_prob_margin = self.bcfg["rest_prob_margin"]
        self.min_any_duration = self.bcfg["min_any_duration"]
        self.feature_dim      = self.bcfg["feature_dim"]
        self.look_ahead  = 0 


        self.pose_window_rel      = deque(maxlen=self.window_size)
        # self.pose_window_minMax   = deque(maxlen=self.window_size)
        self.behavior_probs_window= deque(maxlen=self.window_size)
        self.model_input_window   = deque(maxlen=self.window_size) 
        self.kp_history = [deque(maxlen=self.pcfg["kp_history_len"]) for _ in range(8)]

        self.raw20_queue          = deque(maxlen=self.window_size)

        self.display_scale = float(self.ycfg.get("display_scale", 2.0))  
        self._frame_counter = 0    
        
                      

        # 4. 初始化骨架資訊、kpts 顏色等
        self._init_visuals()

    def _init_yolo_tracker(self):
        self.yolo_conf      = self.ycfg["conf"]
        self.yolo_input_size= self.ycfg["input_size"]
        self.frame_size  = self.ycfg["orig_size"]
        self.orig_w, self.orig_h = self.frame_size

        weight_path = str(self.ycfg["weights"])
        
        if weight_path.endswith(".pt"):
            self.yolo = YOLO(weight_path)
            self.yolo.fuse()
            
            print("[INFO] YOLO 以 PyTorch 後端啟動")

        else:
            # **直接交給 Ultralytics 處理 OpenVINO**：指定後端與裝置即可
            print("[INFO] YOLO 以 OpenVINO 後端啟動，裝置：GPU")
            self.yolo = YOLO(str(weight_path), 
                             task="pose"
                             )



        # 目前沒用到 DeepSort
        self.tracker = DeepSort(max_age=10, max_iou_distance=0.7)

    def _init_behavior_model(self):
        exp_dir = self.paths["experiment_root"]      # ← 一個資料夾
        model_pth = os.path.join(exp_dir, self.paths["model_file"])
        # minmax_pz = os.path.join(exp_dir, self.paths["minmax_file"])
        le_pkl    = os.path.join(exp_dir, self.paths["labelenc_file"])

        # 1. 產生模型
        model_params = self.bcfg.get("model_params") or {}
        self.behavior_model = build_model(
            model_name = self.bcfg["model"],
            feat_dim    = self.bcfg["feature_dim"],
            num_classes = 7,
            model_params      = model_params
        )

        print(f"[INFO] 讀取行為模型權重: {model_pth}")
        ckpt = torch.load(model_pth, map_location="cpu")

        # --- ▸▸ 把 ckpt 轉成「純 state_dict」並去掉 'model.' prefix ◂◂ ---
        if isinstance(ckpt, dict) and "model" in ckpt:      # case 1: {"model": sd, ...}
            ckpt = ckpt["model"]

        # 若 key 依舊帶 "model." 前綴 → 全面移除
        if any(k.startswith("model.") for k in ckpt.keys()):
            ckpt = {k.replace("model.", ""): v for k, v in ckpt.items()}

        # 2) 確保 in_channels = feat_dim 全對齊
        missing, unexpected = self.behavior_model.load_state_dict(ckpt, strict=True)
        assert not missing and not unexpected, f"還有對不上的層：{missing} / {unexpected}"

        print("[INFO] 權重成功對齊 ✔")
        self.behavior_model.eval().to(self.device)
        # if self.device == "xpu" and ipex is not None:
        #     self.behavior_model = ipex.optimize(
        #        self.behavior_model,
        #        dtype=torch.float32,  # 可改用 torch.bfloat16
        #        level="O2"            # 可選 O1/O2/O3
        #    )
        #     print("[InferenceEngine] 已對行為模型套用 IPEX 優化")

        # === 載入 Min‧Max & LabelEncoder ===
        # npz = np.load(minmax_pz)
        # self.X_min = np.asarray(npz["X_min"], dtype=np.float32)
        # self.X_max = np.asarray(npz["X_max"], dtype=np.float32)
        # self.stats = {"mins": self.X_min, "maxs": self.X_max}


        with open(le_pkl, "rb") as f:
            self.le = pickle.load(f)
        self.behavior_names = self.le.classes_.tolist()
        print(self.behavior_names)

    def _init_visuals(self):
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

        # 1. 小鼠骨架(關鍵點)連線
        self.skeleton = [
                    (0,1),  # nose ↔ body
                    (3,1),  # front_right ↔ body
                    (5,7),  # rear_right ↔ hip
                    (7,6),  # hip ↔ rear_left
                    (4,1),  # front_left ↔ body
                    (2,7),  # tail_base ↔ hip
                    (7,1) ]  # hip ↔ body


        # 2. 關鍵點顏色
        self.point_colors = [
            (255,0,0),(0,255,0),(0,0,255),(128,128,0),
            (255,128,0),(0,255,255),(255,0,255),(128,0,255)
        ]
        self.line_color = (180,180,180)

        # 3. 行為種類
        self.rest_index = self.behavior_names.index("rest")

    def process_frame(self, frame: np.ndarray, curr_frame: int):
        # 1. YOLO 推論
        results = self.yolo(frame, conf=self.yolo_conf)[0]
        if len(results.boxes) == 0:
            return frame

        # 2. 只取第一隻偵測到的老鼠
        bbox = results.boxes.xyxy[0].cpu().numpy()   # shape (4,)
        kps_xy   = results.keypoints.xy[0].cpu().numpy()
        
        kps_conf = results.keypoints.conf[0].cpu().numpy()

        annotated = self._process_single_mouse(frame,bbox, kps_xy,kps_conf, curr_frame)

        annotated_disp = cv2.resize(annotated, None, fx=self.display_scale, fy=self.display_scale, interpolation=cv2.INTER_LINEAR)

        return annotated_disp    
   

    def _process_single_mouse(self, frame, bbox, kps_xy, kps_conf, curr_frame):
        # 1. 平滑並取得 valid skeleton
        valid = self._get_valid_kps(bbox, kps_xy)

        # 2. 計算 raw20 feature_vector，並推入 deque 
        if bbox is None:
            X20 = np.zeros(20, dtype=np.float32)
        else:
            X20 = self.make_raw20(bbox, valid)
        self.raw20_queue.append(X20)
        self._frame_counter += 1

        # 3. 當raw20 queue視窗滿，且每 stride 預測一次
        if ((self._frame_counter % self.predict_stride == 0) and (len(self.raw20_queue) == self.window_size)):

            X20 = np.stack(self.raw20_queue, axis=0)[None, ...]  # (1,T,20)

            X_full, _, _, _ = build_features(
                X         = X20,
                y         = ["dummy"],
                orig_size = (self.orig_w, self.orig_h),
                vel_delta = self.vel_delta,
                smooth_window_length = self.cfg["pose"]["smooth_window_length"],
                polyorder = self.cfg["pose"]["polyorder"],
                # stats= self.stats 
            )   # (1,T,F)
                                                           
            single_window = X_full[0]
            final_lbl, top3 = self._predict_behavior(single_window)
            if final_lbl:
                self._last_behavior = final_lbl

        # 4. 繪製結果
        # 4.1 放大影格
        s = float(self.display_scale)
        # 先放大原框／骨架畫布
        disp = cv2.resize(frame, None, fx=s, fy=s,
                        interpolation=cv2.INTER_LINEAR) if s!=1.0 else frame.copy()

        # 4.2 繪製 bbox
        x1i, y1i, x2i, y2i = [int(v * s) for v in bbox]
        cv2.rectangle(disp, (x1i, y1i), (x2i, y2i),
                    self.line_color, 1, cv2.LINE_AA)

        # 4.3 用 kps_conf 來決定哪個點可信（例如 > 0.3 就畫出來）
        CONF_TH = self.yolo_conf
        scaled_valid = {}
        for idx, (kx, ky) in valid.items():
            if kps_conf[idx] > CONF_TH:
                scaled_valid[idx] = (int(kx*s), int(ky*s))

        # 4.4 畫骨架連線
        for a, b in self.skeleton:
            if a in scaled_valid   and b in scaled_valid  :
                cv2.line(disp,
                        scaled_valid[a], scaled_valid[b],
                        self.line_color, 1, cv2.LINE_AA)
        # 4.5 畫所有 keypoints
        for idx, pt in scaled_valid.items():
            cv2.circle(disp, pt, 3, self.point_colors[idx],
                    -1, cv2.LINE_AA)

        # 5. 在畫面標註最新行為
        if hasattr(self, "_last_behavior"):
            cv2.putText(
                disp,
                self._last_behavior,
                (x1i, y1i - int(round(10 * s))),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        return disp

    def _get_valid_kps(self, bbox, kps_xy):
        """
        從yolo預測結果拿bbox, kpts位置
        若缺失 => 用上一幀的補
        仍缺失 => 用 box 中心

        bbox: (x1, y1, x2, y2)
        kps_xy: np.ndarray (8,2) 

        output: 補完之後的點 (因為behavior model不能吃 Nan)
        """
        x1, y1, x2, y2 = bbox
        cx, cy = (x1+x2)/2, (y1+y2)/2
        valid = {}
        for idx, (kx, ky) in enumerate(kps_xy):
            if kx > 0 and ky > 0:
                pt = (int(kx), int(ky))
                self.kp_history[idx].append(pt)
            if self.kp_history[idx]:
                avg = np.mean(self.kp_history[idx], axis=0)
                valid[idx] = (int(avg[0]), int(avg[1]))
            else:
                valid[idx] = (cx, cy)
        return valid

    def make_raw20(self,
                bbox: tuple[int,int,int,int],
                valid: dict[int,tuple[int,int]]
    ) -> np.ndarray:
        """
        接收經過 _get_valid_kps 平滑後的 valid，
        回傳原始size的 raw20 特徵：4 維 box + 16 維 keypoints
        """
        x1, y1, x2, y2 = bbox
        box_raw = [x1, y1, x2-x1, y2-y1]

        pts_raw = []
        for idx in range(8):
            kx, ky = valid[idx]
            pts_raw += [kx, ky]

        return np.array(box_raw + pts_raw, dtype=np.float32)


    def _prepare_input(self, win_np: np.ndarray) -> np.ndarray: 
        """
        input: (T, feat_dim)
        基於 SHAPE_SWITCH 的資訊 reshape input shape => fit model的輸入        
        """
        fn = SHAPE_SWITCH.get(self.bcfg["model"], SHAPE_SWITCH["default"])
        tensor = torch.tensor(win_np, dtype=torch.float32).to(self.device)
        out    = fn(tensor)
        # print(f"[DEBUG] model_in shape={out.shape}, dtype={out.dtype}, device={out.device}")
        return out
    
    def _predict_behavior(self, single_window: np.ndarray):
        """
        當 model_input_window 滿後，送進 行為模型 預測，並做平滑
        回傳 (final_label: str, top3: List[(label, prob)])
        """
        model_in = self._prepare_input(single_window)

        # 1. forward + softmax
        with torch.no_grad():
            logits = ( self.behavior_model(*model_in)              # tuple 版本
                    if isinstance(model_in, tuple)
                    else self.behavior_model(model_in) )        # 單 tensor 版本
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # 2. 推到queue (用於平滑預測)
        self.behavior_probs_window.append(probs)
        prob_seq  = np.array(self.behavior_probs_window)
        smoothed = self.smooth_predictions(prob_seq)

        if len(smoothed) <= self.look_ahead:
            return "", []
        
        mid_idx = -self.look_ahead-1
        final_id = int(smoothed[mid_idx])
        final    = self.behavior_names[final_id]

        # 4. top3
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3 = [(self.behavior_names[i], round(float(probs[i]), 4)) for i in top3_idx]

        # # 5. for debug
        # print(
        #     f"[DEBUG] logits[:5]={logits[0][:5].cpu().numpy()}, "
        #     f"probs[:5]={probs[:5]}",
        #     flush=True
        # )

        return final, top3

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

            # 1. 避免 短暫停留被判斷成rest => 用兩側行為補(相同)
            if lbl == REST_IDX:
                low = prob_arr[seg, lbl] < self.rest_prob_margin
                if low.any():
                    top2 = np.argsort(prob_arr[seg], axis=1)[:, -2:]   # 每幀後兩高
                    for i, (a, b) in enumerate(top2):
                        if low[i]:
                            out[t+i] = a if b == lbl else b

            # 2. 兩側行為不同 => 依兩側多數決補值 
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
        # if hasattr(self, 'pose_window_minMax'):
        #     self.pose_window_minMax.clear()
        if hasattr(self, 'pose_window_rel'):
            self.pose_window_rel.clear()
        self.model_input_window.clear()
        self.behavior_probs_window = []

    def predict_behavior_batch(self, X: np.ndarray):
        model_in = self._prepare_input(X)
        if self.device == "xpu":
            # ctx = torch.xpu.amp.autocast(enabled=True, dtype=torch.bfloat16)
            ctx = torch.no_grad()

        else:
            ctx = torch.no_grad()
            

        with ctx:
            # forward
            logits = ( self.behavior_model(*model_in)
                    if isinstance(model_in, tuple)
                    else self.behavior_model(model_in) )   # shape: (M, C)

            probs = torch.softmax(logits, dim=1).cpu().numpy()  # shape: (M, C)

        results = []
        for p in probs:
            top3_idx = np.argsort(p)[-3:][::-1]
            top3 = [(self.behavior_names[i], float(round(p[i], 4))) for i in top3_idx]
            results.append((top3[0][0], top3))  # top1 label 是 top3 的第一個

        return results





# =============== for API ===============
import yaml
from pathlib import Path

def load_config(cfg_path: str) -> dict:
    """讀入 yaml config，回傳 dict"""
    with open(cfg_path, encoding='utf-8') as f:
        return yaml.safe_load(f)

def make_engine(cfg_path: str, device: str | None = None) -> InferenceEngine:
    """
    讀 config → 建立並回傳 InferenceEngine
    """
    cfg = load_config(cfg_path)
    return InferenceEngine(cfg=cfg, device=device)

