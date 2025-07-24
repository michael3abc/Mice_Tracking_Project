import sys, os
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) #前兩層
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from models.BiLSTM import BehaviorBiLSTM_v3
from models.ST_TR import SimpleSTTR, STTRNet
from models.Hybird import SimpleSTTR_BiLSTM, STTR_BiLSTM , SimpleSTTR_BiLSTM_v2, LSTM_Transformer
from models.ST_GCN import BehaviorSTGCN_BiLSTM, A_norm
import torch

def build_model(model_name: str,
                feat_dim: int = 70,
                num_classes: int = 7,
                model_params: dict | None = None
                ):
    """依名稱回傳已初始化的模型 (含 device.to())"""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    name_map = {
        "SimpleSTTR": lambda: SimpleSTTR(feat_dim, **model_params).to(device),

        "SimpleSTTR_BiLSTM": lambda: SimpleSTTR_BiLSTM(feat_dim, **model_params).to(device),

        "SimpleSTTR_BiLSTM_v2": lambda : SimpleSTTR_BiLSTM_v2(feat_dim, **model_params).to(device),

        "LSTM_Transformer" : lambda : LSTM_Transformer(feat_dim, **model_params).to(device),


        "BehaviorSTGCN_BiLSTM": lambda: BehaviorSTGCN_BiLSTM(
                                            orig_feat_dim = feat_dim,
                                            num_classes   = num_classes,
                                            **model_params).to(device),

        "BehaviorBiLSTM_v3":   lambda: BehaviorBiLSTM_v3(
                                            input_dim   = feat_dim,
                                            num_classes = num_classes,
                                            **model_params).to(device),

        "STTRNet":              lambda: STTRNet(
                                in_channels     = feat_dim,
                                num_classes     = num_classes,
                                stgcn_channels  = model_params["stgcn_channels"],
                                sttr_dims       = model_params["sttr_dims"],
                                num_heads       = model_params["num_heads"],
                                A_norm          = A_norm,                    
                                t_kernel        = model_params["t_kernel"],
                                dropout         = model_params["dropout"],
                                num_layers      = model_params["num_layers"]
                            ).to(device),

        "STTR_BiLSTM": lambda: STTR_BiLSTM(
            in_channels = 2,
            **model_params                # 其餘參數直接展開
        ).to(device),                         
    }
    

    if model_name not in name_map:
        raise ValueError(f"Unknown model: {model_name}")
    
    return name_map[model_name]()


# orig: (T = win_size, F = feat_dim) -> model input shape
# B: batch size 在inference時 B = 1
SHAPE_SWITCH = {
    # ───── single-branch LSTM / ST-GCN 需要 (B = 1,T,F) ─────
    "BehaviorBiLSTM_v3":      lambda x: x.unsqueeze(0),       # (T,F) → (1,T,F)
    "BehaviorSTGCN_BiLSTM":   lambda x: x.unsqueeze(0),

    # ───── SimpleSTTR: (B = 1,feature_dim,T,1) ─────
    "SimpleSTTR_BiLSTM":   lambda x: x.permute(1, 0).unsqueeze(0).unsqueeze(-1),
    "SimpleSTTR_BiLSTM_v2":   lambda x: x.permute(1, 0).unsqueeze(0).unsqueeze(-1),

    # ───── ST-TR / ST-GCN route → (1,2,T,8) ─────
    "STTRNet":                lambda x: x[:, :16].reshape(-1,8,2).permute(2,0,1).unsqueeze(0),
    "LSTM_Transformer":       lambda x: x[:, :16].reshape(-1,8,2).permute(2,0,1).unsqueeze(0),

    # ───── 雙路 STTR_BiLSTM  ─────
    "STTR_BiLSTM":            lambda x: (
                                    x.unsqueeze(0),                                             # full : (1,T,70)
                                    x[:, :16].reshape(-1,8,2).permute(2,0,1).unsqueeze(0)       # kpts : (1,2,T,8)
                            ),

    # ───── 其餘未知模型 ─────
    "default":                lambda x: x.unsqueeze(0),    # (T,70)→(1,T,70)
}