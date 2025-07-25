import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List
from .BiLSTM import BehaviorBiLSTM_v3
from .ST_TR import STTRNet, SimpleSTTR
from .ST_GCN import A_norm

class SimpleSTTR_BiLSTM(nn.Module):
    """
    STTR ➜ Bi-LSTM ➜ FC
    input: (B, F, T, 1) —— 與原 SimpleSTTR 相同
    """
    def __init__(self,
                 in_channels: int,        # feature dim (e.g. 70)
                 d_model: int      = 128,
                 num_heads: int    = 4,
                 num_classes: int  = 7,
                 lstm_hidden: int  = 64,
                 lstm_layers: int  = 2,
                 dropout_p: float  = 0.2):
        super().__init__()

        # --- Temprol  ---
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=1),
            nn.Flatten(2),                # (B, d_model, T)
            nn.Dropout(dropout_p)
        )

        # --- Transformer Encoder (STTR) ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model    = d_model,
            nhead      = num_heads,
            batch_first= True,
            dropout    = dropout_p
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=2
        )

        # --- 追加 Bi-LSTM ---
        # self.ln_in  = nn.LayerNorm(d_model)

        self.lstm = nn.LSTM(
            input_size  = d_model,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            bidirectional = True,
            dropout = dropout_p
        )

        # self.ln_out  = nn.LayerNorm(lstm_hidden*2)

        # --- 最終分類層 ---
        self.fc_out = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(lstm_hidden * 2, num_classes)
        )

    def forward(self, x):
        """
        x : (B, F, T, 1)   # 來自 STTRDataset，F=feat_dim(70)，T=seq_len(73)
        """

        # ----------------- Conv2d 投影 -----------------
        # 需求：Conv2d 要看到 (B, in_channels=F, H=T, W=1)
        # 目前 x 已經是 (B, F, T, 1)
        x = self.proj(x)                # (B, d_model=128, T, 1)

        # ------------- 壓掉最後一維，排成 (B,T,d_model) -------------
        x = x.squeeze(-1)               # (B, 128, T)
        x = x.permute(0, 2, 1)          # (B, T, 128)

        # -------- Transformer Encoder --------
        tx = self.transformer(x)         # (B, T, 128)
        res = tx + x

        # -------- Bi-LSTM --------
        # res = self.ln_in(res)
        lstm_out, _ = self.lstm(res)      # (B, T, 2*hidden)
        
        # lstm_out = self.ln_out(lstm_out)
        feat = lstm_out[:, -1, :]       # 取最後一步 (B, 2*hidden)

        # -------- 分類 --------
        logits = self.fc_out(feat)      # (B, num_classes)
        return logits

class SimpleSTTR_BiLSTM_v2(nn.Module):
    """
    混合模型：STTR ➜ Bi-LSTM ➜ SE ➜ Mean+Max 池化 ➜ FC Head
    輸入 x: (B, F, T, 1)，與原 SimpleSTTR 相同
    """
    def __init__(self,
                 in_channels: int,        # feature dim (e.g. 70)
                 d_model:     int  = 128,
                 sttr_dropout_p: int = 0.15,
                 num_heads:   int  = 4,
                 num_classes: int  = 7,
                 lstm_hidden: int  = 64,
                 lstm_layers: int  = 2,
                 se_ratio:    int = 16,
                 lstm_dropout_p: float  = 0.15
                 ):
        super().__init__()


        # --- 時序位置編碼 & 線性投影 ---
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=1),
            nn.Flatten(2),                # (B, d_model, T)
            nn.Dropout(sttr_dropout_p)
        )

        # --- Transformer Encoder  ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model    = d_model,
            nhead      = num_heads,
            batch_first= True,
            dropout    = sttr_dropout_p
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=2
        )

        # --- 追加 Bi-LSTM ---
        self.ln_in  = nn.LayerNorm(d_model)
        self.lstm = nn.LSTM(
            input_size  = d_model,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            bidirectional = True,
            dropout = lstm_dropout_p
        )
        self.ln_out  = nn.LayerNorm(lstm_hidden*2)

         # 3a) 殘差投影：把 transformer 輸出投影到 Bi-LSTM 輸出維度 (2H)
        self.resid_proj = nn.Linear(d_model, lstm_hidden*2, bias=False)

        # 4. SE 通道重標定 (在 time-axis pooling 之前作用於 (B,T,2H))
        chn = lstm_hidden * 2
        self.se_fc1 = nn.Linear(chn, chn//se_ratio, bias=False)
        self.se_fc2 = nn.Linear(chn//se_ratio, chn, bias=False)

        # 5. Mean + Max 池化後的 LayerNorm + Dropout
        self.dropout = nn.Dropout(lstm_dropout_p)
        self.ln_post = nn.LayerNorm(chn * 2)

        # 6. FC Head：先降維再分類
        self.hidden = nn.Linear(chn, lstm_hidden)
        self.fc     = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):
        """
        x: (B, F, T, 1)  → 投影 → (B, d_model, T) → (B, T, d_model)
           → Transformer → (B, T, d_model)
           → Bi-LSTM → (B, T, 2H)
           → SE + 池化 → (B, 4H) → FC Head → (B, num_classes)
        """
        # 1. 投影
        x = self.proj(x)             # (B, d_model, T, 1) → after flatten: (B, d_model, T)
        x = x.squeeze(-1).permute(0, 2, 1)  # (B, T, d_model)

        # 2. Transformer Encoder
        x_trans = self.transformer(x)      # (B, T, d_model)

        # 3. Bi-LSTM 前後文編碼
        x_ln = self.ln_in(x_trans)
        lstm_out, _ = self.lstm(x_ln)   # (B, T, 2H)
        out = self.ln_out(lstm_out)

        # --- 殘差 1：Transformer→Bi-LSTM ---
        # 先將 x_trans 投影到 (B,T,2H)，再加到 out
        res1 = self.resid_proj(x_trans)   # (B, T, 2H)
        out = out + res1

        # SE 前保留一份
        before_se = out.clone()

        # 4. SE 通道重標定
        #    通道壓縮→激活→還原，再以 sigmoid 生成權重
        se = out.mean(dim=1)         # (B, 2H)
        se = F.relu(self.se_fc1(se)) # (B, 2H/r)
        se = torch.sigmoid(self.se_fc2(se))  # (B, 2H)
        out = out * se.unsqueeze(1)  # (B, T, 2H)

        # --- 殘差 2：SE 前→SE 後 ---
        out = out + before_se             # (B, T, 2H)

        # # 5. Mean+Max 池化
        # mean_pool = out.mean(dim=1)            # (B, 2H)
        # max_pool  = out.max(dim=1).values      # (B, 2H)
        # pooled    = torch.cat([mean_pool, max_pool], dim=1)  # (B, 4H)

        # # 6. Head 前處理
        # pooled = self.ln_post(pooled)
        # pooled = self.dropout(pooled)
        # h = F.relu(self.hidden(pooled))        # (B, H)

        feat = out[:, -1, :]
        h = F.relu(self.hidden(feat))
        h = self.dropout(h)

        # 7. 最後分類
        logits = self.fc(h)                    # (B, num_classes)
        return logits

class STTR_BiLSTM(nn.Module):
    """
    3-stream ST-TR-BiLSTM 網路 (add BehaviorBiLSTM_v3)
    前段 ST-GCN → S-TR / T-TR 各 N 層 → Pool & Fuse → LogSoftmax

    ST-TR: 分析骨架; 
    BiLSTM: 分析骨架 + 速度、加速度等特徵

    將input: feature拆出前 16 dim (8 kpt * 2 xy) => ST-TR
    原始input => BiLSTM

    """
    def __init__(self,
                 in_channels : int,
                 num_classes : int,
                 stgcn_channels : List[int], # ex. [64, 128, 256]
                 sttr_dims :  List[int],
                 num_heads : int = 8,
                 tcn_kernel : int = 9,
                 dropout : float = 0.2,
                 num_layers : int = 3,

                 # BehaviorBiLSTM_v3 
                 lstm_input : int = 70,
                 lstm_hidden : int = 64,
                 lstm_layers : int =3,
                 se_ratio : int = 16,
                 dropout_p : int = 0.2
                 ):
        super().__init__()

        # (A) STTRNet 
        self.sttr = STTRNet(
            in_channels = in_channels,
            num_classes = num_classes,
            stgcn_channels = stgcn_channels ,
            sttr_dims= sttr_dims,
            num_heads   = num_heads,
            A_norm = A_norm,
            t_kernel  = tcn_kernel,
            dropout     = dropout,
            num_layers  = num_layers,
            use_conv_only= False
        )
        # (B) BehaviorBiLSTM_v3 
        self.bilstm = BehaviorBiLSTM_v3(
            input_dim  = lstm_input,
            hidden_dim = lstm_hidden,
            num_layers = lstm_layers,
            num_classes= num_classes,
            se_ratio   = se_ratio,
            dropout_p  = dropout_p
        )

    def forward(self, x_full : torch.Tensor) -> torch.Tensor:
        """
        x_node: (B, C = 2, T = win_size, V = 8);    V: 8個kpts, C: 一個v兩個dim (x,y)      
        x_full: (B, T, F)        # LSTM 要 (B, T, feat_dim)
        """
        # STTRNet + BiLSTM
        B, T, _ = x_full.shape                # (B, T, feat_dim)
        x_node = (
            x_full[..., :16]                  # (B, T, 16)
            .reshape(B, T, 2, 8)              # (B, T, C=2, V=8)
            .permute(0, 2, 1, 3).contiguous() # (B, C, T, V)
        )

        logits_sttr = self.sttr(x_node)
        logits_bilstm = self.bilstm(x_full)
        logits = logits_sttr + logits_bilstm 

        return logits

class LSTM_Transformer(nn.Module):
    """
    先 Bi-LSTM → 再 Transformer → Pooling → FC
    輸入 x: (B, F, T, 1)
    """
    def __init__(self,
                 in_channels: int,     # 原始特徵維度 F
                 d_model: int    = 128,# Transformer embed dim
                 num_heads: int  = 4,
                 lstm_hidden: int= 64,
                 lstm_layers: int= 2,
                 num_classes: int= 7,
                 dropout_p: float= 0.2):
        super().__init__()

        # 1) 投影層：把 F → d_model
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=1),
            nn.Flatten(2),     # (B, d_model, T)
            nn.Dropout(dropout_p)
        )

        # 2) Bi-LSTM
        self.lstm = nn.LSTM(
            input_size   = d_model,
            hidden_size  = lstm_hidden,
            num_layers   = lstm_layers,
            batch_first  = True,
            bidirectional= True,
            dropout      = dropout_p if lstm_layers>1 else 0.0
        )

        # 3) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model    = lstm_hidden*2,
            nhead      = num_heads,
            batch_first= True,
            dropout    = dropout_p
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 4) Pooling + FC
        self.norm = nn.LayerNorm(lstm_hidden*2)
        self.fc   = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(lstm_hidden*2, num_classes)
        )

    def forward(self, x):
        # x: (B, F, T, 1)
        x = self.proj(x)                # → (B, d_model, T)
        x = x.squeeze(-1).permute(0,2,1)# → (B, T, d_model)

        # Bi-LSTM
        lstm_out, _ = self.lstm(x)      # → (B, T, 2H)

        # Transformer
        trans_out = self.transformer(lstm_out)  # → (B, T, 2H)

        # 最後取 [B, 最後一幀, :] 或 mean pooling
        feat = trans_out[:, -1, :]      # (B, 2H)

        # 正規化 & 分類
        feat = self.norm(feat)
        logits = self.fc(feat)          # (B, num_classes)
        return logits



