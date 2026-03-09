
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

# ==============================================GCN + LSTM==================================================================================

# 1) 建立相鄰矩陣 A => A_norm

V = 8
skeleton = [
    (0,1), (3,1), (5,7),
    (7,6), (4,1), (2,7),
    (7,1),
]

A = np.zeros((V, V), dtype=np.float32)
for i, j in skeleton:
    A[i, j] = A[j, i] = 1.0
for i in range(V):
    A[i, i] = 1.0


# 1. let D: diagonal matrix，each entries = each row.sum()
# 2. take D^(-1/2), s.t. D A_norm D = A
# => A_norm  = D^(-1/2)AD^(-1/2), s.t. for all eigenvalue in [-1, 1]

D = A.sum(axis=1)
D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-6))
A_norm_np = D_inv_sqrt @ A @ D_inv_sqrt     # @ 矩陣乘法

# 轉成 torch.Tensor
A_norm = torch.from_numpy(A_norm_np).float()  # shape: (V, V)

# 2) ST-GCN Block 
class STGCNBlock(nn.Module):
    """
    ST-GCN Block：Graph Conv（空间） + Temporal Conv + 殘差
    输入 x: (B, C_in, T, V) : (batch_size, 輸入通道 = 2 (x,y), frame數, 節點數 = 8)
    输出   (B, C_out, T, V)
    
    整個 Block 的功能：
    1. 空間圖卷積 (Graph Convolution)：利用圖結構 A_norm 聚合鄰居節點資訊
    2. 時間卷積 (Temporal Convolution)：在時間軸上做 1D 卷積，捕捉動態變化
    3. 殘差連接 (Residual)：引入 shortcut 分支，使輸入能跨層傳遞以穩定訓練
    """
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 A_norm, 
                 t_kernel_size=9, 
                 t_stride=1, 
                 dropout=0.5):
        super().__init__()
        # 作為非學習參數load到GPU
        self.register_buffer('A_norm', A_norm)

        # 空间图卷积：1x1 Conv + 邻接相乘
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # TCN層
        padding = (t_kernel_size - 1)//2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=(t_kernel_size,1),
                      stride=(t_stride,1),
                      padding=(padding,0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )

        # 残差分支
        if in_channels != out_channels or t_stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(t_stride,1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        # x: (B, C_in, T, V)
        # 1) 空间卷积
        x_sp = torch.einsum('vw,bctw->bctv', self.A_norm, x)
        x_sp = self.gcn(x_sp)
        # 2) 时间卷积
        y = self.tcn(x_sp)
        # 3) 残差 + 激活
        res = self.residual(x)
        return F.relu(y + res)

# 3) 行为识别模型：ST-GCN + Bi-LSTM + SE + Pool + FC
class BehaviorSTGCN_BiLSTM(nn.Module):
    def __init__(self,
                 orig_feat_dim=70,
                 num_classes=7,
                 gcn_channels=[16,32],
                 lstm_hidden=64,
                 lstm_layers=3,
                 se_ratio=16,
                 dropout_p=0.2):
        super().__init__()
        # 注册归一化邻接
        self.register_buffer('A_norm', A_norm)

        # —— ST-GCN 分支 —— 
        in_c = 2  # 每节点特征维度 = (x, y)
        self.stgcn_blocks = nn.ModuleList()
        for out_c in gcn_channels:
            self.stgcn_blocks.append(
                STGCNBlock(in_c, out_c, A_norm,
                           t_kernel_size=9, t_stride=1, dropout=dropout_p)
            )
            in_c = out_c
        self.gcn_out_c = gcn_channels[-1]

        # —— Bi-LSTM 分支 —— 
        self.lstm = nn.LSTM(
            input_size=self.gcn_out_c + orig_feat_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_p
        )
        chn = lstm_hidden * 2  # 因为双向
        # SE 模块
        self.se_fc1 = nn.Linear(chn, chn//se_ratio, bias=False)
        self.se_fc2 = nn.Linear(chn//se_ratio, chn, bias=False)
        # 池化后 FC 头
        self.ln_post = nn.LayerNorm(chn*2)
        self.dropout = nn.Dropout(dropout_p)
        self.hidden = nn.Linear(chn*2, lstm_hidden)
        self.fc     = nn.Linear(lstm_hidden, num_classes)

    def forward(self, X_full):
        """
        X_full: (B, T, orig_feat_dim)
                 前16维是每帧 8 个关节点的 (x,y) 归一化坐标
        """
        if X_full.shape[1] == self.lstm.input_size - self.gcn_out_c:
            X_full = X_full.permute(0, 2, 1)

        B, T, feat_dim = X_full.shape
        # —— ST-GCN 分支 —— 
        x_rel = X_full[:, :, :16]         # (B, T, 16)
        x_rel = x_rel.view(B, T, 8, 2)    # (B, T, V=8, 2)
        x_rel = x_rel.permute(0, 3, 1, 2) # (B, 2, T, 8)
        # 过 GCNBlock
        for blk in self.stgcn_blocks:
            x_rel = blk(x_rel)            # (B, Cg, T, V)
        x_gcn = x_rel.mean(-1)           # (B, Cg, T)
        x_gcn = x_gcn.permute(0,2,1)     # (B, T, Cg)


        # —— 融合 & LSTM —— 
        X_cat = torch.cat([X_full, x_gcn], dim=2)  # (B, T, orig+Cg)
        out, _ = self.lstm(X_cat)                 # (B, T, 2H)

        # —— SE 模块 —— 
        se = out.mean(1)                           # (B, 2H)
        se = F.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se))       # (B, 2H)
        out = out * se.unsqueeze(1)               # (B, T, 2H)

        # —— 池化 & 分类头 —— 
        mean_p = out.mean(1)                      # (B,2H)
        max_p  = out.max(1).values                # (B,2H)
        pooled = torch.cat([mean_p, max_p], dim=1)  # (B,4H)

        h = self.ln_post(pooled)
        h = self.dropout(h)
        h = F.relu(self.hidden(h))
        logits = self.fc(h)                       # (B,num_classes)
        return logits

