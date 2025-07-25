import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# for simple STTR
from .blocks import SpatialSelfAttention, TemporalSelfAttention
class SimpleSTTR(nn.Module):
    """
    Spatial–Temporal Transformer (ST-TR) 主模型，
    包含通道升維、空間自注意力、時間自注意力、殘差連接與分類器。
    """
    def __init__(self, in_channels: int, d_model:int, num_heads: int, num_classes:int):
        """
        初始化 SimpleSTTR 模型。
        
        參數:
        - in_channels: 輸入通道數 (例如 kpts (x,y) 為 2)
        - d_model: 注意力機制的特徵維度 (Embed Dimension)
        - num_heads: 多頭注意力的頭數
        - num_classes: 最終分類的行為類別數
        """
        super(SimpleSTTR, self).__init__()
        # 1. 1x1 卷積 升維: in_channels → d_model
        self.embedding = nn.Conv2d(in_channels, d_model, kernel_size=1)
        # 2. 空間模組
        self.ssa = SpatialSelfAttention(d_model, num_heads)
        # 3. 時間模組
        self.tsa = TemporalSelfAttention(d_model, num_heads)

        # --- LayerNorm ---
        self.ln1 = nn.LayerNorm(d_model)   # after SSA
        self.ln2 = nn.LayerNorm(d_model)   # after TSA
        
        # pooling: (B, d_model, T, V) → (B, d_model, 1, 1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(d_model, num_classes)

    @staticmethod
    def _chan_last(x: torch.Tensor, fn):
        """
        將 (B, C, T, V) 轉為 (B*T*V, C)，
        執行 fn (通常是 LayerNorm) 後再轉回 (B, C, T, V)。
        """
        B, C, T, V = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, C)  # → (B*T*V, C)
        x = fn(x)                                           # LayerNorm
        return x.view(B, T, V, C).permute(0, 3, 1, 2)       # ← (B, C, T, V)
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播函式
        
        輸入:
        - x: 張量，形狀為 (B, in_channels, T, V)
        
        處理流程:
        1. 通道升維: Conv2d 1x1
        2. 空間自注意力 + 殘差
        3. 時間自注意力 + 殘差
        4. 全局平均池化
        5. 分類層線性映射

        * 殘差連接: 把錢一層輸入X，直接加到新的輸出 => 避免梯度消失
        * avg pooling: 把該輸入在時間維度跟V節點做平勳 => 避免偏重單一節點、減少計算量
        
        輸出:
        - out: 張量，形狀為 (B, num_classes)
        """

        x = self.embedding(x) # (B, d_model, T, V)
        x_ssa = self.ssa(x)   # 空間        
        x = x_ssa+ x          # 殘差連接
        x = self._chan_last(x, self.ln1) # LayerNorm

        x_tsa = self.tsa(x)         # 時間
        x = x_tsa + x               # 殘差連接
        x = self._chan_last(x, self.ln2) # LayerNorm


        x = self.pool(x)            # pooling -> (B, d_model, 1, 1)
        x = x.view(x.size(0), -1)   # (B, d_model)
        out = self.classifier(x)    # (B, num_class)
        return out

# STTRNet
from .blocks import STRBlock, TTRBlock, ConvOnlyBlock, SimpleSTGCNBlock
class STTRNet(nn.Module):
    """
    2-stream ST-TR 網路
    前段 ST-GCN → S-TR / T-TR 各 N 層 → Pool & Fuse → LogSoftmax
    """
    def __init__(self,
                 in_channels: int,
                 num_classes: int,                 
                 stgcn_channels: List[int],         # 前段 ST‐GCN 的channel;        e.g. [64, 128, 256]              
                 sttr_dims:        List[int],       # S-TR / T-TR 的 d_model 層數;  e.g. [64, 128, 256]
                 num_heads:       int,                 
                 A_norm:          torch.Tensor,     # STGCNBlock 需要的骨架矩陣
                 
                 t_kernel:        int     = 3,      # SimpleSTGCNBlock 內的 temporal kernel
                 dropout:         float   = 0.1,
                 num_layers:      int     = 3,
                 use_conv_only = False
                 ):
        super().__init__()

        self.register_buffer('A_norm', A_norm)  # shape: (V, V)

        # ======== 1. ST-GCN 抽取特徵: ========
        if use_conv_only:      # 你可以在 cfg 裡加一個 flag
            gcn_block = ConvOnlyBlock
        else:
            gcn_block = SimpleSTGCNBlock

        self.backbone = nn.ModuleList()
        prev = in_channels  # 記錄前一層的output_channels
        for c in stgcn_channels:
            self.backbone.append(
                gcn_block(
                    in_channels  = prev,
                    out_channels = c,
                    A_norm       = self.A_norm,
                    t_kernel     = t_kernel
                ))
            prev = c        # 這裡更新: 這層out -> 下層 in

        assert len(sttr_dims) == num_layers

        # ======== 2. S-TR (Spatial Transformer Stream) ========
        self.s_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.s_blocks.append(
                STRBlock(d_model= sttr_dims[i], 
                        num_head = num_heads, 
                        tcn_kernel=t_kernel,
                        dropout=dropout)
            )          
                  

        # ======== 3. T-TR Temporal Transformer Stream) ========
        self.t_blocks = nn.ModuleList()
        for i in range(num_layers):
            in_c  = sttr_dims[i-1] if i>0 else stgcn_channels[-1]
            out_c = sttr_dims[i]
            self.t_blocks.append(
                TTRBlock(in_channels  = in_c,
                         out_channels = out_c,
                         num_heads    = num_heads,
                         dropout      = dropout)
            )
        
        self.fc = nn.Linear(sttr_dims[-1], num_classes)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V)
        # ST-GCN 特徵抽取
        for g in self.backbone:
            x = F.relu(g(x))
        
        s = x.clone()
        t = x.clone()
        for sb, tb in zip(self.s_blocks, self.t_blocks):
            s = sb(s)
            t = tb(t)
        
        s = s.mean(dim=[2, 3])
        t = t.mean(dim=[2, 3])

        logits = self.fc(s) + self.fc(t)
        logits = F.log_softmax(logits, dim = 1)
        return logits

