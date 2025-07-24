import torch
import torch.nn as nn
import torch.nn.functional as F
from .ST_GCN import A_norm
from einops import rearrange


# ============================================== Attention ==============================================
# 1. Spatial
class SpatialSelfAttention(nn.Module):
    """
    Spatial_Self_Attention模組，偵測同一幀內不同節點之間的關係。
    """
    def __init__(self, d_model: int, num_heads:int ):
        '''
        d_model: 每個 token 的向量維度 (輸入輸出維度)
        num_heads: 拆成幾個子空間，各自內部做attention
        (num_heads要能整除d_model)
        '''
        super(SpatialSelfAttention, self).__init__()
  
        #pytorch內建mutihead模組: 輸入為 (B, T, F)
        self.attn = nn.MultiheadAttention(embed_dim=d_model,
                                          num_heads=num_heads,
                                          batch_first=True)
        
        self.ln1 = nn.LayerNorm(d_model)  # 在 reshape → attn 前
        self.ln2 = nn.LayerNorm(d_model)  # 在 attn → 還原 後
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: input tensor，形狀為 (B, d_model, T, V = 8)
            B: 批次大小
            d_model: 通道數（注意力維度）
            T: 時序長度（時間步數）
            V: 節點數（骨架關節數）

        step:
        1. 取得輸入X.shape = (B, D, T, V)
        2. 轉為 (B*T, V, D) 把時間做為batch分組，在不同V上 (把所有時間點展平為 batch)
        3. self attn => attn_output (針對每一個 frame 裡的 V 個關鍵點做 Attention)
        4. 將output還原成出入格式 (B, d_model, T, V)
        '''

        B, D, T, V = x.shape
        # 1. x2 = x.permute(0,2,3,1).reshape(B*T, V, C)
        x2 = rearrange(x, 'b c t v -> (b t) v c')        
        # 2. Pre‐Norm: 先對每個節點的特徵做正規化
        x2 = self.ln1(x2)
        # 3. self-Attention
        y2, _ = self.attn(x2, x2, x2) # self attm所以都放自己: (Q, K, V) = (x2, x2, x2)
        # 4. Post‐Norm: 再下一次正規化
        y2 = self.ln2(y2)

        # 5. 還原回 (B, C, T, V)
        y = rearrange(y2, '(b t) v c -> b c t v', b=B, t=T, v=V)
        return y

# 2. Temporal
class TemporalSelfAttention(nn.Module):
    '''
    捕捉跨幀的注意力 => long-term dependency
    (B*V, T, D): 把節點V當作batch內部分組 => 在T時間軸上
    '''
    def __init__(self,
                 d_model : int,
                 num_heads: int,
                 max_len:int=200,
                 dropout : float = 0.1):
        super(TemporalSelfAttention, self).__init__()

        assert d_model % num_heads == 0

        # 1. muti_head attn: embed_dim 對應 d_model；batch_first=True 表示輸入為 (batch, seq_len, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=d_model,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          batch_first=True)
        
        # 2. 時間相對位置bias matrix (T × T):         
        self.rel_bias = nn.Parameter(torch.zeros(max_len, max_len))
        nn.init.trunc_normal_(self.rel_bias, std = 0.02)

        # 3. 前後 LayerNorm
        self.ln1 = nn.LayerNorm(d_model)  # 在 reshape → attn 前
        self.ln2 = nn.LayerNorm(d_model)  # 在 attn → 還原 後

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T, V)
        return: (B, C, T, V)
        """
        B, C, T, V = x.shape

        # permute+reshape → (B*V, T, C)
        x2 = rearrange(x, 'b c t v -> (b v) t c')

        # Pre-Norm
        x2 = self.ln1(x2)

        attn_bias = self.rel_bias[:T, :T]

        # Attention
        y2, _ = self.attn(x2, x2, x2, attn_mask = attn_bias)

        # Post-Norm
        y2 = self.ln2(y2)

        # 還原形狀 → (B, C, T, V)
        y = rearrange(y2, '(b v) t c -> b c t v', b=B, v=V)
        return y



# ============================================== GCN module ==============================================
# 1. 
class GrapgConv(nn.Module):
    """
    GCN: 輸入／輸出皆為 (B, C, T, V)
    """
    def __init__(self, in_channels: int, out_channels : int):
        super().__init__()
        # 1*1 捲積
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x: (B, C, T, V)
        y = self.conv(x)
        y = self.bn(y)
        return y

# 2. ST-GCN Block
class SimpleSTGCNBlock(nn.Module):
    """
    精簡版 ST‐GCN Block：
      1) 空間聚合：用 A_norm 做鄰接加權
      2) 1×1 空間卷積 + BN + ReLU
      3) 3×1 時域卷積 + BN
      4) 殘差連 + ReLU

    輸入 x: Tensor(B, C_in, T, V)
    輸出   : Tensor(B, C_out, T, V)
    """
    def __init__(self,
                 in_channels:  int,
                 out_channels: int,
                 A_norm:       torch.Tensor, # 骨架資訊
                 t_kernel:     int = 3):
        super().__init__()

        # 將歸一化骨架鄰接矩陣註冊為 buffer (不參與梯度更新)
        self.register_buffer('A_norm', A_norm)  # (V, V)

        # 空間 1×1 卷積
        self.conv_spatial = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            bias=False
        )
        # 時域 3×1 卷積，padding 保持時間長度不變
        pad = (t_kernel // 2, 0)
        self.conv_temporal = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(t_kernel, 1),
            padding=pad,
            bias=False
        )
        # 單一組 BN + ReLU
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 殘差分支：如果通道數或步距不匹配，就做 1×1 投影
        if in_channels != out_channels:
            self.residual = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, bias=False
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C_in, T, V)
        """
        # 1) 空間鄰接聚合：x_sp shape = (B, C_in, T, V)
        x_sp = torch.einsum('vw, bctw -> bctv', self.A_norm, x)

        # 2) 1×1 空間卷積
        x_sp = self.conv_spatial(x_sp)           # (B, C_out, T, V)

        # 3) 時域卷積
        x_tp = self.conv_temporal(x_sp)          # (B, C_out, T, V)

        # 4) 殘差 + BN + ReLU
        res = self.residual(x)                   # (B, C_out, T, V)
        out = self.bn(x_tp + res)
        return self.relu(out)                    # (B, C_out, T, V)

# 3. CNN only (for test)
class ConvOnlyBlock(nn.Module):
    """
    ConvOnlyBlock：用 1×1 Conv 取代 Graph Conv，仍保留 Temporal Conv + Residual
    介面保持與 STGCNBlock 相同
    x  : (B, C_in, T, V)
    out: (B, C_out, T, V)
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 A_norm=None,               # 參數保留，但此版本不用
                 t_kernel: int = 9,
                 ):
        super().__init__()

        t_stride = 1
        dropout = 0.5
        # 1×1 卷積 (不與鄰接矩陣相乘)
        self.conv_sp = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Temporal Conv 與原本一致
        padding = (t_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                      out_channels,
                      kernel_size=(t_kernel, 1),
                      stride=(t_stride, 1),
                      padding=(padding, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )

        # Residual branch
        if in_channels != out_channels or t_stride != 1:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(t_stride, 1)),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # step-1：純 1×1 Conv 做「空間」特徵轉換
        x_sp = self.conv_sp(x)              # ← 不再與 A_norm 相乘
        # step-2：Temporal Conv
        y = self.tcn(x_sp)
        # step-3：Residual + ReLU
        return F.relu(y + self.residual(x))
    

# ============================================== S-TR / T-TR (for ST-TRNet) ==============================================
class STRBlock(nn.Module):
    """
    S-TR 模組：SSA → TCN → Residual → ReLU
    """
    def __init__(self, 
                 d_model:int, 
                 num_head = 8, 
                 tcn_kernel = 3, # TCN 一次要看的幀數
                 dropout = 0.1):
        super().__init__()
        self.ssa = SpatialSelfAttention(d_model, num_head)

        # 時域卷積：kernel=(tcn_kernel,1)
        pad = (tcn_kernel//2, 0)
        self.tcn = nn.Conv2d(
            d_model,            # in 
            d_model,            # out
            kernel_size=(tcn_kernel, 1),   # tcn_kernel 個時間步 × 1 個節點
            padding=pad         # 時間維度兩側各補 tcn_kernel//2，節點維度不補
            )

        self.bn = nn.BatchNorm2d(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, V)
        y = self.ssa(x)
        y = self.tcn(y)
        y = self.bn(y)
        y = self.dropout(y)
        return F.relu(x + y) #殘差
    
class TTRBlock(nn.Module):
    """
    T-TR 模組：GCN → TSA → Residual → ReLU
    """
    def __init__(self,
                 in_channels:   int,
                 out_channels:  int, 
                 num_heads: int,
                 dropout:   float = 0.1):
        
        super().__init__()

        # 1. GCN
        self.gcn = GrapgConv(in_channels, out_channels)

        # 2. 時間注意力:
        self.tsa = TemporalSelfAttention(out_channels, 
                                         num_heads, 
                                         max_len=200, 
                                         dropout  = dropout)


        self.dropout = nn.Dropout(dropout)

        # 若 in_channels ≠ out_channels，需 proj；此處假設相同
        self.res_proj = None

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        y = self.gcn(x)
        y = self.tsa(y)
        y = self.dropout(y)
        res = x if self.res_proj is None else self.res_proj(x)
        return F.relu(y + res)
