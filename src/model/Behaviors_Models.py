
from .utils import  Sliding_windows, Balance_windows, Encode_labels, impute_windows, compute_velocity_acc, smooth_short_events, impute_windows_with_center
import torch
import torch.nn as nn
import torch.nn.functional as F


class Behavior1D_CNN(nn.Module):
    def __init__(self, num_features = 16, num_classes = 7):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels= 64, kernel_size= 3 , padding= 1) #輸入16維；輸出:64組時間模式；ker:時間上每三幀的特徵 ；輸出保持原來
        self.bn1 = nn.BatchNorm1d(64) #輸出做標準化（均值 0，變異數 1）
        self.dropout1 = nn.Dropout(0.2) #訓練時隨機讓 20% 的神經元輸出為 0，防止 overfitting。

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)

        self.global_pool = nn.AdaptiveAvgPool1d(1) #把32幀壓縮(N, 128, 32) → (N, 128, 1)

        
        self.fc = nn.Linear(128, num_classes) #全連接層(Dense)層，把 128 維的 representation 轉換成 7 個類別的預測（logits）
    
    def forward(self, x):
        # 第一層卷積 + BN + ReLU + Dropout
        x = F.relu(self.bn1(self.conv1(x))) # shape: (N, 64, 32)
        x = self.dropout1(x)

            # 第二層卷積 + BN + ReLU + Dropout
        x = F.relu(self.bn2(self.conv2(x)))  # shape: (N, 128, 32)
        x = self.dropout2(x)

        x = self.global_pool(x) #壓縮整個時間序列
        x = x.view(x.size(0), -1) #.view() 是 reshape: (N, 128, 1) 壓成 (N, 128)
        out = self.fc(x) #128 -> 7 class
        return out

class BehaviorBiLSTM(nn.Module):
    def __init__(self, input_dim = 64, hidden_dim = 64, num_layers = 3, num_classes = 7):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers, #堆兩層
                            batch_first=True,
                            bidirectional=True #雙向 LSTM，會有 forward 和 backward，各 64 → concat 後變成 128 維
                            )
        
        self.hidden = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch, channels=16, seq_len=32) → LSTM 要 (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)                  # → (batch, 32, 16)
        out, _ = self.lstm(x)                  # out: (batch, 32, hidden_dim*2)
        last = out[:, -1, :]                   # 拿最後一個 time step
        h = F.relu(self.hidden(last))
        logits = self.fc(h)                 # → (batch, num_classes)
        return logits

class BehaviorBiLSTM_v2(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, num_layers=3, num_classes=7, dropout_p=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,  # 堆兩層
                            batch_first=True,
                            bidirectional=True,  # 雙向 LSTM，會有 forward 和 backward，各 64 → concat 後變成 128 維
                            dropout = dropout_p
                            )
        self.ln_post = nn.LayerNorm(hidden_dim * 2)  # 雙向 → 2H
        self.dropout = nn.Dropout(dropout_p)

        self.hidden = nn.Linear(hidden_dim * 4, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch, channels=16, seq_len=32) → LSTM 要 (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # → (batch, 32, 16)
        out, _ = self.lstm(x)  # out: (batch, 32, hidden_dim*2)

        mean_pool = out.mean(dim = 1)
        max_pool = out.max(dim = 1).values
        pooled = torch.cat([mean_pool, max_pool], dim = 1)
        h = F.relu(self.hidden(pooled))

        # last = out[:, -1, :]  # 仍取最後一步做比較
        # h = self.ln_post(last)  # ← LayerNorm
        # h = self.dropout(h)  # ← Dropout
        # h = F.relu(self.hidden(h))

        logits = self.fc(h)  # → (batch, num_classes)
        return logits

class BehaviorBiLSTM_v3(nn.Module):
    #Bi-LSTM → (B,T,2H) ──► SE (通道重標定) ──► 池化 (mean+max) ──► FC
    def __init__(self, input_dim=64, hidden_dim=64, num_layers=3, num_classes=7, se_ratio=16, dropout_p=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,  # 堆兩層
                            batch_first=True,
                            bidirectional=True,  # 雙向 LSTM，會有 forward 和 backward，各 64 → concat 後變成 128 維
                            dropout = dropout_p
                            )
        chn = hidden_dim*2
        self.se_fc1 = nn.Linear(chn, chn//se_ratio, bias= False)
        self.se_fc2 = nn.Linear(chn // se_ratio, chn, bias=False)

        self.ln_post = nn.LayerNorm(hidden_dim * 4)  # 雙向 → 4H
        self.dropout = nn.Dropout(dropout_p)

        self.hidden = nn.Linear(chn*2, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch, channels=16, seq_len=32) → LSTM 要 (batch, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # → (batch, 32, 16)
        out, _ = self.lstm(x)  # out: (batch, 32, hidden_dim*2)

        # ---------- ★ Channel SE ----------
        se = out.mean(dim=1)                      # (B,2H)  Global Avg over time
        se = F.relu(self.se_fc1(se))              # (B,2H/r)
        se = torch.sigmoid(self.se_fc2(se))       # (B,2H) in (0,1)
        out = out * se.unsqueeze(1)               # (B,T,2H)  通道重標定

        # ---------- Mean + Max 時序池化 -----
        mean_pool = out.mean(dim=1)
        max_pool  = out.max(dim=1).values
        pooled    = torch.cat([mean_pool, max_pool], dim=1)  # (B,4H)

        # ---------- FC head ---------------
        pooled = self.ln_post(pooled)
        pooled = self.dropout(pooled)
        h = F.relu(self.hidden(pooled))
        logits = self.fc(h)
        return logits

class BehaviorTransformer(nn.Module):
    def __init__(self, in_dim=64, d_model=128, nhead=4,
                 num_layers=2, num_classes=7, seq_len=64):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_embed  = nn.Parameter(torch.randn(seq_len, d_model))   # learned PE
        encoder_layer   = nn.TransformerEncoderLayer(d_model, nhead,
                                                     dim_feedforward=256,
                                                     dropout=0.1,
                                                     batch_first=True)
        self.encoder    = nn.TransformerEncoder(encoder_layer, num_layers)
        self.cls_head   = nn.Linear(d_model, num_classes)

    def forward(self, x):  # x: (B, C=64, T=64)
        x = x.permute(0, 2, 1)                     # (B,T,C)
        x = self.input_proj(x) + self.pos_embed    # (B,T,d_model)
        z = self.encoder(x)                        # (B,T,d_model)
        z = z.mean(dim=1)                          # global mean pool
        return self.cls_head(z)


