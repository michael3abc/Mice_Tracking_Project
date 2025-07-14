
# from .utils import  Sliding_windows, Balance_windows, Encode_labels, impute_windows, compute_velocity_acc, smooth_short_events, impute_windows_with_center
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
    def __init__(self, input_dim=70, hidden_dim=64, num_layers=3, num_classes=7, se_ratio=16, dropout_p=0.2):
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

class PositionlEncoding(nn.Module): #transformer的時序資訊
    def __init__(self, d_model,  max_length = 500):
        super().__init__()
        pe = torch.zeros(max_length, d_model) # [max_len, d_model]
        pos = torch.arange(0, max_length, dtype = torch.float).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1,max_len,d_model]

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class BehaviorTransformer(nn.Module):
    def __init__(self, feature_dim=64, d_model=128, nhead=4,
                 num_layers=3, num_classes=8, dropout = 0.1):
        super().__init__()
        # 1. 原始特徵投射到 d_model 維
        self.input_proj = nn.Linear(feature_dim, d_model)
        # 2. 位置編碼
        self.pos_encoder = PositionlEncoding(d_model=d_model)
        # 3. 多層 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer = encoder_layer, num_layers=num_layers)        
        # 4. 全域池化 + classifier
        self.pool = nn.AdaptiveAvgPool1d(1)

        #   把 Transformer 編碼好的向量（長度 d_model）映射成 8 類的行為分類分數（logits）。
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, num_classes)
        )

    def forward(self, x):
        # x: [batch, seq_len, feature_dim]
        x = self.input_proj(x)                          # → [batch, seq_len, d_model]
        x = self.pos_encoder(x)                         # 加位置
        x = self.transformer_encoder(x)                 # → [batch, seq_len, d_model]
        x = x.transpose(1,2)                            # → [batch, d_model, seq_len]
        x = self.pool(x).squeeze(-1)                    # → [batch, d_model] 對每個樣本，針對每個通道（也就是 d_model），把 整個序列維（seq_len）平均壓縮成 1 值
        out = self.classifier(x)                        # → [batch, num_classes]
        return out


"""
可以加入的特徵:
1. Posture
2. Behavior Pattern
3. 空間距離特徵
    鼻子與尾巴距離
    前後腳距離
4. 方向單位向量: 2 (dx, dy)
5. 總速度 std	1	vel 全體變化度

"""