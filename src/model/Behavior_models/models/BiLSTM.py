
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch


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
    def __init__(self,
                 input_dim=70,
                 hidden_dim : int = 64,
                 num_layers : int = 2,
                 num_classes : int =7,
                 se_ratio=16,
                 dropout_p=0.2):
        
        super().__init__()
        self.input_size = input_dim
        # 先ln
        self.ln_in  = nn.LayerNorm(input_dim)
        self.ln_out = nn.LayerNorm(hidden_dim*2)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,  # 堆兩層
                            batch_first=True,
                            bidirectional=True,  # 雙向 LSTM，會有 forward 和 backward，各 64 → concat 後變成 128 維
                            dropout = dropout_p if num_layers > 1 else 0.0
                            )
        chn = hidden_dim*2
        self.se_fc1 = nn.Linear(chn, chn//se_ratio, bias= False)
        self.se_fc2 = nn.Linear(chn // se_ratio, chn, bias=False)

        self.dropout = nn.Dropout(dropout_p)
        self.ln_post = nn.LayerNorm(chn * 2)  # 4H after mean+max

        self.hidden = nn.Linear(chn*2, hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch, channels=16, seq_len=32) → LSTM 要 (batch, seq_len, input_dim)
        # 必須是 (batch, seq_len, feature_dim)
        assert x.dim() == 3, \
            f"Expect 3-D tensor (batch, seq_len, feat), got {x.shape}"
        # 檢查最後一維長度
        assert x.size(-1) == self.input_size, \
            f"Feature dim mismatch: expect {self.input_size}, got {x.size(-1)}"
        
        # Pre-LN: 保證 LSTM 收到的輸入在每個 feature 維度上分布穩定
        x = self.ln_in(x)
        out, _ = self.lstm(x)  # out: (batch, 32, hidden_dim*2)
        out = self.ln_out(out)

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
