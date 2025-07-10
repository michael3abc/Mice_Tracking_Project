'''
模型概要:
Input: (batch_size, 16, 32)
 ↓
Conv1d(16→64, kernel_size=3, stride=1) → ReLU → BatchNorm1d → Dropout
 ↓
Conv1d(64→128, kernel_size=3, stride=1) → ReLU → BatchNorm1d → Dropout
 ↓
GlobalAvgPool1d (在時間維上平均)
 ↓
Flatten → Linear(128 → num_classes)
 ↓
Softmax (或用 CrossEntropyLoss)

'''
# from .data_processing import  Sliding_windows, Balance_windows, Encode_labels, impute_windows
# 在 One_D_CNN.py 最上面
from .utils import  Sliding_windows, Balance_windows, Encode_labels, impute_windows, compute_velocity_acc, smooth_short_events

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.utils import resample
import glob

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
    
def get_unique_path(path):
    """
    若檔案已存在，自動加入 _1, _2, ... 編號避免覆蓋
    """
    base, ext = os.path.splitext(path)
    counter = 1
    new_path = path
    while os.path.exists(new_path):
        new_path = f"{base}_{counter}{ext}"
        counter += 1
    return new_path

def train_model(model,
                train_loader,
                val_loader, 
                num_epochs = 10, 
                lr = 0.001, 
                device ="cuda" if torch.cuda.is_available() else 'cpu',
                save_folder = r'C:\Users\micha\Desktop\python_workspace\YOLO Mice Project\src\Pose_to_behavior\model\best_models'):

    raw_save_path = os.path.join(save_folder, f'best_epoch{num_epochs:03d}_{model.__class__.__name__}.pth')
    save_path = get_unique_path(raw_save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok = True)

    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',    #mode='min' →  val loss 越小越好
                                  factor=0.5,    #factor=0.5 → 如果 val loss 沒變好，就把 lr * 0.5
                                  patience=5,    #連續 5 個 epoch 沒改善才會觸發
                                  min_lr=1e-6,
                                  verbose = True #verbose=True → 打印出每次調整 lr 的訊息
                                  )
   


    best_loss = float('inf')   # 找「最小 loss」
    # best_acc  = 0.0            # 用 accuracy 
    for epoch in range(1,num_epochs+1):
        model.train()
        train_loss = 0.0
        train_correct = train_total = 0

        loop = tqdm(train_loader, desc=f"Epoch{epoch+1}/{num_epochs}", leave=False)
        for X_batch, y_batch in loop:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            #1. Forward
            outputs = model(X_batch) #shape:  (batch_size, num_classes)

            #2. loss
            loss = criterion(outputs, y_batch) 

            #3. Backpropagation
            optimizer.zero_grad() #把模型參數歸零 (才不會"累加梯度"學歪)
            loss.backward() #計算梯度
            optimizer.step() #更新權重 w=w−η⋅∇loss 

            #4. 統計
            train_loss += loss.item()*X_batch.size(0) #累加整個 epoch 的總 loss
            #loss.item()：將 tensor 轉為純數值（scalar）； X_batch.size(0)：這個 batch 的樣本數

            _, predicted = outputs.max(1)
            train_correct += (predicted == y_batch).sum().item()       
            train_total += y_batch.size(0)

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        #驗證階段
        model.eval()
        val_loss = 0.0
        val_correct = val_total = 0
        with torch.no_grad():
            for Xb, yb in tqdm(val_loader, desc=f"Val   Epoch {epoch}", leave=False):
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                loss = criterion(logits, yb)

                val_loss    += loss.item() * Xb.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total   += yb.size(0)

        avg_val_loss = val_loss / val_total
        val_acc      = val_correct / val_total
        
        scheduler.step(avg_val_loss)



        #存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)        

        print(f"Epoch {epoch}/{num_epochs}  "
              f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}  "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")

    return best_loss


def main():
    
    #0. 引入資料、清洗
    df = pd.read_csv(r"C:\Users\micha\Desktop\python_workspace\YOLO Mice Project\src\Pose_to_behavior\dataset\keypoints_with_behavior.csv")    
    df = df[(df["behavior"] != "unknown") & (df["behavior"] != "drink")]
    df = df.sort_values(by = ['video_id', 'frame']).reset_index(drop=True)

    smoothed_rows = []
    for vid, grp in df.groupby("video_id"):
        labels = grp["behavior"].tolist()
        labels_smoothed = smooth_short_events(labels, min_duration=60)
        grp = grp.copy()
        grp["behavior"] = labels_smoothed
        smoothed_rows.append(grp)
    df_clean = pd.concat(smoothed_rows, ignore_index=True)


    #1. 轉成Sliding_windows
    X, y = Sliding_windows(df_clean, window_size=64, stride=8)

    print("\n滑動窗口後檢查:")
    print(f"X形狀: {X.shape}, y形狀: {y.shape}")
    print(f"X有NaN: {np.isnan(X).any()}, y有NaN: {pd.isnull(y).any()}")

    #2. 平衡data數量
    X_bal, y_bal = Balance_windows(X, y, random_state=42)   
    X_bal = impute_windows(X_bal)
    
    
    
    
    #2.5. relative: 用原始影片長寬 相對位置 正規化
    orig_w, orig_h = 320, 240
    X_bal_norm = X_bal.copy().astype(np.float32)
    X_bal_norm[:, :, 0::2] /= orig_w #i.e. 索引 0 開始，每隔 2 個元素取一個 (偶數欄位是kpt_x)
    X_bal_norm[:, :, 1::2] /= orig_h #同理
    X_rel = X_bal_norm

    #2.6.用 min-MAx正規化[0,1]      
    X_min = X_bal.min(axis=(0,1), keepdims=True)  # shape (1,1,16)
    X_max = X_bal.max(axis=(0,1), keepdims=True)
    np.savez("minmax_values.npz", X_min=X_min, X_max=X_max)
    X_mM = (X_bal - X_min) / (X_max - X_min + 1e-6)


    #2.7 計算速度、加速度特徵:
    vel, acc = compute_velocity_acc(X_mM, delta = 5)

    #綁成64維輸入
    X_full = np.concatenate([X_rel, X_mM, vel, acc], axis=2)    
    X_final_input = X_full
    

    #3. 分train/val
    X_tr, X_va, y_tr, y_va = train_test_split(X_final_input, y_bal, test_size=0.2, stratify=y_bal, random_state=42)

    #4. encoding
    y_tr_encoded, le = Encode_labels(y_tr)
    y_va_encoded      = le.transform(y_va)  

    #5. 轉成tensor
    X_tr_tensor = torch.tensor(X_tr).permute(0,2,1).float()  #把輸入位置重排: ( N, 32, 16 ) → permute(0, 2, 1) → ( N, 16, 32 ) (每幀 16 維 × 32 幀 )
    X_va_tensor = torch.tensor(X_va).permute(0,2,1).float()   

    #class => PyTorch 的整數 tensor   
    y_tr_tensor = torch.tensor(y_tr_encoded).long()
    y_va_tensor = torch.tensor(y_va_encoded).long()
    
    #5. 包成dataloder
    train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    val_dataset = TensorDataset(X_va_tensor, y_va_tensor)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False )

    # 6. 開始訓練模型
    # model = Behavior1D_CNN(num_features=16, num_classes=len(le.classes_))
    model = BehaviorBiLSTM(input_dim = 64, hidden_dim = 64, num_layers = 3, num_classes = 7)

    best_loss = train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001)
    print(f" Training completed, best val loss = {best_loss:.4f}")
    

if __name__ == '__main__':
    main()

