
from utils import  (Sliding_windows, Balance_windows, Encode_labels, impute_windows, compute_velocity_acc, smooth_short_events,
                    impute_windows_with_center, compute_cos_np,  compute_space_distances, compute_direction_unit, compute_speed_std,
                    )
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from Behaviors_Models import BehaviorBiLSTM_v3, BehaviorTransformer
from collections import Counter
from sklearn.utils import resample
import matplotlib.pyplot as plt



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
                save_folder = r'src\model\best_models'):

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
                                  # threshold=1e-4,
                                  verbose = True #verbose=True → 打印出每次調整 lr 的訊息
                                  )

    best_f1 = 0.0
    best_loss = float('inf')   # 找「最小 loss」
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
        all_pred, all_gt = [], []
        with torch.no_grad():
            for Xb, yb in tqdm(val_loader, desc=f"Val   Epoch {epoch}", leave=False):
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                loss = criterion(logits, yb)

                val_loss    += loss.item() * Xb.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == yb).sum().item()
                val_total   += yb.size(0)

                all_pred.append((preds.cpu()))
                all_gt.append(yb.cpu())

        #計算loss、ACC、F1、CM
        avg_val_loss = val_loss / val_total
        val_acc      = val_correct / val_total

        y_pred = torch.cat(all_pred)
        y_true = torch.cat(all_gt)
        val_f1 = f1_score(y_true, y_pred, average="macro")
        cm     = confusion_matrix(y_true, y_pred)
        
        scheduler.step(avg_val_loss)



        #存最佳模型
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            # 只在刷新最佳時印一次詳細報告
            print("\nNew best model!  Confusion matrix:")
            print(cm)
            print("\n", classification_report(
                y_true, y_pred, digits=3, target_names=val_loader.dataset.tensors[1].unique().numpy().astype(str)))

        # 若 label 名稱已有 `le.classes_`，可替換 target_names
        print(f"Epoch {epoch}/{num_epochs}  "
              f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f}  "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}, "
              f"Val F1={val_f1:.4f}")

    return best_f1, best_loss

def main(data_path, window_size, stride, orig_size, vel_delta ):
    
    #0. 引入資料、清洗
    df = pd.read_csv(data_path)
    df = df[(df["behavior"] != "unknown") & (df["behavior"] != 'drink')]
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
    X, y = Sliding_windows(df_clean, window_size=window_size, stride=stride)
    X_imp = impute_windows(X)
    y_imp = y.copy()

    # print("\n滑動窗口後檢查:")
    # print(f"X形狀: {X.shape}, y形狀: {y.shape}")
    # print(f"X有NaN: {np.isnan(X).any()}, y有NaN: {pd.isnull(y).any()}")

    #2. 平衡data數量 + 補點
    # X_bal, y_bal = Balance_windows(X, y, random_state=42)   


    #2. 平衡data數量: 做上下採樣
    target = np.sum(y_imp == 'rest')  # 比如 57224

    new_idxs = []
    for cls in np.unique(y_imp):
        idx = np.where(y_imp == cls)[0]
        cnt = len(idx)
        if cnt > target:
            # 多余的下采样（不放回）
            idx_res = resample(idx, replace=False, n_samples=target, random_state=42)
        elif cnt < target:
            # 不足的上采样（放回）
            idx_res = resample(idx, replace=True,  n_samples=target, random_state=42)
        else:
            idx_res = idx
        new_idxs.append(idx_res)

    all_idx = np.concatenate(new_idxs)
    np.random.shuffle(all_idx)

    # 2.3 应用到 X_imp, y_imp 得到平衡后的 X_bal, y_bal
    X_bal = X_imp[all_idx]
    y_bal = y_imp[all_idx]

    y_encoded, le = Encode_labels(y_bal)
    num_classes = len(le.classes_)  
    
    
    #2.5. relative: 用原始影片長寬 相對位置 正規化
    orig_w, orig_h = orig_size
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
    vel, acc = compute_velocity_acc(X_mM, delta = vel_delta)

    #2.8 計算body cos (角度)    
    N, seq_len, _ = X_rel.shape # 先把 X_rel (N, seq_len, 16) 重整成 (N, seq_len, 8, 2)
    kpt_seq = X_rel.reshape(N, seq_len, 8, 2)
    
    # 為每個 window 計算 cos 特徵
    cos_feats = np.zeros((N, seq_len, 1), dtype=np.float32)
    for i in range(N):
        cos_feats[i] = compute_cos_np(kpt_seq[i])  # (seq_len,1)

    # 現在把這 1 維的 cos_feats 一起 concat
    X_base = np.concatenate([X_rel, X_mM, vel, acc, cos_feats], axis=2)
    space_feats = compute_space_distances(X_rel)                  # (N,T,3)
    dir_feats   = compute_direction_unit(vel)         # (N,T,2)
    std_feats   = compute_speed_std(vel)                         # (N,T,1)

    X_full = np.concatenate([
    X_base,
    space_feats,
    dir_feats,
    std_feats,    
    ], axis=2)
    
    feature_dim = X_full.shape[2]

    X_tmp, X_test, y_tmp, y_test = train_test_split(
    X_full,            # 或 X_bal, 視你是對哪個做特徵拼接
    y_encoded,
    test_size=0.10,    # 10% 當 test
    stratify=y_bal,    # 用原始字串標籤做分層
    random_state=42
)


    #3. 分train/val

    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp,
        y_tmp,
        test_size=0.20,    # 20% of the 90% ⇒ 18% overall
        stratify=y_tmp,
        random_state=42
    )
    X_tr, X_va = X_train, X_val
    y_tr, y_va = y_train, y_val


    #5. 轉成tensor
    # LSTM 的輸入
    X_tr_tensor = torch.tensor(X_tr).permute(0,2,1).float()  #把輸入位置重排: ( N, 32, 16 ) → permute(0, 2, 1) → ( N, 16, 32 ) (每幀 16 維 × 32 幀 )
    X_va_tensor = torch.tensor(X_va).permute(0,2,1).float()   

    # Transformer 的輸入
    # X_tr_tensor = torch.tensor(X_tr).float()      # (N_tr, seq_len, feature_dim)
    # X_va_tensor = torch.tensor(X_va).float()      # (N_va, seq_len, feature_dim)
    
    y_tr_tensor = torch.tensor(y_tr).long()
    y_va_tensor = torch.tensor(y_va).long()

    #5. 包成dataloder
    #  dataset
    train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
    val_dataset   = TensorDataset(X_va_tensor, y_va_tensor)

    #  DataLoader
    train_loader = DataLoader(
        train_dataset,    # ← dataset
        batch_size=64,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,      # ← dataset
        batch_size=64,
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_test).permute(0,2,1).float(),
            torch.tensor(y_test).long()
        ),
        batch_size=64, shuffle=False
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 6. 開始訓練模型  
    num_classes = len(le.classes_) 
    model = BehaviorBiLSTM_v3(
        input_dim   = feature_dim,  # 例如 65 或者你实际的那串特征维度
        hidden_dim  = 64,           # 你可以继续用 64
        num_layers  = 3,            # 同之前
        num_classes = num_classes,
        se_ratio    = 16,           # squeeze‐excitation ratio
        dropout_p   = 0.2
    ).to(device)

    # model = BehaviorTransformer(
    #     feature_dim=feature_dim,
    #     d_model = 64,
    #     nhead=4,
    #     num_layers=1,
    #     num_classes=num_classes,
    #     dropout=0.1
    # ).to(device)

    best_f1, best_loss = train_model(model, train_loader, val_loader, num_epochs=100, lr=0.001)
    print(f" Training completed, best_d1 = {best_f1:.4f} ; best val loss = {best_loss:.4f} ; best f1 = {best_f1:.4f}")
    np.save("classes.npy", le.classes_)
    #8. 模型評估:
    model.eval()
    test_loss = 0.0
    test_correct = test_total = 0
    all_test_pred, all_test_gt = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            preds = logits.argmax(dim = 1)

            test_correct += (preds == yb).sum().item()
            test_total += yb.size(0)

            all_test_pred.append(preds.cpu())
            all_test_gt.append(yb.cpu())
        
        test_acc = test_correct / test_total
        y_test_pred = torch.cat(all_test_pred)
        y_test_gt   = torch.cat(all_test_gt)
        
        test_f1 = f1_score(y_test_gt, y_test_pred, average='macro')
        print(f"→ Test  Acc = {test_acc:.4f}, Test F1 = {test_f1:.4f}")

        cm = confusion_matrix(y_test_gt, y_test_pred)   # shape = (num_classes, num_classes)
        classes = le.classes_       
        # 轉成每一行的比率：cm_norm[i,j] = cm[i,j] / sum(cm[i,:])
        cm_norm = cm.astype(np.float32) / cm.sum(axis=1, keepdims=True)
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=np.arange(len(classes)),
            yticks=np.arange(len(classes)),
            xticklabels=classes,
            yticklabels=classes,
            ylabel='True label',
            xlabel='Predicted label',
            title='Confusion Matrix'
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

       
        thresh = cm_norm.max() / 2.
        for i in range(cm_norm.shape[0]):
            for j in range(cm_norm.shape[1]):
                ax.text(
                    j, i, format(cm_norm[i, j], 'd'),
                    ha='center', va='center',
                    color='white' if cm_norm[i, j] > thresh else 'black'
                )

        fig.tight_layout()
        plt.show()





if __name__ == '__main__':
    import yaml
    cfg_path = r"src\model\hyp_behavior.yaml"
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f) 
    
    data_path = cfg['data_path']
    window_size = cfg['window_size']
    stride = cfg['stride']
    orig_size = cfg['orig_size']
    vel_delta = cfg['vel_delta']

    main(data_path,window_size,stride,orig_size,vel_delta )