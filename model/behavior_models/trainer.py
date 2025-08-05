import copy, os, time, yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

from sklearn.metrics import (f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score)
from train_utils.polt_and_save_metrices import export_metrics, save_run_artifacts, plot_cm
from train_utils.optim_scheduler import build_optim_and_schedulers
from train_utils.data_preprocess import FallbackWrapper

def train_and_save( model,
                    train_loader,
                    val_loader, 
                    test_loader,
                    class_names,
                    num_epochs,
                    cfg,
                    save_folder = r'src\model\best_models',
                    metrics_folder = r"C:\Users\micha\Desktop\behaviors_model_metrics" ):
    """
    執行模型訓練並儲存最佳模型，同時匯出各項指標曲線。

    參數:
    - model: torch.nn.Module
    - train_loader, val_loader: DataLoader
    - num_epochs: int, 訓練輪數
    - lr: float, 學習率
    - device: str 或 torch.device
    - save_folder: str, 模型與圖表儲存路徑

    回傳:
    - best_epoch: int
    - best_f1: float
    - best_loss: float
    - best_path: str
    """


    # 1. 建 optimizer
    
    # base_model = model
    device = cfg["train"]["device"]
    # model = model.to(device)
    # model = FallbackWrapper(base_model, rest_idx=5, tau=cfg["train"]["rest_threshold"])
    model.to(device)
    optimizer, warmup_sched, plateau_sched = build_optim_and_schedulers(model, cfg)    


    scaler = GradScaler()  # 如果要用 AMP

    start_time = time.time()
    os.makedirs(save_folder, exist_ok=True)

    weights = torch.tensor(cfg["train"]["weight"], device=device)
    criterion = nn.CrossEntropyLoss(label_smoothing = cfg["train"]["label_smoothing"],
                                    weight=weights)


    train_losses, val_losses = [], []
    train_accs,  val_accs   = [], []
    train_f1s,   val_f1s    = [], []
    precisions,  recalls     = [], []
    aucs                     = []

    best_val_f1      = 0.0
    best_loss    = float("inf")
    best_epoch   = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_path    = None


    early_stopping_patience   = int(cfg["early_stopping"]["patience"])
    no_improve= 0


    rest_idx  = 5
    tau_rest  = cfg["class_improvement"]["rest_threshold"]          # rest 信心門檻 e.g. 0.6
    alpha_rest  = cfg["class_improvement"]["rest_margin_weight"]    # 額外loss權重

    micro_idx = 3
    tau_micro = cfg["class_improvement"]["micro_threshold"]         
    alpha_micro = cfg["class_improvement"]["micro_margin_weight"]

    groom_idx = 1
    tau_groom = cfg["class_improvement"]["groom_threshold"]
    alpha_groom = cfg["class_improvement"]["groom_margin_weight"]


    for epoch in range(1, num_epochs+1):

# ================== train ==================
        model.train()
        train_loss = train_correct = train_total = 0
        train_preds, train_gts = [], []
        
        loop = tqdm(train_loader, desc=f"Epoch{epoch}/{num_epochs}", leave=False)
        for X_batch, y_batch in loop:            
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad() #把模型參數歸零

            # AMP: 前向 + loss（混合精度）
            with autocast(device_type='cuda'):
                # 1) forward & CE loss
                logits = model(X_batch)                     # (B, C)
                # loss   = criterion(logits, y_batch)         # scalar
                ce_loss = criterion(logits, y_batch)

                probs   = F.softmax(logits, dim=1)              # (B, C) → 轉成每個類別的機率分布

                p_rest          = probs[:, rest_idx]                    # (B,) → 取出 rest_idx 這一維的機率值
                non_rest_maxk   = (y_batch != rest_idx).float()       # != rest 表示不是rest，val = 1
                hinge_nonrest   = F.relu(p_rest - tau_rest)* non_rest_maxk # 假設給非rest的信心過大 => (p_rest - tau) > 0 => relu 只留正的
                rest_mask       = (y_batch == rest_idx).float()      # (B,)：真實標籤是 rest 的樣本標記為 1
                hinge_rest      = F.relu(tau_rest - p_rest) * rest_mask   # 同理rest信心過低也罰
                margin_rest     = hinge_nonrest.mean() + hinge_rest.mean()               


                p_micro         = probs[:, micro_idx]                    # (B,) → 取出 rest_idx 這一維的機率值
                non_micro_mask  = (y_batch != micro_idx).float()       # != rest 表示不是rest，val = 1
                hinge_nonmicro  = F.relu(p_micro - tau_micro) * non_micro_mask
                micro_mask      = (y_batch == micro_idx).float()
                hinge_micro     = F.relu(tau_micro - p_micro) * micro_mask
                margin_micro    = hinge_nonmicro.mean() + hinge_micro.mean()

                p_groom         = probs[:, groom_idx]
                non_groom_mask  = (y_batch != groom_idx).float()
                hinge_nongroom  = F.relu(p_groom - tau_groom)* non_groom_mask
                groom_mask      = (y_batch == groom_idx).float()
                hinge_groom     = F.relu(tau_groom - p_groom)* groom_mask
                margin_groom    = hinge_nongroom.mean() + hinge_groom.mean()

                loss = ce_loss + alpha_rest* margin_rest + alpha_micro* margin_micro + alpha_groom* margin_groom
  





              

            scaler.scale(loss).backward()
            # scale step + update scaler
            scaler.step(optimizer)
            scaler.update()
            
            if epoch <= cfg["train"]["warmup_epochs"]:
                warmup_sched.step()


            #4. 統計
            train_loss += loss.item()*X_batch.size(0) #累加整個 epoch 的總 loss
            preds  = logits.argmax(dim = 1)
            train_correct += (preds  == y_batch).sum().item()       
            train_total += y_batch.size(0)

            train_preds.append(preds.cpu().numpy())
            train_gts.append(y_batch.cpu().numpy())

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        train_f1 = f1_score(np.concatenate(train_gts), np.concatenate(train_preds), average="macro")

        #──────────────────────────────val─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
        model.eval()

        val_loss = val_correct = val_total = 0
        all_preds, all_gts, all_probs = [], [], []
        with torch.no_grad():
            for Xb, yb in tqdm(val_loader, desc=f"Val   Epoch {epoch}", leave=False):
                Xb, yb = Xb.to(device), yb.to(device)
                logits = model(Xb)
                loss = criterion(logits, yb)

                val_loss    += loss.item() * Xb.size(0)
                preds = logits.argmax(dim=1)
                # preds = model(Xb)
                val_correct += (preds == yb).sum().item()
                val_total   += yb.size(0)

                all_preds.append(preds.cpu().numpy())
                all_gts.append(yb.cpu().numpy())
                all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())

        #計算loss、ACC、F1、CM
        avg_val_loss = val_loss / val_total
        val_acc      = val_correct / val_total        

        y_pred  = np.concatenate(all_preds)
        y_true  = np.concatenate(all_gts)
        y_score = np.vstack(all_probs)

        # 計算指標
        val_f1   = f1_score(y_true, y_pred, average="macro")
        val_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        val_rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
        val_auc  = roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")

        plateau_sched.step(avg_val_loss)


        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)   
        val_f1s.append(val_f1)
        precisions.append(val_prec)
        recalls.append(val_rec)
        aucs.append(val_auc)

        # 存最佳模型
        if  val_f1 > best_val_f1:
            best_val_f1   = val_f1
            best_loss = avg_val_loss
            best_epoch = epoch   # ← 把當前 epoch 記錄下來
            no_improve = 0
            # 重新組檔名，包含模型、epoch、val F1
            fname = f"{model.__class__.__name__}_epoch{epoch:02d}_valF1{val_f1:.4f}.pth"
            best_path = os.path.join(save_folder, fname)
            best_model_wts = copy.deepcopy(model.state_dict())



        else:
            no_improve += 1
            if no_improve >= early_stopping_patience:
                print(f"⚠ Early-Stopping triggered at epoch {epoch}；best_epoch = {best_epoch}")
                break







        


        print(
            f"Epoch {epoch}/{num_epochs}  "
            f"Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f},  "
            f"Val Acc={val_acc:.4f}, F1={val_f1:.4f},  "
            f"Prec={val_prec:.4f}, Rec={val_rec:.4f}, AUC={val_auc:.4f}"
        )

    # ── Test loop ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    model.eval()

    test_correct = test_total = 0
    all_pred, all_gt = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds  = logits.argmax(dim=1)
            # preds  = model(xb)
            test_correct += (preds == yb).sum().item()
            test_total   += yb.size(0)
            all_pred.append(preds.cpu().numpy())
            all_gt.append(yb.cpu().numpy())

    test_acc = test_correct / test_total
    y_test_pred = np.concatenate(all_pred)
    y_test_gt   = np.concatenate(all_gt)
    test_f1 = f1_score(y_test_gt, y_test_pred, average='macro')
    cm      = confusion_matrix(y_test_gt, y_test_pred)    

    # 最終模型檔名 (不含 .pth)
    model_name = f"{model.__class__.__name__}_best_epoch{best_epoch:02d}_test_F1_{test_f1:.4f}"
    exp_folder = os.path.join(metrics_folder, model_name)
    os.makedirs(exp_folder, exist_ok=True)

    # 儲存模型權重
    model_path = os.path.join(exp_folder, "best_model.pth")
    best_path = model_path  
    torch.save(best_model_wts, model_path)

    print(f"\nSaved best model to: {model_path}")

    # ── 存數值檔 ────────────────────────
    epochs = list(range(1, len(train_losses)+1))

    save_run_artifacts(
        exp_folder    = exp_folder,
        class_names   = class_names,
        model         = model,
        epochs        = epochs,
        train_losses  = train_losses, val_losses = val_losses,
        train_accs    = train_accs,   val_accs   = val_accs,
        train_f1s     = train_f1s,    val_f1s    = val_f1s,
        precisions    = precisions,   recalls    = recalls,
        aucs          = aucs,
        y_test_gt     = y_test_gt,
        y_test_pred   = y_test_pred,
        cm            = cm,
        best_epoch    = best_epoch,
        best_val_f1   = best_val_f1,
        test_acc      = test_acc,
        test_f1       = test_f1,
        scheduler_state = plateau_sched.state_dict(),
        start_time    = start_time
    )
    
    export_metrics(
        epochs,
        train_losses, val_losses,
        train_accs,  val_accs,
        train_f1s,   val_f1s, 
        precisions,  recalls,
        aucs,
        exp_folder
    )    

    plot_cm(exp_folder=exp_folder,
            class_names=class_names,
            cm=cm)


    with open(os.path.join(exp_folder, "run_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    return best_epoch, best_loss, best_val_f1, test_f1, best_path, exp_folder
