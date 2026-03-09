import numpy as np
import matplotlib.pyplot as plt
import os, time, yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report

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

# utils/metrics_utils.py

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import numpy as np


def export_metrics(epochs,
                   train_losses, val_losses,
                   train_accs, val_accs,
                   train_f1s, val_f1s,
                   precisions, recalls,
                   aucs,
                   save_folder):
    """
    匯出並儲存訓練過程中的各項指標曲線。
    
    參數:
    - epochs: list of int, epoch 編號 (從 1 開始)
    - train_losses, val_losses: list of float, 訓練/驗證 loss
    - train_accs, val_accs:       list of float, 訓練/驗證 accuracy
    - precisions, recalls:        list of float, precision & recall (macro)
    - aucs:                       list of float, AUC-ROC (macro)
    - save_folder: str, 圖片儲存資料夾
    """

    os.makedirs(save_folder, exist_ok=True)
    # 1. Loss 
    plt.figure()
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses,   label="Val   Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "loss_curve.png"))
    plt.close()

    # 2. Accuracy 
    plt.figure()
    plt.plot(epochs, train_accs, label="Train Acc")
    plt.plot(epochs, val_accs,   label="Val   Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "accuracy_curve.png"))
    plt.close()

    # 3. Precision & Recall 曲線
    plt.figure()
    plt.plot(epochs, precisions, label="Precision")
    plt.plot(epochs, recalls,    label="Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "precision_recall_curve.png"))
    plt.close()

    # 3. F1 曲線  ←新增
    plt.figure()
    plt.plot(epochs, train_f1s, label="Train F1")
    plt.plot(epochs, val_f1s,   label="Val   F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "f1_curve.png"))
    plt.close()

    # 4. AUC-ROC 曲線
    plt.figure()
    plt.plot(epochs, aucs, label="AUC-ROC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "auc_roc_curve.png"))
    plt.close()

def save_run_artifacts(
    *,
    exp_folder: str,
    model,                         # torch model，方便統計參數量
    epochs: list[int],
    train_losses: list[float], val_losses: list[float],
    train_accs: list[float],  val_accs: list[float],
    train_f1s: list[float],   val_f1s: list[float],
    precisions: list[float],  recalls: list[float],
    aucs: list[float],
    y_test_gt: np.ndarray,    y_test_pred: np.ndarray,
    cm: np.ndarray,                               # 混淆矩陣 (num_cls×num_cls)
    best_epoch: int,
    best_val_f1: float,
    test_acc: float,
    test_f1: float,
    scheduler_state: dict,
    start_time: float,
    class_names: list[str]
):
    """
    將一次實驗的所有「可程式化」成果匯出到 exp_folder
    --------------------------------------------------------------------
    - metrics_epoch.csv              ：逐 epoch 指標
    - classification_report.csv      ：Test 各類別 P/R/F1
    - confusion_matrix.npy           ：原始 CM 數值
    - best_summary.yaml              ：訓練摘要（最終指標、參數量…）
    --------------------------------------------------------------------
    """

    os.makedirs(exp_folder, exist_ok=True)

    # 1. 逐 epoch 指標 -------------------------------------------------
    df_ep = pd.DataFrame({
        "epoch": epochs,
        "loss_train": train_losses,
        "loss_val":   val_losses,
        "acc_train":  train_accs,
        "acc_val":    val_accs,
        "f1_train":   train_f1s,
        "f1_val":     val_f1s,
        "precision":  precisions,
        "recall":     recalls,
        "auc":        aucs,
    })
    df_ep.to_csv(Path(exp_folder, "metrics_epoch.csv"), index=False)

    # 2. classification report ----------------------------------------
    report = classification_report(
        y_test_gt, y_test_pred,target_names=class_names, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).T.to_csv(
        Path(exp_folder, "classification_report.csv"))

    # 3. confusion matrix (數值) --------------------------------------
    np.save(Path(exp_folder, "confusion_matrix.npy"), cm)

    # 4. 總結 YAML -----------------------------------------------------
    summary = dict(
        best_epoch     = int(best_epoch),
        best_val_f1    = float(best_val_f1),
        test_acc       = float(test_acc),
        test_f1        = float(test_f1),
        param_cnt      = int(sum(p.numel() for p in model.parameters())),
        train_time_sec = float(time.time() - start_time),
        lr_scheduler   = scheduler_state,
    )
    yaml.safe_dump(summary,
                   open(Path(exp_folder, "best_summary.yaml"), "w"),
                   allow_unicode=True)

    print(f"[✓] 所有數值檔案已儲存 → {exp_folder}")

def plot_cm(exp_folder, class_names, cm ):
    os.makedirs(exp_folder, exist_ok=True)
    cm_path = os.path.join(exp_folder, "test_confusion_matrix.png")
    classes = class_names    
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
                j, i, 
                f"{cm_norm[i, j]:.2f}",
                ha='center', va='center',
                color='white' if cm_norm[i, j] > thresh else 'black'
            )

    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    print(f"✓ Test CM saved → {cm_path}")
    # plt.show()
    plt.close(fig)     # 釋放記憶體