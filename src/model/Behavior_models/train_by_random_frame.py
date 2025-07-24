# src/train.py
import yaml, torch, numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from utils.datasets_loader import build_dataloaders, build_model

from utils.data_preprocess import load_and_clean
from utils.sampler import sample_center_windows, sample_stratified_by_video
from utils.build_feature import build_features


from trainer import train_and_save

def main(cfg):
    # 0. 讀參數
    data_path         = cfg["data"]["path"]
    orig_size         = cfg["data"]["orig_size"]
    vel_delta         = cfg["vel_delta"]
    dl_cfg            = cfg["dataloader"]
    model_name        = cfg["model"]["name"]
    model_params      = cfg["model"]["params"]
    train_cfg         = cfg["train"]
    device            = train_cfg["device"]
    seed              = train_cfg["seed"]


    
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
        


    # 1. 獨資料/抽樣
    df = load_and_clean(data_path)    
    X_raw, y_raw, video_ids = sample_center_windows(
        df,
        center_per_class = cfg["sampling"]["num_per_class"],
        half_window      = cfg["sampling"]["half_window"],
        random_state     = seed
    )   
    # X_raw, y_raw, video_ids = sample_stratified_by_video(
    #     df,
    #     center_per_class = cfg["sampling"]["num_per_class"],
    #     half_window      = cfg["sampling"]["half_window"],
    #     random_state     = seed
    # )

    # 2.1. 先切出 test
    test_vids   = [8, 9]
    mask_test   = np.isin(video_ids, test_vids)     # True 表示該樣本屬於 test

    test_idx    = np.where(mask_test)[0]
    train_val_idx = np.where(~mask_test)[0]

    X_test , y_test  = X_raw[test_idx]  , y_raw[test_idx]
    X_trval, y_trval = X_raw[train_val_idx], y_raw[train_val_idx]


    # 2.2. 由 train+val 部分再 Stratified 切出 val
    val_ratio = cfg["split"]["val_size"]      # 例如 0.2
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=val_ratio, random_state=seed
    )
    train_idx, val_idx = next(sss.split(X_trval, y_trval))

    X_train, y_train = X_trval[train_idx], y_trval[train_idx]
    X_val  , y_val   = X_trval[val_idx]  , y_trval[val_idx]

    # 3. Build feature_vector
    X_tr_feat, y_tr_enc, le     = build_features(X_train, y_train,  orig_size = orig_size,  vel_delta = vel_delta)
    X_val_feat, y_val_enc, _    = build_features(X_val, y_val,      orig_size = orig_size,  vel_delta = vel_delta)
    X_te_feat, y_te_enc, _      = build_features(X_test, y_test,    orig_size = orig_size,  vel_delta = vel_delta)


    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 4. DataLoader（依模型類型自動選 Dataset）
    train_loader, val_loader, test_loader = build_dataloaders(
        model_type  = model_name,
        Xy_tr       = (X_tr_feat, y_tr_enc),
        Xy_val      = (X_val_feat, y_val_enc),
        Xy_te       = (X_te_feat, y_te_enc),
        batch_size  = dl_cfg["batch_size"],
        num_workers = dl_cfg["num_workers"],
        pin_memory  = dl_cfg["pin_memory"]
    )

    # 5. init Model
    # xb, yb = next(iter(train_loader))
    # xb shape = (B, C, T, V)
    # feat_dim = xb.shape[1]  # 這裡就會是 2   
    feat_dim = X_tr_feat.shape[2] 
    num_classes  = len(le.classes_)

    model = build_model(
        model_name   = model_name,
        feat_dim     = feat_dim,
        num_classes  = num_classes,
        model_params = model_params
    )

    # 6. 送進 train_and_save()
    best_epoch, best_loss, best_val_f1, test_f1, ckpt_path = train_and_save(
            model          = model,
            train_loader   = train_loader,
            val_loader     = val_loader,
            test_loader    = test_loader,
            class_names    = le.classes_,  
            num_epochs     = cfg["train"]["epochs"],
            cfg            = cfg,
            save_folder    = cfg["save"]["model_folder"],
            metrics_folder = cfg["save"]["metrics_folder"],
    )

    print(f"Best model: epoch={best_epoch}, val_f1={best_val_f1:.4f},test_f1={test_f1:.4f}, saved at {ckpt_path}")
 

if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open(r"src\model\Behavior_models\behavior_model_config.yaml", encoding="utf-8"))
    main(cfg)
