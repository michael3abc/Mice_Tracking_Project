# src/train.py
import yaml, torch, numpy as np
import random, os, pickle, sys
from sklearn.model_selection import GroupShuffleSplit 

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, src_path)

from datasets_loader import build_dataloaders 
from trainer import train_and_save

from src.utils.model_factory import build_model
from src.utils.build_feature import build_features

from train_utils.data_preprocess import load_and_clean
from train_utils.sampler import sliding_windows, sample_windows_by_bout, assign_bout_ids, print_bouts_per_behavior, print_windows_per_behavior


"""
抽樣邏輯:
1. 把一段持續行為視為一個bout，將df加上bout_id欄位
2. 依bout_id + video_id切出不同sliding_windows (其中，過短: < window_size的省略； 中等長度: >window_size * min_pad_ratio 的線性插值)
3. 把video: [8, 9]當作test set (因兩部之分布~整體分布，且各行為都有涵蓋)
4. 每個behavior抽樣 'n_per_bt' 個bout，不到就全抽 
5. 以 bout_id 切分train/val，避免資料洩漏
"""


def main(cfg):
    # 0. 讀參數
    data_path   = cfg["data"]["path"]
    orig_size   = cfg["data"]["orig_size"]
    vel_delta   = cfg["vel_delta"]
    test_videos = cfg["sampling"]["test_videos"]           # e.g. [8,9]
    n_per_bt    = cfg["sampling"]["n_per_behavior"]        # e.g. 50
    dl_cfg      = cfg["dataloader"]
    model_name  = cfg["model"]["name"]
    model_params= cfg["model"]["params"]
    train_cfg   = cfg["train"]
    device      = train_cfg["device"]
    seed        = train_cfg["seed"]

    smooth_window_length = cfg["smooth"]["window_length"]
    polyorder = cfg["smooth"]["polyorder"]    

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)       


    # 1. 讀資料/抽樣
    df = load_and_clean(data_path)   
    df = assign_bout_ids(df)
    kpt_cols = [c for c in df.columns if c.startswith("kpt")]   # [kpt0_x,...kpt7_y]: 8*2 = 16
    box_cols = [b for b in df.columns if b.startswith("box")]   # ["box_x", "box_y", "box_w", "box_h"]

    X_raw, y_raw, vids_raw, bouts_raw = sliding_windows(
        df=df,
        window_size = cfg["sliding_window"]["window_size"],
        min_pad_ratio=cfg["sliding_window"]["min_pad_ratio"], 
        stride = cfg["sliding_window"]["stride"] , 
        kpt_cols = kpt_cols, 
        box_cols= box_cols,
        pad_mode=cfg["sliding_window"]["pad_mode"]
    )

    # 2. 先用影片切出 test
    test_mask     = np.isin(vids_raw, test_videos)
    test_idx      = np.where(test_mask)[0]
    trainval_idx  = np.where(~test_mask)[0]

    X_test = X_raw[test_idx]; y_test = y_raw[test_idx]
    bouts_test = bouts_raw[test_idx]

    X_tv    = X_raw[trainval_idx]
    y_tv    = y_raw[trainval_idx]
    bouts_tv= bouts_raw[trainval_idx]


    # 3.1. 用 sample_windows_by_bout 對 train+val 做抽樣」 
    X_trval, y_trval, bouts_trval = sample_windows_by_bout(
        X_tv, y_tv, bouts_tv,
        n_per_behavior = n_per_bt,
        max_win_per_behavior=cfg["sampling"]["max_win_per_behavior"],   # 限制每個行為最大windows數
        random_state   = seed
    )

    # 3.2. 從 train+val 用bouts 切出 val (避免同一個bout切出的windows同時落在train/avl)
    val_ratio = cfg["split"]["val_size"]     
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(
        gss.split(X_trval, y_trval, groups=bouts_trval)
    )

    X_train, y_train, bouts_train = X_trval[train_idx], y_trval[train_idx], bouts_trval[train_idx]
    X_val  , y_val, bouts_val   = X_trval[val_idx]  , y_trval[val_idx], bouts_trval[val_idx]



    # 4. Build feature_vector -> feature_vector (T = win_size, F = feat_dim)
    X_tr_feat, y_tr_enc, le, stats = build_features(X=X_train ,y= y_train,  orig_size = orig_size,  vel_delta = vel_delta, smooth_window_length=smooth_window_length, polyorder=polyorder )
    X_val_feat, y_val_enc, _, _    = build_features(X_val, y_val,      orig_size = orig_size,  vel_delta = vel_delta, smooth_window_length=smooth_window_length, polyorder=polyorder, le=le, stats=stats)
    X_te_feat, y_te_enc, _, _      = build_features(X_test, y_test,    orig_size = orig_size,  vel_delta = vel_delta, smooth_window_length=smooth_window_length, polyorder=polyorder, le=le, stats=stats)
    
    # 檢查各behavior的bout數量
    print_bouts_per_behavior("Test",  y_test,  bouts_test)
    print_windows_per_behavior("Test",  y_test)

    print_bouts_per_behavior("Train", y_train, bouts_train)
    print_windows_per_behavior("Train", y_train)

    print_bouts_per_behavior(" Val ", y_val,   bouts_val)
    print_windows_per_behavior("Val",   y_val)


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
    feat_dim = X_tr_feat.shape[2] 
    num_classes  = len(le.classes_)

    model = build_model(
        model_name   = model_name,
        feat_dim     = feat_dim,
        num_classes  = num_classes,
        model_params = model_params
    )

    # 6. 送進 train_and_save()
    best_epoch, best_loss, best_val_f1, test_f1, ckpt_path, exp_folder = train_and_save(
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
# ----- 訓練結束後 -----
    
    os.makedirs(exp_folder, exist_ok=True)
    # 1. save minmax資訊
    # minmax_pth = os.path.join(exp_folder,"minmax_stats.npz")    
    # np.savez(
    #     minmax_pth,
    #     X_min = stats["mins"],
    #     X_max = stats["maxs"]
    # )
    # print(f"✔ 已寫入 {minmax_pth}")

    # 2. save class對應的idx
    le_path = os.path.join(exp_folder,"label_encoder.pkl")
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    print(f"✔ 已寫入 {le_path}")

    return test_f1


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(open(r"model\behavior_models\train_config.yaml", encoding="utf-8"))
    main(cfg)
