import yaml
import optuna
from train import main

def objective(trial):
    # 1) 建議要搜尋的超參數
    # 樣本數

    groom_cap = trial.suggest_int("sampling.max_win_per_behavior.groom",1000, 4000, 200)
    micromovement_cap = trial.suggest_int("sampling.max_win_per_behavior.micromovement",1000, 4000, 200)
    rest_cap    = trial.suggest_int("sampling.max_win_per_behavior.rest",1000, 4000, 200)
    walk_cap = trial.suggest_int("sampling.max_win_per_behavior.walk",1000, 4000, 200)
    default_cap = trial.suggest_int("sampling.max_win_per_behavior.default",1000, 4000, 500)

    # sliding_window
    window_size = trial.suggest_int("sliding_window.window_size", 16, 64)
    stride      = trial.suggest_int("sliding_window.stride", 4, 16)
    vel_delta   = trial.suggest_int("vel_delta", 1, 5)

    val_size_float    = trial.suggest_float("split.val_size", low = 0.1, high = 0.3, log = False)
    
    # -------------------------------------------------------------------
    # 學習率、warmup epochs
    lr             = trial.suggest_loguniform("train.lr", 1e-5, 1e-2)
    warmup_epochs  = trial.suggest_int("train.warmup_epochs", 0, 10)

    # class weight: 假設共有 7 個行為
    # 預設範圍 0.5–1.5，可依需要調整
    weight = [
        trial.suggest_uniform(f"train.weight[{i}]", 0.7, 1.3)
        for i in range(7)
    ]

    # class_improvement 各參數
    rest_threshold         = trial.suggest_uniform("class_improvement.rest_threshold",      0.5, 0.9)
    rest_margin_weight     = trial.suggest_uniform("class_improvement.rest_margin_weight",  0.0, 1.0)
    micro_threshold        = trial.suggest_uniform("class_improvement.micro_threshold",     0.5, 0.9)
    micro_margin_weight    = trial.suggest_uniform("class_improvement.micro_margin_weight", 0.0, 1.0)
    groom_threshold        = trial.suggest_uniform("class_improvement.groom_threshold",     0.5, 0.9)
    groom_margin_weight    = trial.suggest_uniform("class_improvement.groom_margin_weight", 0.0, 1.0)

    # model params
    # ST‐GCN channels
    stg1 = trial.suggest_categorical("model.params.stgcn_channels[0]", [32,64,128])
    stg2 = trial.suggest_categorical("model.params.stgcn_channels[1]", [32,64,128])

    # STTR dims & heads
    dim1 = stg2
    dim2 = stg2
    heads = trial.suggest_categorical("model.params.num_heads", [2, 4, 8])

    # TCN kernel
    kernel = trial.suggest_categorical("model.params.tcn_kernel", [3,5,7,9,11])

    # dropout
    do1 = trial.suggest_uniform("model.params.dropout", 0.0, 0.5)
    do2 = trial.suggest_uniform("model.params.dropout_p", 0.0, 0.5)

    # 層數
    ll = trial.suggest_int("model.params.lstm_layers", 1, 3)

    # LSTM hidden
    lh = trial.suggest_categorical("model.params.lstm_hidden", [64,128,256])

    # SE ratio
    se = trial.suggest_categorical("model.params.se_ratio", [4,8,16,32])

    # 2) 載入並修改 config
    cfg = yaml.safe_load(open("model/behavior_models/train_config.yaml", encoding="utf-8"))

    # 將 trial 建議的值填回去
    cfg["sampling"]["max_win_per_behavior"] = {
                                               "groom":   groom_cap,
                                               "micromovement": micromovement_cap,
                                               "rest":    rest_cap,
                                               "walk":  walk_cap,
                                               "default": default_cap
                                                }

    cfg["sliding_window"]["window_size"] = window_size
    cfg["sliding_window"]["stride"]      = stride
    cfg["vel_delta"]                     = vel_delta

    cfg["split"]["val_size"]       = val_size_float

    cfg["train"]["lr"]             = lr
    cfg["train"]["warmup_epochs"]  = warmup_epochs
    cfg["train"]["weight"]         = weight

    ci = cfg.setdefault("class_improvement", {})
    ci["rest_threshold"]       = rest_threshold
    ci["rest_margin_weight"]   = rest_margin_weight
    ci["micro_threshold"]      = micro_threshold
    ci["micro_margin_weight"]  = micro_margin_weight
    ci["groom_threshold"]      = groom_threshold
    ci["groom_margin_weight"]  = groom_margin_weight

    # 把模型參數也寫回 config
    mp = cfg.setdefault("model", {}).setdefault("params", {})
    mp["stgcn_channels"] = [stg1, stg2]
    mp["sttr_dims"]      = [dim1, dim2]
    mp["num_heads"]      = heads
    mp["tcn_kernel"]     = kernel
    mp["dropout"]        = do1
    # BiLSTM
    mp["lstm_hidden"]    = lh
    mp["lstm_layers"]    = ll
    mp["se_ratio"]       = se
    mp["dropout_p"]      = do2

    # 3) 執行一次訓練流程，main(cfg) 必須回傳 best_val_f1
    best_val_f1 = main(cfg)

    # 4) 回傳要 maximize 的指標
    return best_val_f1


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="mouse_behavior",                        # 為你的 Study 命名
        storage="sqlite:///optuna_mouse_v2.db",                # 指定 SQLite 檔案位置
        load_if_exists=True, 
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=2)
    )
    study.optimize(objective, n_trials=100, timeout=8*60*60)

    print("Best hyperparameters:", study.best_params)
    print("Best test f1:", study.best_value)