# utils/optim_utils.py

import torch
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

def build_optim_and_schedulers(model: torch.nn.Module, cfg: dict):
    """
    根据 cfg 一次性构造 optimizer、warmup_scheduler、plateau_scheduler。
    返回 (optimizer, warmup_scheduler, plateau_scheduler) 三元组。
    """

    # 1) Optimizer
    opt_cfg   = cfg["optimizer"]
    train_cfg = cfg["train"]
    OptimizerClass = getattr(torch.optim, opt_cfg["type"])
    optimizer = OptimizerClass(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        betas=tuple(opt_cfg["betas"]),
        eps=float(opt_cfg["eps"]),
        weight_decay=float(opt_cfg["weight_decay"])
    )

    # 2) Warmup scheduler
    warmup_epochs = train_cfg.get("warmup_epochs", 0)
    def _warmup_lr_lambda(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=_warmup_lr_lambda)

    # 3) ReduceLROnPlateau scheduler
    sch_cfg = cfg["scheduler"]["plateau"]
    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode       = sch_cfg["mode"],
        factor     = float(sch_cfg["factor"]),
        patience   = int(sch_cfg["patience"]),
        threshold  = float(sch_cfg["threshold"]),
        min_lr     = float(sch_cfg["min_lr"])
    )

    return optimizer, warmup_scheduler, plateau_scheduler
