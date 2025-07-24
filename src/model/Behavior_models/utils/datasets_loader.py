# src/datasets.py
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from models.BiLSTM import BehaviorBiLSTM_v3
from models.ST_TR import SimpleSTTR, STTRNet
from models.Hybird import SimpleSTTR_BiLSTM, STTR_BiLSTM , SimpleSTTR_BiLSTM_v2, LSTM_Transformer
from models.ST_GCN import BehaviorSTGCN_BiLSTM, A_norm

# class: models' dataset
class LSTMDataset(torch.utils.data.Dataset):
    """把 (N, T, F) NumPy 轉成 (N, T, F) Tensor，僅做一次。"""
    def __init__(self, X_feat: np.ndarray, y_enc: np.ndarray):
        # 直接轉成 tensor，保持 batch_first
        self.X = torch.from_numpy(X_feat).float()          # (N, T, F)
        self.y = torch.from_numpy(y_enc).long()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class Simple_STTRDataset(Dataset):
    """
    將 (N, T, F) NumPy 特徵一次轉成 (N, F, T, 1) Tensor，減少 runtime 開銷。
    """
    def __init__(self, X_feat: np.ndarray, y_enc: np.ndarray):
        # (N, T, F) → (N, F, T, 1)
        self.X = torch.from_numpy(X_feat).float().permute(0, 2, 1).unsqueeze(-1)
        self.y = torch.from_numpy(y_enc).long()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class STTRDataset(Dataset):
    def __init__(self, X_feat: np.ndarray, y_encoded : np.ndarray, V: int = 8):
        """
        (B, T, F_total) → (B, T, V, C) → (B, C, T, V)
        B: batch_size
        T: window_size
        F: 特徵維度 (這裡只取norm_kpts，不取速度等其他的)
        V: num_nodes (keypoints)
        C: 一個keypoint有幾維度， (x,y) => C = 2
        """
        
        # 1. X_feat.shape = (B, T, 70) => 取前16 dim
        B, T, F, = X_feat.shape
        per_node = X_feat[:, :, :16]       # (N, T, 16)
        V = 8        
        C = per_node.shape[2]//V      # C = 2

        # 2. 先 reshape 成 (B, T, V=8, C=2)
        X = per_node.reshape(B, T, V, C)

        # 3. 再 permute 成 (B, C, T, V)，正好對應 model.forward(x: B,C,T,V)
        self.X = torch.from_numpy(X).float().permute(0,3,1,2)
        self.y = torch.from_numpy(y_encoded).long()

    
    def __len__(self):
        """回傳樣本 total size"""
        return self.y.shape[0]

    def __getitem__(self, idx):
        """
        回傳第 idx 筆樣本 (x, y)
        x: Tensor(C, T, V)
        y: Tensor()
        """
        return self.X[idx], self.y[idx]

class STTR_BiLSTMDataset(Dataset):
    def __init__(self, X_feat, y_enc):
        """
        X_feat: (N, T, F_total=70)
        y_enc : (N,)
        """
        # for LSTM 
        X_full = X_feat.copy()
        self.X_full = torch.from_numpy(X_full).float()

        # for STTR
        # 1. X_feat.shape = (B, T, 70) => 取前16 dim
        B, T, F, = X_feat.shape
        X_kpts = X_feat[:, :, :16]       # (N, T, 16)
        V = 8        
        C = X_kpts.shape[2]//V      # C = 2

        # 2. 先 reshape 成 (B, T, V=8, C=2)
        X_kpts = X_kpts.reshape(B, T, V, C)

        # 3. 再 permute 成 (B, C, T, V)，正好對應 model.forward(x: B,C,T,V)
        self.X_kpts = torch.from_numpy(X_kpts).float().permute(0,3,1,2)      
        self.y = torch.from_numpy(y_enc).long()

    
    def __len__(self):
        """回傳樣本 total size"""
        return self.y.shape[0]

    def __getitem__(self, idx):
        """
        回傳第 idx 筆樣本 (x, y)
        x: Tensor(C, T, V)
        y: Tensor()
        """
        return self.X_full[idx], self.y[idx]



# build lodars and model
def build_dataloaders(model_type, Xy_tr, Xy_val, Xy_te,
                      batch_size, num_workers, pin_memory):
    """依模型種類回傳 train/val/test DataLoader 三件組"""
    
    # === 選 Dataset 類型 ===
    if model_type.startswith("SimpleSTTR"):
        ds_class = Simple_STTRDataset
        dl_kwargs = dict(persistent_workers=True)

    elif model_type == "STTRNet" or  model_type == "SimpleSTTR_BiLSTM_v2" or model_type == "LSTM_Transformer":
        ds_class = STTRDataset
        dl_kwargs = dict(persistent_workers=True, prefetch_factor=4)
    elif model_type == "STTR_BiLSTM":
        ds_class = STTR_BiLSTMDataset
        dl_kwargs = dict(persistent_workers=True, prefetch_factor=4)

    else:                    
        ds_class = LSTMDataset
        dl_kwargs = dict(persistent_workers=True, prefetch_factor=4)

    # === 建三個 DataLoader ===
    def make(Xy, shuffle):
        ds = ds_class(*Xy)
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory,
            **dl_kwargs
        )
    return (make(Xy_tr, True),
            make(Xy_val, False),
            make(Xy_te,  False))




