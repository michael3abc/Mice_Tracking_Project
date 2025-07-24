# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import os
# import pandas as pd
# from tqdm import tqdm

# # -------------------------------------------------------------------
# # 參數設定
# # -------------------------------------------------------------------
# GT_CSV      = r"data_prediction\dataset\kpt_gt_behavior\kpts_with_gt_behavior_yolo11L.csv"
# META_CSV    = r"data_prediction\prediction_results\4. kpts_heatmaps\samples\sampled_metadata.csv"
# OUT_FOLDER  = r"data_prediction\prediction_results\4. kpts_heatmaps\samples"
# os.makedirs(OUT_FOLDER, exist_ok=True)
# OUT_CSV     = os.path.join(OUT_FOLDER, "sampled_kpts.csv")

# # -------------------------------------------------------------------
# # 1. 讀入原始標註 CSV 與 sampled metadata
# # -------------------------------------------------------------------
# gt_df   = pd.read_csv(GT_CSV)
# meta_df = pd.read_csv(META_CSV)

# # 將 seq_idxs 拆成 DataFrame：欄位 t0…t14
# seq_cols = [f"t{t}" for t in range(len(meta_df["seq_idxs"].iloc[0].split(",")))]
# seq_df   = meta_df["seq_idxs"].str.split(",", expand=True)
# seq_df.columns = seq_cols

# # 把 metadata 和拆好的 seq_df 合併
# meta_full = pd.concat([meta_df.drop(columns="seq_idxs"), seq_df], axis=1)

# # -------------------------------------------------------------------
# # 2. 針對每一段序列，依 time step 轉一列
# # -------------------------------------------------------------------
# records = []
# for _, row in tqdm(meta_full.iterrows(), total=len(meta_full), desc="Building sampled CSV"):
#     # row.behavior, row.video_id, row.center_idx, row.t0…t14
#     for t, col in enumerate(seq_cols):
#         gi = int(row[col])           # 全局 GT_CSV 索引
#         gt = gt_df.iloc[gi]          # 該幀所有 kpt 與行為
#         rec = {
#             "sample_id":  _,         # 序列編號 0…2999
#             "t":          t,         # 0…14
#             "behavior":   row["behavior"],
#             "video_id":   row["video_id"],
#             "frame":      int(gt["frame"])
#         }
#         # 把所有 kpt 欄位複製
#         for k in range(8):
#             rec[f"kpt{k}_x"] = gt[f"kpt{k}_x"]
#             rec[f"kpt{k}_y"] = gt[f"kpt{k}_y"]
#         records.append(rec)

# # -------------------------------------------------------------------
# # 3. 輸出合併後的 CSV，行數 ≈ num_samples×15
# # -------------------------------------------------------------------
# sampled_kpts_df = pd.DataFrame(records)
# # 按 sample_id, t 排序
# sampled_kpts_df = sampled_kpts_df.sort_values(["sample_id", "t"]).reset_index(drop=True)
# sampled_kpts_df.to_csv(OUT_CSV, index=False)
# print(f"■ 已輸出合併後 CSV → {OUT_CSV}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import torch
import h5py
from tqdm import tqdm


# GPU 上生成 heatmap 的函式（複用）
def kpts_to_heatmaps_torch(kpts, H, W, sigma, device):
    B, N, _ = kpts.shape
    ys = torch.arange(H, device=device).view(1,1,H,1).expand(B, N, H, W)
    xs = torch.arange(W, device=device).view(1,1,1,W).expand(B, N, H, W)
    x_i = kpts[:,:,0].view(B, N, 1, 1)
    y_i = kpts[:,:,1].view(B, N, 1, 1)
    dist2 = (xs - x_i)**2 + (ys - y_i)**2
    hm = torch.exp(-dist2 / (2 * sigma * sigma))
    return hm / (hm.amax(dim=(2,3), keepdim=True) + 1e-6)

def main():
    # -------------------------------------------------------------------
    # 參數
    # -------------------------------------------------------------------
    CSV_PATH    = r"data_prediction\prediction_results\4. kpts_heatmaps\samples\sampled_kpts.csv"
    OUT_FOLDER  = r"data_prediction\prediction_results\4. kpts_heatmaps\min_io"
    os.makedirs(OUT_FOLDER, exist_ok=True)
    H5_PATH     = os.path.join(OUT_FOLDER, "heatmaps_min_io.h5")

    H, W        = 40, 30           # heatmap 尺寸
    ORIG_W,ORIG_H= 320, 240        # 原圖尺寸
    SIGMA       = 1.5
    DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # -------------------------------------------------------------------
    # STEP 1. 讀 CSV 並準備座標陣列
    # -------------------------------------------------------------------
    df = pd.read_csv(CSV_PATH)
    num_samples = int(df["sample_id"].max()) + 1
    window_size = int(df["t"].max())       + 1
    N = 8  # 關鍵點數量

    # 把 (sample_id, t, kpt0_x…kpt7_y) 轉座標
    coords = np.empty((len(df), N, 2), dtype=np.float32)
    for i, row in enumerate(df.itertuples(index=False)):
        # row.sample_id, row.t, row.kpt0_x…
        for k in range(N):
            coords[i,k,0] = getattr(row, f"kpt{k}_x") * (W  / ORIG_W)
            coords[i,k,1] = getattr(row, f"kpt{k}_y") * (H  / ORIG_H)

    # 扁平索引對應：每一 row 對應一幀→ 總幀數
    num_frames = coords.shape[0]

    # -------------------------------------------------------------------
    # STEP 2. 建 HDF5 並一次性定義 dataset
    # -------------------------------------------------------------------
    # 建議 chunk 大小：100 段 sample／批次
    samples_per_batch = 100
    frames_per_batch  = samples_per_batch * window_size

    with h5py.File(H5_PATH, "w") as f:
        dset = f.create_dataset(
            "heatmaps",
            shape=(num_samples, window_size, N, H, W),
            dtype="float16",
            compression="gzip",
            chunks=(samples_per_batch, window_size, N, H, W)
        )

        # -------------------------------------------------------------------
        # STEP 3. 批次運算並最小化 I/O
        # -------------------------------------------------------------------
        for start in tqdm(range(0, num_frames, frames_per_batch),
                          desc="Batches", unit="batch"):
            end = min(num_frames, start + frames_per_batch)
            B = end - start
            # 要保證 B 可以被 window_size 整除
            assert B % window_size == 0, "請確保 frames_per_batch=%d 是 window_size 的倍數" % frames_per_batch

            # GPU 上批次算出 (B, N, H, W)
            kpts_tensor = torch.from_numpy(coords[start:end]).to(DEVICE)
            hm_tensor   = kpts_to_heatmaps_torch(kpts_tensor, H, W, SIGMA, DEVICE)
            hm_np       = hm_tensor.cpu().numpy().astype(np.float16)

            # 重塑為 (S, window_size, N, H, W)，S = B // window_size
            S = B // window_size
            block = hm_np.reshape(S, window_size, N, H, W)

            # 計算對應的 sample_id 範圍
            first_sample = start // window_size
            last_sample  = first_sample + S

            # 一次性寫入整個 block，IO 次數最小化
            dset[first_sample:last_sample, ...] = block

    print(f"■ 已輸出至 {H5_PATH}")

if __name__ == "__main__":
    main()

