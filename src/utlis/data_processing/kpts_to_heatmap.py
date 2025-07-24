import os
import pandas as pd
import h5py
import torch
from tqdm import tqdm

def kpts_to_heatmaps_torch(kpts, H, W, sigma, device):
    """
    输入：
      kpts: Tensor, shape (B, N, 2)
      H, W: heatmap 高度、宽度
      sigma: Gaussian 标准差
      device: 'cuda' or 'cpu'
    返回：
      heatmaps: Tensor, shape (B, N, H, W)
    """
    B, N, _ = kpts.shape

    # 生成坐标网格，shape 都先变成 (1,1,H,1)/(1,1,1,W)
    ys = torch.arange(H, device=device).view(1,1,H,1)
    xs = torch.arange(W, device=device).view(1,1,1,W)
    # 扩展到 (B,N,H,W)
    ys = ys.expand(B, N, H, W)
    xs = xs.expand(B, N, H, W)

    # 提取每个 keypoint 的 x/y，reshape 到 (B,N,1,1)
    x_i = kpts[:,:,0].view(B, N, 1, 1)
    y_i = kpts[:,:,1].view(B, N, 1, 1)

    # 计算平方距离，shape=(B,N,H,W)
    dist2 = (xs - x_i)**2 + (ys - y_i)**2

    # Gaussian 生成 heatmap
    hm = torch.exp(-dist2 / (2 * sigma * sigma))

    # 每个通道分别做归一化
    max_val = hm.amax(dim=(2,3), keepdim=True)  # shape (B,N,1,1)
    heatmaps = hm / (max_val + 1e-6)

    return heatmaps

def main():
    # 1. 读入 keypoints CSV
    csv_path = r"data_prediction\prediction_results\4. kpts_heatmaps\samples\sampled_kpts.csv"
    df = pd.read_csv(csv_path)
    T = len(df)

    # 2. heatmap & 原图尺寸
    H, W   =  40,  30   # 输出 heatmap 大小
    orig_W, orig_H = 320, 240  # 原始视频宽、高

    # 3. 准备 HDF5 存储
    save_folder = r"data_prediction\prediction_results\4. kpts_heatmaps"
    os.makedirs(save_folder, exist_ok=True)
    h5_path = os.path.join(save_folder, "mouse_heatmaps_40x30.h5")
    f = h5py.File(h5_path, 'w')
    dset = f.create_dataset(
        'heatmaps',
        shape=(T, 8, H, W),
        dtype='float16',
        chunks=(1,8,H,W),
        compression='gzip',
    )

    # 4. 设定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 5. 逐帧生成并写入
    for t, row in tqdm(df.iterrows(), total=T, desc="Generating heatmaps"):
        # 5.1 缩放并收集 keypoints
        kpts = []
        for i in range(8):
            # 原 CSV 列名 kpt{i}_x, kpt{i}_y
            x_orig = row[f'kpt{i}_x']
            y_orig = row[f'kpt{i}_y']
            # 缩放到 heatmap 尺寸
            x = x_orig * (W / orig_W)
            y = y_orig * (H / orig_H)
            kpts.append([x, y])
        # 转成 (1,8,2) Tensor
        kpts_tensor = torch.tensor([kpts], dtype=torch.float32, device=device)

        # 5.2 GPU 上生成 heatmaps -> (1,8,H,W)
        heatmaps = kpts_to_heatmaps_torch(kpts_tensor, H, W, sigma=1.5, device=device)

        # 5.3 拿回 CPU 并转换成 NumPy，shape -> (8,H,W)
        heat_np = heatmaps[0].cpu().numpy()

        # 5.4 写入 HDF5
        dset[t, ...] = heat_np

    # 6. 关闭文件
    f.close()
    print(f"Saved HDF5 dataset to {h5_path}")

if __name__ == '__main__':
    main()
