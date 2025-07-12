import re #正規表達
import pandas as pd
import os

def load_behaviors(n, txt_folder):
    """
    讀入行為標籤 txt，格式：
      frame: 00000001-00000019 behaviors
    傳回 list of (vid, start_frame, end_frame, label)
    """
    intervals = [] #放(i, start, end, label)
    for i in range(1, n+1):
        txt_path = os.path.join(txt_folder, f'mice{i}.txt')
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith('frame:'):
                    continue
                m = re.match(r'frame:\s*(\d+)-(\d+)\s+(\w+)', line)
                if m:
                    start = int(m.group(1))
                    end = int(m.group(2))
                    label = str(m.group(3))
                    intervals.append((i, start, end, label))
                else:
                    print(f"⚠️ 無法解析此行: {line}")
    return intervals



def load_all_keypoints(n, kpt_folder):
    """
    讀取 keypoints_1.csv ~ keypoints_n.csv，合併成一張大表
    並加上一個「video_id」欄位，區分video
    """

    df_list  = []
    for vidx in range(1,n+1):
        path = os.path.join(kpt_folder,fr'keypoints_{vidx}.csv')
        df = pd.read_csv(path)
        df['video_id'] = vidx
        df['frame'] = df['frame'].astype(int) #確保frame欄位是int
        df_list .append(df)
    return pd.concat(df_list , ignore_index= True) #重設索引，從 0 開始編到最後


def assign_behavior_to_kpt(df_kpt, intervals):
    """
    根據 intervals list，把對應的 behavior 塞到 df_kpt['behavior']
    """
    #預設unknow
    df_kpt['behavior'] = 'unknown'
    for (vid, start, end, label) in intervals:
        mask = (df_kpt['video_id'] == vid) & (df_kpt['frame'] >= start) & (df_kpt['frame']<= end)

        df_kpt.loc[mask, 'behavior'] = label #把True的index的"behavior"設label

    '''
    1. mask: 布林遮罩
        ex. df_kpt['frame'] >= 3
        
        0    False
        1    False
        2     True
        3     True
        4     True
    
    2. loc[rows, cols] 取哪幾row的哪個cols
        ex. rows = [1,3,4], cols = ['A', 'C', 'F'] 
    '''
    return df_kpt


if __name__ == '__main__':
    # 1. 讀行為標籤
    txt_folder = r"C:\Users\micha\Desktop\python_workspace\YOLO Mice Project\src\Pose_to_behavior\dataset\behaviors"

    behaviors = load_behaviors(5, txt_folder)

    # 2. 讀所有 keypoints
    kpt_folder = r"C:\Users\micha\Desktop\python_workspace\YOLO Mice Project\src\Pose_to_behavior\dataset"
    df_kpt = load_all_keypoints(5,kpt_folder )

    # 3. 合併行為
    df_labeled = assign_behavior_to_kpt(df_kpt, behaviors)

    # 4. 輸出
    out_csv = r'C:\Users\micha\Desktop\python_workspace\YOLO Mice Project\src\Pose_to_behavior\dataset\keypoints_with_behavior.csv'
    df_labeled.to_csv(out_csv, index=False)
    print(f"✅ 完成：已輸出 {out_csv}")

