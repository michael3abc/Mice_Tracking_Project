import os
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

def add_bout_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    標記每支影片內的連續事件 (bout) 編號。
    輸入 df 需包含 ['video_id','frame','behavior']。
    回傳時會新增 ['bout_id']，相同 bout_id 代表同一段連續行為。
    """
    df = df.copy()
    df['bout_change'] = (df['behavior'] != df['behavior'].shift()).astype(int)
    df['bout_id'] = df.groupby('video_id')['bout_change'].cumsum()
    return df.drop(columns=['bout_change'])

def extract_segments(df: pd.DataFrame) -> pd.DataFrame:
    """
    將 frame-level 的行為標註/預測轉成 segments：
    對相同 (video_id, behavior, bout_id) 聚合，取最小/最大 frame。
    回傳 DataFrame 欄位 ['video_id','behavior','start','end']。
    """
    df = add_bout_ids(df)
    segs = df.groupby(['video_id','behavior','bout_id'])['frame'] \
             .agg(start='min', end='max') \
             .reset_index()[['video_id','behavior','start','end']]
    return segs

def iou(seg_a: pd.Series, seg_b: pd.Series) -> float:
    """
    計算兩個 segment 的 Intersection over Union：
    seg_a, seg_b 皆須擁有 'start','end' 欄位。
    """
    inter = max(0, min(seg_a.end, seg_b.end) - max(seg_a.start, seg_b.start) + 1)
    union = (seg_a.end - seg_a.start + 1) + (seg_b.end - seg_b.start + 1) - inter
    return inter / union if union > 0 else 0.0

def segment_level_f1(gt_segs: pd.DataFrame,
                     pred_segs: pd.DataFrame,
                     iou_thresh: float = 0.5) -> dict:
    """
    計算每個行為類別的 segment-level Precision/Recall/F1：
    - GT segment 與 Pred segment 同 video_id 同 behavior，IoU ≥ 門檻 視為 TP
    - Pred segment 若無任何 GT match → FP；GT segment 無任何 Pred match → FN
    回傳 dict: {behavior: (precision, recall, f1)}
    """
    behaviors = sorted(gt_segs['behavior'].unique())
    results = {}
    # 以 behavior 分組
    for beh in behaviors:
        gt_b = gt_segs[gt_segs['behavior'] == beh]
        pd_b = pred_segs[pred_segs['behavior'] == beh]
        matched_pred = set()
        TP = 0
        # 對每個 GT 找 match
        for _, gt in gt_b.iterrows():
            # 同 video_id 裡所有 Pred
            candidates = pd_b[pd_b['video_id'] == gt.video_id]
            found = False
            for pi, pdrow in candidates.iterrows():
                if iou(gt, pdrow) >= iou_thresh:
                    TP += 1
                    matched_pred.add(pi)
                    found = True
                    break
            # 若沒找到，自動當 FN
        FP = len(pd_b) - len(matched_pred)
        FN = len(gt_b) - TP
        prec = TP / (TP + FP) if TP + FP > 0 else 0.0
        rec  = TP / (TP + FN) if TP + FN > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        results[beh] = (prec, rec, f1)
    return results

def main(start_video=1, end_video=12,
         gt_folder=r"data_prediction/dataset/gt_behaviors",
         pred_folder=r"data_prediction\prediction_results\2_behavios\ST-TR",
         iou_thresh=0.5):
    # 1. 讀取並產生全部 GT & Pred 的 segments
    all_gt_segs   = []
    all_pred_segs = []
    for vid in range(start_video, end_video + 1):
        gt_path   = os.path.join(gt_folder, f"gt_behavior_mice{vid}.csv")
        pred_path = os.path.join(pred_folder, f"behaviors_mice{vid}.csv")
        df_gt   = pd.read_csv(gt_path)[['frame','behavior']].rename(columns={'behavior':'behavior'})
        df_pred = pd.read_csv(pred_path)[['frame','behavior']].rename(columns={'behavior':'behavior'})
        df_gt['video_id']   = vid
        df_pred['video_id'] = vid

        # 轉成 segments
        all_gt_segs.append(extract_segments(df_gt))
        all_pred_segs.append(extract_segments(df_pred))

    gt_segs   = pd.concat(all_gt_segs, ignore_index=True)
    pred_segs = pd.concat(all_pred_segs, ignore_index=True)

    # 2. 計算 segment-level F1
    seg_results = segment_level_f1(gt_segs, pred_segs, iou_thresh=iou_thresh)

    # 3. 輸出結果
    print(f"=== Segment-level F1 (IoU ≥ {iou_thresh}) ===")
    df_res = pd.DataFrame.from_dict(seg_results, orient='index',
                                    columns=['Precision','Recall','F1'])
    df_res['Precision'] = df_res['Precision'].map("{:.3f}".format)
    df_res['Recall']    = df_res['Recall'].map("{:.3f}".format)
    df_res['F1']        = df_res['F1'].map("{:.3f}".format)
    print(df_res)

if __name__ == "__main__":
    main()
