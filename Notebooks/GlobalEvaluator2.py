import numpy as np
import pandas as pd
import laspy

class GlobalEvaluator:
    def __init__(self):
        # Semantic counters (points)
        self.total_tp_sem = 0
        self.total_fp_sem = 0
        self.total_fn_sem = 0

        # Instance store (objects)
        self.all_gt_instances = []   # list of (num_points, max_iou_found)
        self.all_pred_instances = [] # list of (score, iou_with_gt, is_tp)
        self.total_gt_points_global = 0

    def accumulate_tile(self, tile_metrics):
        """Adds results from one tile to the global counters."""
        self.total_tp_sem += tile_metrics['tp_sem']
        self.total_fp_sem += tile_metrics['fp_sem']
        self.total_fn_sem += tile_metrics['fn_sem']

        self.all_gt_instances.extend(tile_metrics['gt_instances'])
        self.all_pred_instances.extend(tile_metrics['pred_instances'])
        self.total_gt_points_global += tile_metrics['gt_points_total']

    def report_micro_global(self):
        """Computes final metrics from all accumulated data."""
        # --- SEMANTIC METRICS ---
        iou_sem = self.total_tp_sem / (self.total_tp_sem + self.total_fp_sem + self.total_fn_sem + 1e-9)
        precision_sem = self.total_tp_sem / (self.total_tp_sem + self.total_fp_sem + 1e-9)
        recall_sem = self.total_tp_sem / (self.total_tp_sem + self.total_fn_sem + 1e-9)
        f1_sem = 2 * (precision_sem * recall_sem) / (precision_sem + recall_sem + 1e-9)

        # --- INSTANCE METRICS (mWCov and AP) ---
        mwcov = sum([(pts / self.total_gt_points_global) * iou for pts, iou in self.all_gt_instances])

        # AP50 (Average Precision at IoU >= 0.5)
        self.all_pred_instances.sort(key=lambda x: x[0], reverse=True)
        tp_list_50 = [1 if x[2] >= 0.5 else 0 for x in self.all_pred_instances]
        fp_list_50 = [1 if x[2] < 0.5 else 0 for x in self.all_pred_instances]
        acc_tp = np.cumsum(tp_list_50)
        acc_fp = np.cumsum(fp_list_50)
        recalls_ap = acc_tp / (len(self.all_gt_instances) + 1e-9)
        precisions_ap = acc_tp / (acc_tp + acc_fp + 1e-9)
        ap50 = np.trapezoid(precisions_ap, recalls_ap) if len(recalls_ap) > 0 else 0

        # AP25 (Average Precision at IoU >= 0.25)
        self.all_pred_instances.sort(key=lambda x: x[0], reverse=True)
        tp_list_25 = [1 if x[2] >= 0.25 else 0 for x in self.all_pred_instances]
        fp_list_25 = [1 if x[2] < 0.25 else 0 for x in self.all_pred_instances]
        acc_tp = np.cumsum(tp_list_25)
        acc_fp = np.cumsum(fp_list_25)
        recalls_ap = acc_tp / (len(self.all_gt_instances) + 1e-9)
        precisions_ap = acc_tp / (acc_tp + acc_fp + 1e-9)
        ap25 = np.trapezoid(precisions_ap, recalls_ap) if len(recalls_ap) > 0 else 0

        # --- INSTANCE-LEVEL PRECISION, RECALL, F1 (IoU > 0.5) ---
        # True Positive: prediction with IoU >= 0.5 against a GT instance
        tp_inst = sum(1 for x in self.all_pred_instances if x[2] >= 0.5)
        # False Positive: prediction that does NOT reach 0.5 IoU with any GT
        fp_inst = sum(1 for x in self.all_pred_instances if x[2] < 0.5)
        # False Negative: real tree (GT) that no prediction managed to match at IoU >= 0.5
        fn_inst = sum(1 for x in self.all_gt_instances if x[1] < 0.5)

        precision_inst = tp_inst / (tp_inst + fp_inst + 1e-9)
        recall_inst = tp_inst / (tp_inst + fn_inst + 1e-9)
        f1_inst = 2 * (precision_inst * recall_inst) / (precision_inst + recall_inst + 1e-9)

        # --- REPORT DATAFRAME ---
        rows = [
            {"Category": "Semantic", "Metric": "Semantic IoU",       "Value": iou_sem},
            {"Category": "Semantic", "Metric": "Semantic F1-Score",  "Value": f1_sem},
            {"Category": "Instance", "Metric": "Precision (Inst@50)","Value": precision_inst},
            {"Category": "Instance", "Metric": "Recall (Inst@50)",   "Value": recall_inst},
            {"Category": "Instance", "Metric": "F1-Score (Inst@50)", "Value": f1_inst},
            {"Category": "Instance", "Metric": "mWCov",              "Value": mwcov},
            {"Category": "Instance", "Metric": "AP50",               "Value": ap50},
            {"Category": "Instance", "Metric": "AP25",               "Value": ap25},
        ]

        report_df = pd.DataFrame(rows)
        report_df['Value (%)'] = (report_df['Value'] * 100).round(2).astype(str) + '%'
        return report_df


def compute_tile_metrics(pred_df, tree_las_gt, confidences=None):
    """
    Computes instance and point matches for a single tile.
    Assumes pred_df and tree_las_gt are spatially aligned (same point order).
    """
    points_las_gt = laspy.read(tree_las_gt)

    # Extract labels
    # GT: classification holds the instance ID (0 = non-tree)
    gt_labels = np.array(points_las_gt.classification)
    # Classes 1 and 2 are mapped to 0 (adjustment for test runs)
    gt_labels[gt_labels < 3] = 0

    # Pred: 'label' column of the DataFrame
    pred_labels = pred_df['label'].values

    # Semantic metrics (point-by-point)
    gt_bin = gt_labels > 0
    pred_bin = pred_labels >= 0

    tp_sem = np.sum(gt_bin & pred_bin)
    fp_sem = np.sum(~gt_bin & pred_bin)
    fn_sem = np.sum(gt_bin & ~pred_bin)

    # Instance metrics (matching)
    # Build a contingency table for ID overlaps
    df_match = pd.DataFrame({'gt': gt_labels, 'pred': pred_labels})
    df_match = df_match[(df_match['gt'] > 0) | (df_match['pred'] >= 0)]

    # Compute IoU for each (GT_id, Pred_id) pair that overlaps
    overlaps = df_match.groupby(['gt', 'pred']).size().reset_index(name='intersection')

    gt_counts   = df_match[df_match['gt']   > 0].groupby('gt').size()
    pred_counts = df_match[df_match['pred'] >= 0].groupby('pred').size()

    # IoU = Intersection / (Area_GT + Area_Pred - Intersection)
    overlaps['union'] = overlaps.apply(lambda x:
        (gt_counts.get(x['gt'], 0)     if x['gt']   > 0  else 0) +
        (pred_counts.get(x['pred'], 0) if x['pred'] >= 0 else 0) - x['intersection'], axis=1)
    overlaps['iou'] = overlaps['intersection'] / overlaps['union']

    overlaps = overlaps[(overlaps['gt'] > 0) & (overlaps['pred'] >= 0)]

    # mWCov: for each real GT, find its best IoU with any prediction
    gt_instances_results = []
    for gt_id, count in gt_counts.items():
        max_iou = overlaps[overlaps['gt'] == gt_id]['iou'].max() if gt_id in overlaps['gt'].values else 0
        gt_instances_results.append((count, max_iou))

    # AP50: for each prediction, find its max IoU with any real GT
    pred_instances_results = []
    for pred_id, count in pred_counts.items():
        # Confidence score (1.0 for watershed; actual score for DL models)
        score = confidences[pred_id] if confidences is not None else 1.0
        max_iou_with_gt = overlaps[overlaps['pred'] == pred_id]['iou'].max() if pred_id in overlaps['pred'].values else 0
        pred_instances_results.append((score, max_iou_with_gt, max_iou_with_gt))

    return {
        'tp_sem':         tp_sem,
        'fp_sem':         fp_sem,
        'fn_sem':         fn_sem,
        'gt_instances':   gt_instances_results,
        'pred_instances': pred_instances_results,
        'gt_points_total': np.sum(gt_counts),
    }
