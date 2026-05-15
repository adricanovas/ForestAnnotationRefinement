import numpy as np
import pandas as pd
import laspy

class GlobalEvaluator:
    def __init__(self):
        # Contadores Semánticos (Puntos)
        self.total_tp_sem = 0
        self.total_fp_sem = 0
        self.total_fn_sem = 0
        
        # Almacén para Instancias (Objetos)
        self.all_gt_instances = []   # Lista de (num_puntos, max_iou_encontrado)
        self.all_pred_instances = [] # Lista de (score, iou_con_gt, es_tp)
        self.total_gt_points_global = 0

    def acumular_tesela(self, metrics_tesela):
        """Suma los resultados de una tesela al conteo global."""
        self.total_tp_sem += metrics_tesela['tp_sem']
        self.total_fp_sem += metrics_tesela['fp_sem']
        self.total_fn_sem += metrics_tesela['fn_sem']
        
        self.all_gt_instances.extend(metrics_tesela['gt_instances'])
        self.all_pred_instances.extend(metrics_tesela['pred_instances'])
        self.total_gt_points_global += metrics_tesela['gt_points_total']

    def reportar_micro_global(self):
        """Calcula las métricas finales usando todos los datos acumulados."""
        # --- 1. MÉTRICAS SEMÁNTICAS ---
        iou_sem = self.total_tp_sem / (self.total_tp_sem + self.total_fp_sem + self.total_fn_sem + 1e-9)
        precision_sem = self.total_tp_sem / (self.total_tp_sem + self.total_fp_sem + 1e-9)
        recall_sem = self.total_tp_sem / (self.total_tp_sem + self.total_fn_sem + 1e-9)
        f1_sem = 2 * (precision_sem * recall_sem) / (precision_sem + recall_sem + 1e-9)

        # --- 2. MÉTRICAS DE INSTANCIA (mWCov y AP) ---
        mwcov = sum([(pts / self.total_gt_points_global) * iou for pts, iou in self.all_gt_instances])

        # AP50 (Average Precision)
        self.all_pred_instances.sort(key=lambda x: x[0], reverse=True)
        tp_list_50 = [1 if x[2] >= 0.5 else 0 for x in self.all_pred_instances]
        fp_list_50 = [1 if x[2] < 0.5 else 0 for x in self.all_pred_instances]
        acc_tp = np.cumsum(tp_list_50)
        acc_fp = np.cumsum(fp_list_50)
        recalls_ap = acc_tp / (len(self.all_gt_instances) + 1e-9)
        precisions_ap = acc_tp / (acc_tp + acc_fp + 1e-9)
        ap50 = np.trapezoid(precisions_ap, recalls_ap) if len(recalls_ap) > 0 else 0

        # AP25 (Average Precision)
        self.all_pred_instances.sort(key=lambda x: x[0], reverse=True)
        tp_list_25 = [1 if x[2] >= 0.25 else 0 for x in self.all_pred_instances]
        fp_list_25 = [1 if x[2] < 0.25 else 0 for x in self.all_pred_instances]
        acc_tp = np.cumsum(tp_list_25)
        acc_fp = np.cumsum(fp_list_25)
        recalls_ap = acc_tp / (len(self.all_gt_instances) + 1e-9)
        precisions_ap = acc_tp / (acc_tp + acc_fp + 1e-9)
        ap25 = np.trapezoid(precisions_ap, recalls_ap) if len(recalls_ap) > 0 else 0

        # --- 3. NUEVAS MÉTRICAS: PRECISION, RECALL, F1 A NIVEL DE INSTANCIA (IoU > 0.5) ---
        # Un True Positive de instancia es una predicción que tiene IoU >= 0.5 con un GT
        tp_inst = sum(1 for x in self.all_pred_instances if x[2] >= 0.5)
        # Un False Positive de instancia es una predicción que NO alcanza el 0.5 de IoU con ningún GT
        fp_inst = sum(1 for x in self.all_pred_instances if x[2] < 0.5)
        # Un False Negative de instancia es un árbol real (GT) que ninguna predicción logró "atrapar" con IoU >= 0.5
        fn_inst = sum(1 for x in self.all_gt_instances if x[1] < 0.5)

        precision_inst = tp_inst / (tp_inst + fp_inst + 1e-9)
        recall_inst = tp_inst / (tp_inst + fn_inst + 1e-9)
        f1_inst = 2 * (precision_inst * recall_inst) / (precision_inst + recall_inst + 1e-9)

        # --- 4. CONSTRUCCIÓN DEL DATAFRAME ---
        datos = [
            {"Categoría": "Semántica", "Métrica": "IoU Semántico", "Valor": iou_sem},
            {"Categoría": "Semántica", "Métrica": "F1-Score Semántico", "Valor": f1_sem},
            {"Categoría": "Instancia", "Métrica": "Precision (Inst@50)", "Valor": precision_inst},
            {"Categoría": "Instancia", "Métrica": "Recall (Inst@50)", "Valor": recall_inst},
            {"Categoría": "Instancia", "Métrica": "F1-Score (Inst@50)", "Valor": f1_inst},
            {"Categoría": "Instancia", "Métrica": "mWCov", "Valor": mwcov},
            {"Categoría": "Instancia", "Métrica": "AP50", "Valor": ap50},
            {"Categoría": "Instancia", "Métrica": "AP25", "Valor": ap25}
        ]

        df_reporte = pd.DataFrame(datos)
        df_reporte['Valor (%)'] = (df_reporte['Valor'] * 100).round(2).astype(str) + '%'
        
        return df_reporte

def calcular_metricas_tesela(pred_df, tree_las_gt, confidences=None):
    """
    Calcula los matches de instancias y puntos para una sola tesela.
    Asume que pred_df y tree_las_gt están alineados espacialmente (mismo orden de puntos).
    """
    points_las_gt = laspy.read(tree_las_gt)
    
    # 1. Extraer Etiquetas
    # GT: classification tiene el ID de instancia (0 es no-árbol)
    gt_labels = np.array(points_las_gt.classification)
    # Las clases 1 y 2 deben ser 0 (ajuste para pruebas)
    gt_labels[gt_labels < 3] = 0

    # Pred: columna 'label' del DF
    pred_labels = pred_df['label'].values

    # 2. Métricas Semánticas (Punto a Punto)
    gt_bin = gt_labels > 0
    pred_bin = pred_labels >= 0
    
    tp_sem = np.sum(gt_bin & pred_bin)
    fp_sem = np.sum(~gt_bin & pred_bin)
    fn_sem = np.sum(gt_bin & ~pred_bin)

    # 3. Métricas de Instancia (Matching)
    # Creamos una tabla de contingencia para ver solapamientos de IDs
    df_match = pd.DataFrame({'gt': gt_labels, 'pred': pred_labels})
    df_match = df_match[(df_match['gt'] > 0) | (df_match['pred'] >= 0)]
    
    # Calcular IoU para cada par de (GT_id, Pred_id) que se solapan
    overlaps = df_match.groupby(['gt', 'pred']).size().reset_index(name='intersection')
    
    gt_counts = df_match[df_match['gt'] > 0].groupby('gt').size()
    pred_counts = df_match[df_match['pred'] >= 0].groupby('pred').size()

    # Calcular IoU = Intersección / (Area_GT + Area_Pred - Intersección)
    overlaps['union'] = overlaps.apply(lambda x: 
        (gt_counts.get(x['gt'], 0) if x['gt'] > 0 else 0) + 
        (pred_counts.get(x['pred'], 0) if x['pred'] >= 0 else 0) - x['intersection'], axis=1)
    overlaps['iou'] = overlaps['intersection'] / overlaps['union']

    overlaps = overlaps[(overlaps['gt'] > 0) & (overlaps['pred'] >= 0)]

    # Para mWCov: Por cada GT real, buscamos su mejor IoU con alguna predicción
    gt_instances_results = []
    for gt_id, count in gt_counts.items():
        max_iou = overlaps[overlaps['gt'] == gt_id]['iou'].max() if gt_id in overlaps['gt'].values else 0
        gt_instances_results.append((count, max_iou))

    # Para AP50: Por cada Predicción, vemos su IoU máximo con algún GT real
    pred_instances_results = []
    for pred_id, count in pred_counts.items():
        # Score de confianza (usamos 1.0 para watershed y el real para los modelos de DL)
        score = confidences[pred_id] if confidences is not None else 1.0
        max_iou_con_gt = overlaps[overlaps['pred'] == pred_id]['iou'].max() if pred_id in overlaps['pred'].values else 0
        pred_instances_results.append((score, max_iou_con_gt, max_iou_con_gt))

    return {
        'tp_sem': tp_sem,
        'fp_sem': fp_sem,
        'fn_sem': fn_sem,
        'gt_instances': gt_instances_results,
        'pred_instances': pred_instances_results,
        'gt_points_total': np.sum(gt_counts)
    }