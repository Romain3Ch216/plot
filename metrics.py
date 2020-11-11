import numpy as np
import pandas as pd

def confusion_matrix_analysis(mat):
    """
    This method computes all the performance metrics from the confusion matrix. In addition to overall accuracy, the
    precision, recall, f-score and IoU for each class is computed.
    The class-wise metrics are averaged to provide overall indicators in two ways (MICRO and MACRO average)
    Args:
        mat (array): confusion matrix

    Returns:
        per_class (dict) : per class metrics
        overall (dict): overall metrics

    """
    TP = 0
    FP = 0
    FN = 0

    per_class = {}

    for j in range(mat.shape[0]):
        d = {}
        tp = np.sum(mat[j, j])
        fp = np.sum(mat[:, j]) - tp
        fn = np.sum(mat[j, :]) - tp

        d['IoU'] = round((100 * tp / (tp + fp + fn + 1e-20)), 1)
        d['Precision'] = round((100 * tp / (tp + fp + 1e-20)), 1)
        d['Recall'] = round((100 * tp / (tp + fn + 1e-20)), 1)
        d['F1-score'] = round((100 * 2 * tp / (2 * tp + fp + fn + 1e-20)), 1)

        per_class[str(j)] = d

        TP += tp
        FP += fp
        FN += fn

    overall = {}
    overall['micro_IoU'] = round((100 * TP / (TP + FP + FN)), 1)
    overall['micro_Precision'] = round((100 * TP / (TP + FP)), 1)
    overall['micro_Recall'] = round((100 * TP / (TP + FN)), 1)
    overall['micro_F1-score'] = round((100 * 2 * TP / (2 * TP + FP + FN)), 1)

    macro = pd.DataFrame(per_class).transpose().mean()
    overall['MACRO_IoU'] = round(macro.loc['IoU'], 1)
    overall['MACRO_Precision'] = round(macro.loc['Precision'], 1)
    overall['MACRO_Recall'] = round(macro.loc['Recall'], 1)
    overall['MACRO_F1-score'] = round(macro.loc['F1-score'], 1)

    overall['Accuracy'] = round((100 * np.sum(np.diag(mat)) / np.sum(mat)), 1)

    return per_class, overall
