import numpy as np
from sklearn import metrics
import json
import os


def ACC(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.accuracy_score(y_true, y_pred)

def Cohen_Kappa(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.cohen_kappa_score(y_true, y_pred)

def F1_score(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.f1_score(y_true, y_pred, average='macro')

def Recall(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.recall_score(y_true, y_pred, average='macro')

def Precision(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.precision_score(y_true, y_pred, average='macro')

def cls_report(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.classification_report(y_true, y_pred, digits=4)


def confusion_matrix(output, target):
    y_pred = output.argmax(1)
    y_true = target.flatten()
    y_pred = y_pred.flatten()
    return metrics.confusion_matrix(y_true, y_pred)

def write_score2json(score_info, val_anno_file, results_dir):
    score_info = score_info.astype(np.float)
    score_list = []
    anno_info = np.loadtxt(val_anno_file, dtype=np.str_)
    for idx, item in enumerate(anno_info):
        id = item[0].rsplit('/', 1)[-1]
        label = int(item[1])
        score = list(score_info[idx])
        pred = score.index(max(score))
        pred_info = {
            'image_id': id,
            'label': label,
            'prediction': pred,
            # 'benign/maglinant': int(pred in [1,3,6]),
            'score': score,
        }
        score_list.append(pred_info)
    json_data = json.dumps(score_list, indent=4)
    # file = open(os.path.join(results_dir, 'score_preds_unifS-B_mixcutc.json'), 'w')
    # file.write(json_data)
    # file.close()

if __name__ == "__main__":
    import numpy as np
    from collections import OrderedDict
    outputs = np.load('./ckpts_sp2/output_uniformerB_0627_pretr_bs8_cutc-1-0.5-zeros/eval_on_final/preds.npy')
    print(outputs.shape)
    # targets = np.load('data/Annotations_with_ValidationSet/gt.npy')
    targets = np.load('./data/lldmmri_test_set/classification_dataset/labels/labels_test_guess.npy')
    print(targets.shape)
    # val_anno_file = 'data/Annotations_with_ValidationSet/labels.txt'
    val_anno_file = './data/classification_dataset/labels-0802-5fold/val_fold1.txt'
    acc = ACC(outputs, targets)
    f1 = F1_score(outputs, targets)
    recall = Recall(outputs, targets)
    # specificity = Specificity(outputs, targets)
    precision = Precision(outputs, targets)
    kappa = Cohen_Kappa(outputs, targets)
    report = cls_report(outputs, targets)
    cm = confusion_matrix(outputs, targets)
    metrics = OrderedDict([
        ('acc', acc),
        ('f1', f1),
        ('recall', recall),
        ('precision', precision),
        ('kappa', kappa),
        ('confusion matrix', cm),
        ('classification report', report),
    ])
    print(metrics)

    output_str = 'Test Results:\n'
    for key, value in metrics.items():
        if key == 'confusion matrix':
            output_str += f'{key}:\n {value}\n'
        elif key == 'classification report':
            output_str += f'{key}:\n {value}\n'
        else:
            output_str += f'{key}: {value}\n'

    results_file = './ckpts_sp2/output_uniformerB_0627_pretr_bs8_cutc-1-0.5-zeros/eval_on_final/results.txt'
    file = open(results_file, 'w')
    file.write(output_str)
    file.close()

    write_score2json(outputs, val_anno_file, '/'.join(results_file.split('/')[:-1]))