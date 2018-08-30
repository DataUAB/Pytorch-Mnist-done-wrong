from utils.export_results import pkl_export, pkl_concat, pkl_import
import numpy as np
import os


def compute_acc(path, targets):
    
    results = np.array(pkl_import(path))
    
    print('Accuracy: {:.2f}'.format(np.sum(results == targets)) / targets.shape[0])
