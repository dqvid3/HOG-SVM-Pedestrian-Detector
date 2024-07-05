import cv2
import numpy as np
from utils.medie_dataset import fattori_di_scala, target_size
from utils.sliding_window import detect_pedestrians
from utils.non_maximum_suppression import non_maximum_suppression
from report_finale import calcola_risultati
from utils.utility_functions import load_annotations

T_values = np.linspace(0.1, 0.5, num=3)
f1_dict = {}

for T in T_values:
    with open('split_positive/val_assignment.txt', 'r') as file:
        img_names = file.read().split()
    tps = fps = fns = 0
    f1s = []
    scales = fattori_di_scala()
    size = target_size()
    i = 0
    for img_name in img_names[:]:
        i += 1
        print(f'\rT = {T} - {i / len(img_names) * 100:.1f}%', end='')
        boxes, confidence = detect_pedestrians(cv2.imread(f"WiderPerson/Images/{img_name}.jpg"), scales)
        picked_boxes, picked_scores = non_maximum_suppression(boxes, confidence, T)
        img_boxes = load_annotations(img_name)
        tp, fp, fn = calcola_risultati(picked_boxes, img_boxes)
        tps += tp
        fps += fp
        fns += fn
        if tp == 0:
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1s.append((2 * precision * recall) / (precision + recall))
    print(tps, fps, fns, f1s)
    f1 = sum(f1s) / len(f1s)
    f1_dict[T] = f1

best_T = max(f1_dict, key=f1_dict.get)
best_f1 = f1_dict[best_T]
print("Best T:", best_T)
print("Best F1-score:", best_f1)
f = open("best_T.txt", "w")
f.write(str(best_T))
f.close()
