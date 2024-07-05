import os
import cv2
from utils.medie_dataset import fattori_di_scala, target_size
from utils.utility_functions import load_annotations
from utils.non_maximum_suppression import non_maximum_suppression, iou
from utils.sliding_window import detect_pedestrians
from tabulate import tabulate


def calcola_risultati(detected_boxes, annotated_boxes):
    """
    Calcola TP, FP e FN a partire dalle box annotate e dalle box rilevate.
    """
    tp = 0
    fp = 0
    matched_indices = set()  # Indici delle annotazioni già abbinate

    for box in detected_boxes:
        found_match = False
        for i, annotated_box in enumerate(annotated_boxes):
            if i in matched_indices:
                continue
            iou_val = iou(box, list(annotated_box))
            if iou_val > 0.5:
                found_match = True
                matched_indices.add(i)
                break
        if found_match:
            tp += 1
        else:
            fp += 1

    fn = len(annotated_boxes) - len(matched_indices)
    return tp, fp, fn


def modello_opencv():
    with open("split_positive/test_assignment.txt", 'r') as file:
        img_names = file.read().split()
    tps = fps = fns = 0
    precisions = []
    recalls = []
    i = 0
    for img_name in img_names:
        i += 1
        print(f'\rModello open cv: {i / len(img_names) * 100:.1f}%', end='')
        img = cv2.imread(f'WiderPerson/Images/{img_name}.jpg')
        founded_bb_image = disegna_rettangoli(img)
        annotated_bb_image = load_annotations(img_name)
        tp, fp, fn = calcola_risultati(founded_bb_image, annotated_bb_image)
        tps += tp
        fps += fp
        fns += fn
        if tp == 0:
            continue
        precisions.append(tp / (tp + fp))
        recalls.append(tp / (tp + fn))
    print('\n')
    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return tps, fps, fns, precision, recall, f1_score


def disegna_rettangoli(img, disegna=False, boxes=None, original=None):
    if boxes is None:
        color = (0, 0, 255)  # Rosso
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(hog.getDefaultPeopleDetector())
        regions, _ = hog.detectMultiScale(img, winStride=(4, 4), padding=(0, 0), scale=1.05)
        # Ricordiamo che calcola_risultati lavora immagine per immagine
        founded_bb_image = []
        for (x, y, w, h) in regions:
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            founded_bb_image.append([x1, y1, x2, y2])
    else:
        founded_bb_image = boxes
        color = (255, 0, 0)  # Blu
    if disegna:
        if original is not None:
            founded_bb_image = load_annotations(original)
            color = (0, 255, 0) # Verde
        for x1, y1, x2, y2 in founded_bb_image:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    else:
        return founded_bb_image


def scorri_testset(scelta):
    f = open("best_T.txt", "r")
    best_T = float(f.read())
    f.close()
    # Testa dal Test Set (immagini annotate)
    with open('split_positive/test_assignment.txt', 'r') as file:
        img_names = file.read().split()
    tps = fps = fns = 0
    precisions = []
    recalls = []
    i = 0
    scales = fattori_di_scala()
    for img_name in img_names:
        path = f"WiderPerson/Images/{img_name}.jpg"
        img = cv2.imread(path)
        boxes, confidences = detect_pedestrians(img, scales)
        if scelta == 2:
            i += 1
            print(f'\rIl nostro modello: {i / len(img_names) * 100:.1f}%', end='')
            img_boxes = load_annotations(img_name)
            tp, fp, fn = calcola_risultati(boxes, img_boxes)
            tps += tp
            fps += fp
            fns += fn
            if tp == 0:
                continue
            precisions.append(tp / (tp + fp))
            recalls.append(tp / (tp + fn))
        else:
            disegna_rettangoli(img, True)  # Disegna in rosso le box di OpenCV
            boxes, _ = non_maximum_suppression(boxes, confidences, best_T)
            disegna_rettangoli(img, True, original=img_name)  # Disegna in verde le box annotate
            disegna_rettangoli(img, True, boxes)  # Disegna in blu le box del nostro modello
            cv2.imshow(path, img)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('n'):
                cv2.destroyWindow(path)
            elif key == ord('q'):
                cv2.destroyAllWindows()
                break
    if scelta == 2:
        precision = sum(precisions) / len(precisions)
        recall = sum(recalls) / len(recalls)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        print('\n')
        opencv_results = modello_opencv()
        
        table = [
            ["Modello", "TP", "FP", "FN", "Precision", "Recall", "F1-Score"],
            ["Nostro Modello", tps, fps, fns, precision, recall, f1_score],
            ["OpenCV", *opencv_results]
        ]
        
        print("\nConfronTo tra i modelli:")
        print(tabulate(table, headers="firstrow", floatfmt=".4f", numalign="right"))


def main():
    scelta = int(input("(1 = Scorri testset, 2 = Mostra risultati sul testset, 3 = Mostra esempi sul nostro modello): "))
    if scelta == 3:
        f = open("best_T.txt", "r")
        best_T = float(f.read())
        f.close()
        size = target_size()
        scales = fattori_di_scala()
        dir = 'test_images'
        for path in os.listdir(dir):
            img = cv2.imread(f'{dir}/{path}')
            boxes, confidences = detect_pedestrians(img, scales)
            boxes, _ = non_maximum_suppression(boxes, confidences, .1)
            disegna_rettangoli(img, True, boxes)  # Disegna in blu le box del nostro modello
            cv2.imshow(path, img)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('n'):
                cv2.destroyWindow(path)
            elif key == ord('q'):
                cv2.destroyAllWindows()
                break
    elif scelta > 3:
        print("Non esiste questa scelta")
    else:
        scorri_testset(scelta)


if __name__ == "__main__":
    main()
