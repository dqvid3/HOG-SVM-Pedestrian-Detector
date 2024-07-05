# prendere le bounding box per cui pred è 1
# la threshold T serve per capire quando dobbiamo applicare la non_maximum_suppression, ovvero quando ci sono due bounding box sovrapposte per più/meno di T
# poi delle box sopravvissute alla non_maximum_suppression prendiamo solo quella per cui la score (calcolato con decision_function()) è massima

def iou(box1, box2):
    """
    Calcola l'Intersection over Union (IoU) tra due bounding box.

    Args:
        box1 (list): Prima bounding box nella forma [x1, y1, x2, y2]
        box2 (list): Seconda bounding box nella forma [x1, y1, x2, y2]

    Returns:
        float: Il valore IoU tra le due bounding box
    """
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])

    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)

    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area

    return iou


def non_maximum_suppression(boxes, scores, threshold):
    """
    Esegue la Non-Maximum Suppression (NMS) su una lista di bounding box.

    Args:
        boxes (list of lists): Lista di bounding box, dove ogni box è una lista di [x1, y1, x2, y2]
        scores (list): Lista di confidenze corrispondenti a ciascuna box
        threshold (float): Soglia IoU per la soppressione

    Returns:
        list of lists: Lista delle box rimanenti dopo la NMS
        list: Lista delle confidenze corrispondenti
    """
    if len(boxes) == 0:
        return [], []

    n = len(scores)
    sorted_indices = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if scores[i] < scores[j]:
                sorted_indices[i], sorted_indices[j] = sorted_indices[j], sorted_indices[i]
    sorted_boxes = []
    for i in sorted_indices:
        sorted_boxes.append(boxes[i])
    boxes = sorted_boxes
    sorted_scores = []
    for index in sorted_indices:
        sorted_scores.append(scores[index])
    scores = sorted_scores
    picked_boxes = []
    picked_scores = []

    while len(boxes) > 0:
        picked_boxes.append(boxes[0])
        picked_scores.append(scores[0])

        iou_scores = []
        for box in boxes[1:]:
            iou_scores.append(iou(boxes[0], box))
        ious = iou_scores

        filtered_boxes = []
        for i in range(len(ious)):
            if ious[i] <= threshold:
                filtered_boxes.append(boxes[i + 1])
        boxes = filtered_boxes
        filtered_scores = []
        for i in range(len(ious)):
            if ious[i] <= threshold:
                filtered_scores.append(scores[i + 1])
        scores = filtered_scores

    return picked_boxes, picked_scores

