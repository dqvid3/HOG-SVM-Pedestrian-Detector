import os
import cv2
import joblib
import numpy as np
from features_preprocessing import transform_data


def sliding_window(image, window_size, stride):
    for y in range(0, image.shape[0] - window_size[1], stride):
        for x in range(0, image.shape[1] - window_size[0], stride):
            yield x, y, image[y:y + window_size[1], x:x + window_size[0]]


def pyramid(image, scales):
    images = [image,]
    for scale in scales:
        w = int(image.shape[1] * scale)
        temp = cv2.GaussianBlur(image, (3, 3), 0)
        temp = cv2.resize(temp, (w, int(w / image.shape[1] * image.shape[0])))
        images.append(temp)
    return images


def load_preprocessors():
    standardScaler = pca = ipca = None
    for processorname in os.listdir('preprocessing'):
        if processorname.endswith('.pkl'):
            processor = joblib.load(f'preprocessing/{processorname}')
            name = processor.__class__.__name__.lower()
            if 'incrementalpca' == name:
                ipca = processor
            elif 'pca' == name:
                pca = processor
            else:
                standardScaler = processor
    return standardScaler, pca, ipca


def apply_preprocessing(hog_features):
    standardScaler, pca, ipca = load_preprocessors()
    if standardScaler is not None:
        hog_features = standardScaler.transform(hog_features)
        if pca is not None:
            hog_features = pca.transform(hog_features)
        elif ipca is not None:
            batch_size = 50
            hog_features = transform_data('windows', ipca, batch_size, False, hog_features)
    return hog_features


def detect_pedestrians(image, scales):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for model in os.listdir('models'):
        if model.startswith('best_svm_model'):
            svm = joblib.load(f'models/{model}')
            break
    detections = []
    confidences = []
    winSize = (64, 128)
    target_size = (800, 600) #trovata tramite medie_dataset.py seguendo assunti empirici
    original_height, original_width = image.shape[:2]
    image = cv2.resize(image, target_size)
    scale_factor_y = original_height / image.shape[0]
    scale_factor_x = original_width / image.shape[1]
    hog = cv2.HOGDescriptor(_winSize=winSize, _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)
    all_hog_features = []
    locations = []
    for resized in pyramid(image, scales):
        scale = image.shape[0] / resized.shape[0]
        for (x, y, window) in sliding_window(resized, window_size=(64, 128), stride=16):
            if window.shape[0] != 128 or window.shape[1] != 64:
                continue
            hog_features = hog.compute(window)
            all_hog_features.append(hog_features)
            locations.append((x, y, scale))

    all_hog_features = np.vstack(all_hog_features)
    all_hog_features = apply_preprocessing(all_hog_features)

    predictions = svm.predict(all_hog_features)
    confs = svm.decision_function(all_hog_features)

    for i, prediction in enumerate(predictions):
        confidence = confs[i]
        if prediction == 1 and confidence > .7:
            x, y, scale = locations[i]
            x1, y1 = int(x * scale * scale_factor_x), int(y * scale * scale_factor_y)
            x2, y2 = int((x + 64) * scale * scale_factor_x), int((y + 128) * scale * scale_factor_y)
            detections.append((x1, y1, x2, y2))
            confidences.append(confidence)

    return detections, confidences
