import os
import random
import cv2
import joblib
import numpy as np
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

np.random.seed(8)
random.seed(42)


def load_annotations(img_name):
    annotated_boxes = []
    with open(f'WiderPerson/Annotations/{img_name}.jpg.txt', 'r') as file:
        lines = file.readlines()
        for obj in lines[1:]:
            class_label, x1, y1, x2, y2 = map(int, obj.split())
            if class_label == 1 and x2 > x1 and y2 > y1:
                annotated_boxes.append([x1, y1, x2, y2])
    return annotated_boxes


def save(features_name, X, y=None, new_dir=''):
    start_time = cv2.getTickCount() / cv2.getTickFrequency()
    if y is not None:
        np.save(f'features/y_{features_name}.npy', y)
        del y
    if new_dir != '':
        make_dir(f'features{new_dir}')
    np.save(f'features{new_dir}/X_{features_name}.npy', X)
    del X
    end_time = cv2.getTickCount() / cv2.getTickFrequency()
    print(f"Salvate features di {features_name}: {end_time - start_time}")


def make_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)


def count_bounding_boxes(img_names):
    num_bounding_boxes = 0
    for img_name in img_names:
        with open(f'WiderPerson/Annotations/{img_name}.jpg.txt', 'r') as file:
            lines = file.readlines()
            for obj in lines[1:]:
                class_label, x1, y1, x2, y2 = map(int, obj.split())
                if class_label == 1 and x2 > x1 and y2 > y1:
                    num_bounding_boxes += 1
    return num_bounding_boxes


def load_negative_images(img_dir_n):
    img_names_n = os.listdir(f'negative/{img_dir_n}')
    neg_images = []
    for img_name in img_names_n:
        neg_img = cv2.imread(f'negative/{img_dir_n}/{img_name}', cv2.IMREAD_GRAYSCALE)
        if neg_img is not None:
            neg_images.append(neg_img)
    return neg_images


def extract_features(img_dir_p, img_dir_n):
    start_time = cv2.getTickCount() / cv2.getTickFrequency()
    with open(f'split_positive/{img_dir_p}.txt', 'r') as file:
        img_names_p = file.read().split()
    num_bounding_boxes = count_bounding_boxes(img_names_p)
    neg_images = load_negative_images(img_dir_n)
    winSize = (64, 128)
    hog = cv2.HOGDescriptor(_winSize=winSize, _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)
    features = np.zeros((6 * num_bounding_boxes, hog.getDescriptorSize()), dtype=np.float16)
    labels = np.zeros((6 * num_bounding_boxes,), dtype=np.uint8)
    i = 0
    m = 0
    for img_name in img_names_p:
        m += 1
        if m % 10 == 0:
            print(f'\r{m / len(img_names_p) * 100:.1f}%', end='')
        with open(f'WiderPerson/Annotations/{img_name}.jpg.txt', 'r') as file:
            lines = file.readlines()
            img = cv2.imread(f'WiderPerson/Images/{img_name}.jpg', cv2.IMREAD_GRAYSCALE)
            for obj in lines[1:]:
                class_label, x1, y1, x2, y2 = map(int, obj.split())
                if class_label == 1 and x2 > x1 and y2 > y1:
                    cropped_img = img[y1:y2, x1:x2]
                    cropped_img = cv2.resize(cropped_img, winSize)
                    features[i * 6, :] = hog.compute(cropped_img)
                    labels[i * 6] = 1
                    for j in range(5):
                        neg_img = random.choice(neg_images)
                        neg_img_height, neg_img_width = neg_img.shape
                        crop_x1 = random.randint(0, neg_img_width - winSize[0])
                        crop_y1 = random.randint(0, neg_img_height - winSize[1])
                        crop_x2 = crop_x1 + winSize[0]
                        crop_y2 = crop_y1 + winSize[1]
                        neg_img_cropped = neg_img[crop_y1:crop_y2, crop_x1:crop_x2]
                        features[i * 6 + j + 1, :] = hog.compute(neg_img_cropped)
                        labels[i * 6 + j + 1] = 0
                        del neg_img_cropped, neg_img
                    i += 1
            del img
    end_time = cv2.getTickCount() / cv2.getTickFrequency()
    features_name = 'test'
    if img_dir_p == 'train_assignment':
        features_name = 'train'
    elif img_dir_p == 'val_assignment':
        features_name = 'val'
    print(f"Estrazione features di {features_name}: {end_time - start_time}")
    save(features_name, features, labels)
