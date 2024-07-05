import os
from PIL import Image


def target_size():
    image_folder = 'WiderPerson/Images'

    widths = []
    heights = []

    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            img_path = os.path.join(image_folder, filename)
            img = Image.open(img_path)
            widths.append(img.width)
            heights.append(img.height)

    mean_width = sum(widths) / len(widths)
    mean_height = sum(heights) / len(heights)

    return ratio_piu_vicino(int(mean_width), int(mean_height))


def fattori_di_scala():
    image_folder = './WiderPerson/Images'

    annotation_folder = './WiderPerson/Annotations'

    image_list_file = 'split_positive/train_assignment.txt'

    with open(image_list_file, 'r') as f:
        image_list = [line.strip() for line in f.readlines()]

    width_max_list = []
    width_min_list = []
    width_med_list = []
    height_max_list = []
    height_min_list = []
    height_med_list = []

    for image_name in image_list:
        annotation_file = os.path.join(annotation_folder, f'{image_name}.jpg.txt')

        with open(annotation_file, 'r') as f:
            annotation_lines = [line.strip() for line in f.readlines()[1:]]  # salta la prima riga

        annotations = [line.split() for line in annotation_lines if line.split()[0] == '1']

        img_path = f"{image_folder}/{image_name}.jpg"
        img = Image.open(img_path)
        widths = [(int(annotation[3]) - int(annotation[1])) / img.width for annotation in annotations]
        heights = [(int(annotation[4]) - int(annotation[2])) / img.height for annotation in annotations]

        width_max = max(widths) if widths else 0
        width_min = min(widths) if widths else 0
        width_med = sum(widths) / len(widths) if widths else 0
        height_max = max(heights) if heights else 0
        height_min = min(heights) if heights else 0
        height_med = sum(heights) / len(heights) if heights else 0

        width_max_list.append(width_max)
        width_min_list.append(width_min)
        width_med_list.append(width_med)
        height_max_list.append(height_max)
        height_min_list.append(height_min)
        height_med_list.append(height_med)

    width_max_mean = sum(width_max_list) / len(width_max_list)
    width_min_mean = sum(width_min_list) / len(width_min_list)
    width_med_mean = sum(width_med_list) / len(width_med_list)
    height_max_mean = sum(height_max_list) / len(height_max_list)
    height_min_mean = sum(height_min_list) / len(height_min_list)
    height_med_mean = sum(height_med_list) / len(height_med_list)
    fattore1_width = width_med_mean / width_max_mean
    fattore1_height = height_med_mean / height_max_mean
    fattore1 = (fattore1_width + fattore1_height) / 2
    fattore2_width = width_min_mean / width_med_mean
    fattore2_height = height_min_mean / height_med_mean
    fattore2 = (fattore2_width + fattore2_height) / 2
    return (fattore1, fattore1 * fattore2)


def ratio_piu_vicino(width, stride=64):
    # Cerco il multiplo di stride piu' vicino (per eccesso) a width:
    num = width
    while (num % stride != 0):
        num += 1
    # Adeguo il denominatore in modo da avere un aspect ratio di 4:3:
    # 4:3 e' una considerazione empirica.
    # num : den = 4 : 3
    den = int(num * 3 / 4)

    return num, den

