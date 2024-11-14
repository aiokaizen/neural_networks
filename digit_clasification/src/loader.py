import json
from PIL import Image
import random
import os
import numpy as np

def load(type='testing', shuffle=True, return_type='array'):
    image_arrays = []
    label_arrays = []

    images_filename = f"../data/{'t10k' if type == 'testing' else 'train'}-images.idx3-ubyte"
    labels_filename = f"../data/{'t10k' if type == 'testing' else 'train'}-labels.idx1-ubyte"

    with open(images_filename, 'rb') as f:
        f.read(2)  # Skip the first two zeros
        int.from_bytes(f.read(1), 'big')  # Data type
        int.from_bytes(f.read(1), 'big')  # Number of dimensions
        int.from_bytes(f.read(4), 'big')  # D1: Number of images
        img_width = int.from_bytes(f.read(4), 'big')  # D2: Image width
        img_height = int.from_bytes(f.read(4), 'big')  # D3: Image height

        img_bytes = f.read(img_width * img_height)
        while img_bytes:
            if return_type == 'image':
                img = Image.frombytes('L', (img_width, img_height), img_bytes)
                image_arrays.append(img)
            else:
                array = [b for b in img_bytes]
                image_arrays.append(array)
            img_bytes = f.read(img_width * img_height)

    with open(labels_filename, 'rb') as fl:
        fl.read(8)  # Skip header bytes

        label = fl.read(1)
        while label != b'':
            label_arrays.append(int.from_bytes(label, 'big'))
            label = fl.read(1)

    if shuffle:
        data = list(zip(image_arrays, label_arrays))
        random.shuffle(data)
        image_arrays, label_arrays = zip(*data)
        image_arrays = list(image_arrays)
        label_arrays = list(label_arrays)

    return (image_arrays, label_arrays)

def load_from_images(path='../data/new_data', shuffle=True):
    images = []
    labels = []
    for _, _, files in os.walk(path):
        files.sort()
        for f in files:
            full_path = os.path.join(path, f)
            image = Image.open(full_path)
            image = image.convert('L')
            image_array = np.array(image).tolist()
            images.append(flatten(image_array))
            labels.append(int(f.split('.')[0].split('_')[0]))

    if shuffle:
        data = list(zip(images, labels))
        random.shuffle(data)
        image_arrays, label_arrays = zip(*data)
        images = list(image_arrays)
        labels = list(label_arrays)

    return images, labels

def flatten(array):
    flat_list = []
    for i in range(len(array)):
        for j in range(len(array[0])):
            flat_list.append(array[i][j])
    return flat_list


if __name__ == '__main__':
    images, labels = load('train')
    images[123].show()
    print(labels[123])
