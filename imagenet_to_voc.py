#!/usr/bin/env python
import xml.etree.ElementTree as ET
import os
from os.path import join
import numpy as np


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_id):
    with open(ANNO_DIR + '{}.xml'.format(image_id), 'r') as in_file:
        tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    with open(LBLS_DIR + '{}.txt'.format(image_id), 'w') as out_file:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            class_dict[cls] = cls
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w,h), b)
            out_file.write(cls + " " + " ".join([str(a) for a in bb]) + '\n')

def voc_formatter(PROJECT_NM):
    PROJ_DIR = os.environ['HOME'] + '/' + PROJECT_NM
    ANNO_DIR = PROJ_DIR + '/annotated/'
    IMGS_DIR = PROJ_DIR + '/imgs/'
    LBLS_DIR = PROJ_DIR + '/labels/'
    DATA_DIR = os.environ['HOME'] + '/darknet/data/'

    img_lst = os.listdir(IMGS_DIR)
    np.random.shuffle(img_lst)

    N = len(img_lst)
    N_8 = int(0.8 * N)
    N_9 = int(0.9 * N)

    train_lst = img_lst[:N_8]
    val_lst = img_lst[N_8:N_9]
    test_lst = img_lst[N_9:]

    data_dict = {}
    data_dict['train'] = train_lst
    data_dict['validation'] = val_lst
    data_dict['test'] = test_lst

    class_dict = {}

    for image_set in data_dict.keys():
        with open(DATA_DIR + '{}_{}.txt'.format(PROJECT_NM, image_set), 'w') as list_file:
            for image_id in data_dict[image_set]:
                image_id = os.path.splitext(image_id)[0]
                list_file.write(IMGS_DIR + '{}.jpg\n'.format(image_id))
                try:
                    convert_annotation(image_id)
                except:
                    pass

    class_lst = sorted(list(class_dict.keys()))
    class_dict = {cls:idx for idx, cls in enumerate(class_lst)}

    for cls in class_lst:
        subprocess.call("find {} -type f -exec sed -i 's/{}/{}/g' {};".format(LBLS_DIR, cls, class_dict[cls]), shell=True)
    return class_lst
