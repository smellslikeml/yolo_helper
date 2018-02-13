#!/usr/bin/env python
import os
import subprocess
import numpy as np
import xml.etree.ElementTree as ET
import sys


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

def main(PROJECT_NM):
    PROJ_DIR = os.environ['HOME'] + '/' + PROJECT_NM
    ANNO_DIR = PROJ_DIR + '/annotated/'
    IMGS_DIR = PROJ_DIR + '/raw/' #'/imgs/'
    LBLS_DIR = PROJ_DIR + '/labels/'
    DATA_DIR = os.environ['HOME'] + '/darknet/data/'

    if not os.path.exists(LBLS_DIR):
        os.mkdir(LBLS_DIR)

    img_lst = os.listdir(IMGS_DIR)

    class_dict = {}
    full_lst = []

    for img in img_lst:
        img_id = os.path.splitext(img)[0]
        try:
            with open(ANNO_DIR + '{}.xml'.format(img_id), 'r') as in_file:
                tree=ET.parse(in_file)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)
            with open(LBLS_DIR + '{}.txt'.format(img_id), 'w') as out_file:
                for obj in root.iter('object'):
                    cls = obj.find('name').text
                    class_dict[cls] = cls
                    xmlbox = obj.find('bndbox')
                    b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                    bb = convert((w,h), b)
                    out_string = cls + ' ' + ' '.join(list(map(str, bb))) + '\n'
                    out_file.write(out_string)
            if os.path.exists(LBLS_DIR + '{}.txt'.format(img_id)):
                full_lst.append(PROJ_DIR + '/raw/' + img)
        except:
            pass

    np.random.shuffle(full_lst)
    N = len(full_lst)
    N_8 = int(0.8 * N)
    N_9 = int(0.9 * N)

    train_lst = full_lst[:N_8]
    val_lst = full_lst[N_8:N_9]
    test_lst = full_lst[N_9:]

    data_dict = {}
    data_dict['train'] = train_lst
    data_dict['valid'] = val_lst
    data_dict['test'] = test_lst

    for image_set in data_dict.keys():
        with open(DATA_DIR + '{}_{}.txt'.format(PROJECT_NM, image_set), 'w') as list_file:
            for path in data_dict[image_set]:
                list_file.write(path + '\n')

    class_lst = sorted(list(class_dict.keys()))
    class_dict = {cls:idx for idx, cls in enumerate(class_lst)}
    for cls in class_lst:
        swap_string = "find {} -type f -exec sed -i 's/{}/{}/g'".format(LBLS_DIR, cls, class_dict[cls])
        swap_string += " {} \;"
        subprocess.call(swap_string, shell=True)
    return class_lst

if __name__ == '__main__':
    print(main('yellow_leaves'))
