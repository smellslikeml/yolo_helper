#!/usr/bin/env python
import os
import subprocess
from imagenet2voc import *

HOME_DIR = os.environ['HOME']
YOLO_DIR = HOME_DIR + '/darknet/'

PROJECT_NM = input('Please Enter a Unique Project Name: ')
while not os.path.exists(HOME_DIR + '/' + PROJECT_NM):
    os.mkdir(HOME_DIR + '/' + PROJECT_NM)

while not os.path.exists('/'.join([HOME_DIR, PROJECT_NM, 'labels'])):
    os.mkdir('/'.join([HOME_DIR, PROJECT_NM, 'labels']))

if not os.path.exists(YOLO_DIR):
    subprocess.call('git clone https://github.com/pjreddie/darknet', shell=True)
    os.chdir(YOLO_DIR)
    subprocess.call('make', shell=True)

if not os.path.exists(YOLO_DIR + 'darknet19_448.conv.23'):
    subprocess.call('wget https://pjreddie.com/media/files/darknet19_448.conv.23', shell=True)

#class_lst = voc_formatter(PROJECT_NM)
class_lst = main(PROJECT_NM)
num_classes = len(class_lst)

if not os.path.exists(YOLO_DIR + 'data/{}.names'.format(PROJECT_NM)):
    with open(YOLO_DIR + 'data/{}.names'.format(PROJECT_NM), 'w') as nms_out:
        for cls in class_lst:
            nms_out.write(cls + '\n')

if not os.path.exists(YOLO_DIR + PROJECT_NM + '_backup'):
    os.mkdir(YOLO_DIR + PROJECT_NM + '_backup')

if not os.path.exists(YOLO_DIR + 'cfg/{}.data'.format(PROJECT_NM)):
    YOLO_DATA = YOLO_DIR + 'data'
    with open(YOLO_DIR + 'cfg/{}.data'.format(PROJECT_NM), 'w') as cfg_out:
        cfg_out.write('classes={}\n'.format(num_classes))
        cfg_out.write('train={}\n'.format(YOLO_DATA + '/{}_train.txt'.format(PROJECT_NM)))
        cfg_out.write('valid={}\n'.format(YOLO_DATA + '/{}_valid.txt'.format(PROJECT_NM)))
        cfg_out.write('names={}\n'.format(YOLO_DATA + '/{}.names'.format(PROJECT_NM)))
        cfg_out.write('backup={}_backup\n'.format(PROJECT_NM))
else:
    with open(YOLO_DIR + 'cfg/{}.data'.format(PROJECT_NM), 'r') as infile:
        data = infile.readlines()
    num_classes = int(data[0].split('=')[-1].strip())

if not os.path.exists(YOLO_DIR + 'cfg/yolo-{}.cfg'.format(PROJECT_NM)):
    with open(YOLO_DIR + 'cfg/yolo-voc.cfg', 'r') as infile:
        data = infile.readlines()

    data[2] = '# batch=1\n'
    data[3] = '# subdivisions=1\n'
    data[5] = 'batch=64\n'
    data[6] = 'subdivisions=8\n'
    data[-15] = 'classes={}\n'.format(num_classes)
    data[-22] = 'filters={}\n'.format(5 * (num_classes + 5))

    with open(YOLO_DIR + 'cfg/yolo-{}.cfg'.format(PROJECT_NM), 'w') as outfile:
        for line in data:
            outfile.write(line)

train_flag = input('Do you wish to train? [Y/N]')
if train_flag == 'Y':
    os.chdir(YOLO_DIR)
    YOLO_TRAIN_CMD = './darknet detector train cfg/{}.data cfg/yolo-{}.cfg darknet19_448.conv.23 -gpu 0,1'.format(PROJECT_NM, PROJECT_NM)
    num_gpus = int(input('How many gpus? '))
    if num_gpus > 0:
        gpu_flg = ' -gpus ' + ','.join(list(map(str, list(range(num_gpus)))))
        YOLO_TRAIN_CMD += gpu_flg
    subprocess.call(YOLO_TRAIN_CMD, shell=True)
else:
    print('Configuration Complete!')

