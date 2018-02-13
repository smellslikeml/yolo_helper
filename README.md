# yolo_get_prep_config
These scripts help to streamline the installation and configuration of Yolo as well as the preprocessing of ImageNet formatted xml files into the VOC formatted text files for fine-tuning.

Assumes you have raw images as well as the xml files defining the bounding boxes and labels.

We recommend using: [LabelImg](https://github.com/tzutalin/labelImg) to perform the labeling.

You might obtain images through your favorite search engine. Using the Firefox browser, the [Google Image Downloader](https://addons.mozilla.org/en-US/firefox/addon/google-images-downloader/) plugin makes downloading several hundred images a snap.

## Getting Started
```
git clone https://addons.mozilla.org/en-US/firefox/addon/google-images-downloader/
cd yolo_get_prep_config
python yolo_finetune.py
```
You will be prompted to enter a name for the project and the yolo repo will be installed if it is not already in the home directory.

## TODO
* Add argparse
* Make options for CUDA/OpenCV
* options for train/val/test splits
* other model weights & config

