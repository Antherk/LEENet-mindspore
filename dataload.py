import os
import random
import glob

def find_image_file(images_path):
    assert os.path.isdir(images_path), 'Error : %s is not a valid directory' % dir
    image_list_lowlight = glob.glob(images_path + '*_000.*') # 00 0
    train_list = image_list_lowlight[:]
    random.shuffle(train_list)
    return train_list