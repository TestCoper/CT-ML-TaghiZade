import nibabel as nib
import numpy as np
from scipy import ndimage


def read_file(path):
    file = nib.load(path)
    return file.get_fdata(path)


def normalize_volume(volume):
    min_value = -1000
    max_value = 400
    volume[volume < min_value] = min_value
    volume[volume > max_value] = max_value
    volume = (volume - min_value) / (max_value - min_value)
    volume.astype("float32")
    return volume


def resize_volume(img):
    DEPTH = 64
    WIDTH = 128
    HEIGHT = 128
    
    depth = img.shape[-1]
    width = img.shape[0]
    height = img.shape[1]

    depth /= DEPTH
    width /= WIDTH
    height /= HEIGHT

    depth_factor = 1 / DEPTH
    width_factor = 1 / WIDTH
    height_factor = 1 / HEIGHT

    img = ndimage(img, rotate=90, reshape=False)

    img = ndimage(image, (width_factor, height_factor, depth_factor), order=1)
    return img


def read_and_process(path):
    file = read_file(path)
    file = normailze_volume(file)
    file = resize_volume(file)
    return file


