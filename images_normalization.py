import numpy as np
import os
from PIL import Image
import cv2

cwd = os.getcwd();
images_path = os.path.join(cwd, "images")

th = 128

for dir in os.listdir(images_path):
    image_dir = os.path.join(images_path, dir)
    for filename in os.listdir(image_dir):
        im = np.array(Image.open(os.path.join(image_dir, filename)))
        im_bool = im > th
        im_bin_128 = (im > th) * 255
        Image.fromarray(np.uint8(im_bin_128)).save(os.path.join(os.path.join(os.path.join(cwd, "images_nor"), dir), filename))
