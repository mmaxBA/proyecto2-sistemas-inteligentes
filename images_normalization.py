import numpy as np
import os
from PIL import Image
import cv2

cwd = os.getcwd();
images_path = os.path.join(cwd, "images")
save_path = os.path.join(cwd, "images_nor")

THRESHOLD = 200

for dir in os.listdir(images_path):
    image_dir = os.path.join(images_path, dir)
    save_dir = os.path.join(save_path, dir)
    for filename in os.listdir(image_dir):
        img = Image.open(os.path.join(image_dir, filename))
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            bg_colour=(255, 255, 255)
            alpha = img.convert('RGBA').split()[-1]
            bg = Image.new("RGBA", img.size, bg_colour + (255,))
            bg.paste(img, mask=alpha)
            img = bg
        fn = lambda x : 255 if x > THRESHOLD else 0
        r = img.convert('L').point(fn, mode='1')
        image_path = os.path.join(save_dir, filename)
        r.save(image_path)

        image = cv2.imread(image_path)
        copy = image.copy()
        im2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(im2,128,255,0)
        contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        x,y,w,h = 0,0,0,0
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            if h/w > 0.7 and h/w < 1.3:
                break
        ROI = image[y:y+h, x:x+w]
        cv2.imwrite(image_path, ROI)
        print(filename)
        cv2.rectangle(copy,(x,y),(x+w,y+h),(36,255,12),2)

        img = Image.open(image_path)
        resize_img = img.resize((32,32))
        resize_img.convert("RGB")
        resize_img.save(image_path)
