import os
import numpy as np
from PIL import Image, ImageFilter
from multiprocessing import Pool

n_stack = len(os.listdir('input'))
n_frames = len(os.listdir('input/1/02'))
print('Stacking', n_frames, 'frames with', n_stack, 'images each')
width, height = Image.open('input/1/01/000001.jpg').size
width = width * 2
height = height * 2
scene = 2


def process_frame(frame_index):
    print('Stacking', frame_index, 'of', n_frames)

    stack = np.zeros((n_stack, height, width, 3))
    for layer_index in range(1, n_stack+1):
        print('layer', layer_index)
        image = Image.open('input/{}/{}/{}.jpg'.format(layer_index, str(scene).zfill(2), str(frame_index).zfill(6)))
        image = image.resize((width, height), resample=Image.LANCZOS)
        image = image.filter(ImageFilter.GaussianBlur(radius=5))
        stack[layer_index - 1] = np.array(image)

    image_median = np.median(stack, axis=0).astype(np.uint8)
    out_image = Image.fromarray(image_median)
    out_image.save('out/{}/{}.jpg'.format(str(scene).zfill(2), str(frame_index).zfill(6)))


try:
    pool = Pool()
    pool.map(process_frame, range(1, n_frames))
finally:
    pool.close()
    pool.join()
