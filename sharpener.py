import os
from multiprocessing import Pool

from PIL import Image, ImageFilter

n_stack = len(os.listdir('input'))
n_frames = len(os.listdir('input/1/02'))
print('Stacking', n_frames, 'frames with', n_stack, 'images each')
width = 2408
height = 1770
scene = 2


def sharpen_frame(frame_index):
    print('Sharpening', frame_index, 'of', n_frames)

    image = Image.open('out/{}_median/{}.jpg'.format(str(scene).zfill(2), str(frame_index).zfill(6)))
    image = image.filter(ImageFilter.UnsharpMask(radius=10, percent=50, threshold=0))

    image.save('out/{}_median_sharp/{}.jpg'.format(str(scene).zfill(2), str(frame_index).zfill(6)))

    image = Image.open('out/{}_average/{}.jpg'.format(str(scene).zfill(2), str(frame_index).zfill(6)))
    image = image.filter(ImageFilter.UnsharpMask(radius=10, percent=50, threshold=0))

    image.save('out/{}_average_sharp/{}.jpg'.format(str(scene).zfill(2), str(frame_index).zfill(6)))


try:
    pool = Pool()
    pool.map(sharpen_frame, range(1, n_frames))
finally:
    pool.close()
    pool.join()
