import os
from multiprocessing import Pool
import cv2
import numpy as np

n_stack = len([x for x in filter(lambda x: x[0] != '.', os.listdir('input'))])
n_frames = len([x for x in filter(lambda x: x[0] != '.', os.listdir('input/1/02'))])
# scene = 2
size_image = cv2.imread('input/{}/{}/{}.jpg'.format(1, str(1).zfill(2), str(1).zfill(6)))
sz = size_image.shape
height = sz[0]
width = sz[1]
print('Stacking', n_frames, 'frames with', n_stack, height, 'x', width, 'images each.')


def process_frame(scene, frame_index):
    # Specify the number of iterations.
    number_of_iterations = 3
    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 5e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    print('Stacking frame', frame_index)

    stack = np.zeros((n_stack, height, width, 3), dtype=np.float32)
    stack_grey = np.zeros((n_stack, height, width), dtype=np.float32)
    for layer_index in range(0, n_stack):
        im = cv2.imread('input/{}/{}/{}.jpg'.format(layer_index + 1, str(scene).zfill(2), str(frame_index).zfill(6)))
        stack[layer_index] = im
        stack_grey[layer_index] = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    ccs = []
    for layer_index, im_gray in enumerate(stack_grey[1:], 1):
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC(stack_grey[0], im_gray, warp_matrix, cv2.MOTION_EUCLIDEAN, criteria)
        ccs.append(cc)

        stack[layer_index] = cv2.warpAffine(stack[layer_index], warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    print('avg ecc: ', np.mean(ccs))

    scaled_stack = np.zeros((n_stack, 2*height, 2*width, 3), dtype=np.float32)
    for layer_index, im in enumerate(stack):
        scaled_stack[layer_index] = cv2.resize(im, (width*2, height*2), interpolation=cv2.INTER_LANCZOS4)

    median_image = np.median(scaled_stack, axis=0)
    cv2.imwrite('out/{}_median/{}.jpg'.format(str(scene).zfill(2), str(frame_index).zfill(6)), median_image)

    average_image = np.mean(scaled_stack, axis=0)
    cv2.imwrite('out/{}_average/{}.jpg'.format(str(scene).zfill(2), str(frame_index).zfill(6)), average_image)


try:
    pool = Pool()
    pool.map(process_frame, range(1, n_frames))
finally:
    pool.close()
    pool.join()


# for i in range(1, n_frames):
#     process_frame(i)
