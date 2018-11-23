import os
import cv2
import numpy as np
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt

n_stack = len(os.listdir('input'))
n_frames = len(os.listdir('input/1/02'))
print('Stacking', n_frames, 'frames with', n_stack, 'images each')

def process_frame(frame_index):
    print('Stacking frame', frame_index)

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


# Read the images to be aligned
im1 = cv2.imread("input/1/02/000087.jpg")
im2 = cv2.imread("input/2/02/000087.jpg")

height, width, rgb = im1.shape
# im1 = cv2.resize(im1, (height*2, width*2))
# im2 = cv2.resize(im2, (height*2, width*2))

# Convert images to grayscale
im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

# Find size of image1
sz = im1.shape

# Define the motion model
warp_mode = cv2.MOTION_EUCLIDEAN

# Define 2x3 or 3x3 matrices and initialize the matrix to identity
warp_matrix = np.eye(2, 3, dtype=np.float32)

# Specify the number of iterations.
number_of_iterations = 1

# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = 1e-10

# Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

# Run the ECC algorithm. The results are stored in warp_matrix.
(cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)


if warp_mode == cv2.MOTION_HOMOGRAPHY:
    # Use warpPerspective for Homography
    im2_aligned = cv2.warpPerspective(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
else:
    # Use warpAffine for Translation, Euclidean and Affine
    im2_aligned = cv2.warpAffine(im2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

# Show final results
# Image.fromarray(im2_aligned).save('out/testalign.jpg')
cv2.imwrite('out/testalign2.jpg', im2_aligned)
