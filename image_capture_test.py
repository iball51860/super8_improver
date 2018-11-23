import cv2
import matplotlib
import numpy
import time

# cam = cv2.VideoCapture(0)
# print('sleep 2')
# time.sleep(2)
# tookPicture, image = cam.read()  # captures image
# print('sleep 2')
# time.sleep(2)
# tookPicture, image = cam.read()  # captures image
# print('took picture', tookPicture)
#
# if tookPicture:
#     print(tookPicture, [len(image) if image is not None else 0])
#     cv2.imshow("Test Picture", image)  # displays captured image
#     cv2.imwrite("test.jpg", image)
# else:
#     print('Couldn\'t take a picture')
cv2.findTransformECC()

from __future__ import print_function
import uvc
import logging
logging.basicConfig(level=logging.INFO)

dev_list =  uvc.device_list()
print(dev_list)
cap = uvc.Capture(dev_list[0]['uid'])

# Uncomment the following lines to configure the Pupil 200Hz IR cameras:
# controls_dict = dict([(c.display_name, c) for c in cap.controls])
# controls_dict['Auto Exposure Mode'].value = 1
# controls_dict['Gamma'].value = 200

print(cap.avaible_modes)
for x in range(10):
    print(x)
    cap.frame_mode = (640,480,30)
    for x in range(100):
        frame = cap.get_frame_robust()
        print(frame.img.shape)
        #cv2.imshow("img",frame.gray)
        #cv2.waitKey(1)
cap = None