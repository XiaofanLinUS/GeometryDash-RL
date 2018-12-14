from __future__ import print_function
import cv2 as cv
import argparse
max_lowThreshold = 300
max_ratio = 20
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
title_trackbar2 = 'rate:'
ratio = 2
kernel_size = 3

def changeRatio(val):
    ratio = val
    print(ratio)

def CannyThreshold(val):
    low_threshold = val
    print(ratio)
    img_blur = cv.blur(src_gray, (3,3))
    detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
    mask = detected_edges != 0
    dst = src * (mask[:,:,None].astype(src.dtype))
    cv.imshow(window_name, dst)


parser = argparse.ArgumentParser(description='Code for Canny Edge Detector tutorial.')
parser.add_argument('--input', help='Path to input image.', default='cat.jpg')
args = parser.parse_args()
src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image: ', args.input)
    exit(0)
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.namedWindow(window_name)
cv.createTrackbar(title_trackbar, window_name , 0, max_lowThreshold, CannyThreshold)
cv.createTrackbar(title_trackbar2, window_name , 0, max_ratio, changeRatio)
CannyThreshold(0)
k = cv.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv.destroyAllWindows()