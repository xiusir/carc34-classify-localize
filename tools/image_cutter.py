#!/usr/bin/python
# -*- coding: utf-8 -*-
# import the necessary packages
from imutils import paths
import math
import argparse
import cv2

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

def gray_image_laplacian(image):
    # 灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
    help="path to input directory of images")
ap.add_argument("-t", "--threshold", type=float, default=100.0,
    help="focus measures that fall below this value will be considered 'blurry'")
args = vars(ap.parse_args())

# loop over the input images
for imagePath in paths.list_images(args["images"]):
    # load the image, convert it to grayscale, and compute the
    # focus measure of the image using the Variance of Laplacian
    # method
    image = cv2.imread(imagePath)
    ###通道拆分
    ###b,g,r = cv2.split(image)
    ###image_new = cv2.merge((b,g,r))
    ###cv2.imshow("Image", image)
    ###cv2.imshow("Red", r)
    ###cv2.imshow("Green", g)
    ###cv2.imshow("Blue", b)
    ###cv2.imshow("ImageNew", image_new)
    ###key = cv2.waitKey(0)

    ### 灰度化
    ##gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ##fm = variance_of_laplacian(gray)
    print (imagePath, image.shape)

    # CONSTANT 用颜色填充
    BLACK = [0,0,0]
    # top,bottom,left,right
    width = image.shape[1]
    height = image.shape[0]

    if width == 256 and height == 256:
      print ("noaction %s" % imagePath)
      continue


    padding = (width - height) * 0.5
    padding1 = math.ceil(padding)
    padding2 = math.floor(padding)
    if padding < 0:
      print ("skip %s" % imagePath)
      continue
    image = cv2.copyMakeBorder(image,padding1,padding2,0,0,cv2.BORDER_CONSTANT,value=BLACK)
    # 缩放
    image = cv2.resize(image, (256, 256) , interpolation = cv2.INTER_AREA)
    ##img2 = cv2.resize(image, (128, 128) , interpolation = cv2.INTER_AREA)
    ##img3 = cv2.resize(image, (64, 64) , interpolation = cv2.INTER_AREA)
    ##img4 = cv2.resize(image, (32, 32) , interpolation = cv2.INTER_AREA)

    #### 窗口展示图片
    ###cv2.imshow("Image", image)
    ###key = cv2.waitKey(0)

    ##filename = imagePath.split('/')[-1]
    ##cv2.imwrite('train/%s' % filename, image)
    cv2.imwrite(imagePath, image)
    print ("processed %s" % imagePath)


