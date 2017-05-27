#!/usr/bin/python
# -*- coding: utf-8 -*-
# import the necessary packages
from imutils import paths
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
    print imagePath, image.shape

    # 灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)


    width = image.shape[1]
    height = image.shape[0]
    # REPLICATE 当前边界线像素复制延伸
    replicate = cv2.copyMakeBorder(image,100,100,0,0,cv2.BORDER_REPLICATE)
    # REFLECT 当前边界线做镜像
    reflect = cv2.copyMakeBorder(image,100,100,0,0,cv2.BORDER_REFLECT)
    # REFLECT101 当前边界线做镜像
    reflect101 = cv2.copyMakeBorder(image,100,100,0,0,cv2.BORDER_REFLECT_101)
    # wrap 平铺延展，展示相邻图片边缘
    wrap = cv2.copyMakeBorder(image,100,100,0,0,cv2.BORDER_WRAP)

    # CONSTANT 用颜色填充
    BLUE = [255,0,0]
    WHITE = [255,255,255]
    BLACK = [0,0,0]
    # top,bottom,left,right
    constant0= cv2.copyMakeBorder(image,100,100,0,0,cv2.BORDER_CONSTANT,value=BLACK)
    constant1= cv2.copyMakeBorder(image,100,100,0,0,cv2.BORDER_CONSTANT,value=WHITE)
    constant2= cv2.copyMakeBorder(image,100,100,0,0,cv2.BORDER_CONSTANT,value=BLUE)

    # 矩形截取
    # image = image[100:300, 50:500]

    # 缩放
    img0 = cv2.resize(constant0, (512, 512) , interpolation = cv2.INTER_AREA)
    img1 = cv2.resize(constant0, (256, 256) , interpolation = cv2.INTER_AREA)
    img2 = cv2.resize(constant0, (128, 128) , interpolation = cv2.INTER_AREA)
    img3 = cv2.resize(constant0, (64, 64) , interpolation = cv2.INTER_AREA)
    img4 = cv2.resize(constant0, (32, 32) , interpolation = cv2.INTER_AREA)

    # 窗口展示图片
    cv2.imshow("Image0", img0)
    cv2.imshow("Image1", img1)
    cv2.imshow("Image2", img2)
    cv2.imshow("Image3", img3)
    cv2.imshow("Image4", img4)
    key = cv2.waitKey(0)
    ###cv2.imshow("Image", constant1)
    ###key = cv2.waitKey(0)
    ###cv2.imshow("Image", constant2)
    ###key = cv2.waitKey(0)
    ###cv2.imshow("Image", wrap)
    ###key = cv2.waitKey(0)
    ###cv2.imshow("Image", reflect101)
    ###key = cv2.waitKey(0)
    ###cv2.imshow("Image", replicate)
    ###key = cv2.waitKey(0)
    ###cv2.imshow("Image", reflect)
    ###key = cv2.waitKey(0)

