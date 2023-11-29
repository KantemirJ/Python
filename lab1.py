import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("b.jpeg", 0)
img = cv2.resize(img, (500, 500), interpolation = cv2.INTER_AREA)
#img[200:300,250:350] = (0, 0, 255)

#hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img_cw_180 = cv2.rotate(img, cv2.ROTATE_180)
#ret,thresh = cv2.threshold(gray,150,255,0)
# Change color to RGB (from BGR)
#rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
#pixel_vals = img.reshape((-1,3))
 
# Convert to float type
#pixel_vals = np.float32(pixel_vals)


#the below line of code defines the criteria for the algorithm to stop running, 
#which will happen is 100 iterations are run or the epsilon (which is the required accuracy) 
#becomes 85%
#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
 
# then perform k-means clustering with number of clusters defined as 3
#also random centres are initially choosed for k-means clustering
#k = 6
#retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
 
# convert data into 8-bit values
#centers = np.uint8(centers)
#segmented_data = centers[labels.flatten()]
 
# reshape data into the original image dimensions
#segmented_image = segmented_data.reshape((img.shape))

#cv2.imshow("HSB", hsv)
#cv2.imshow("Rotate180", img_cw_180)
#cv2.imshow("Gray", gray)
#cv2.imshow("Binery", thresh)

# 3x3 Y-direction  kernel
#sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
# 3 X 3 X-direction kernel
#sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)
#filtered_image_y = cv2.filter2D(gray, -1, sobel_y)
#filtered_image_x = cv2.filter2D(gray, -1, sobel_x)
#filtered_image = cv2.Canny(gray, threshold1=20, threshold2=100)

binr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1] 
# define the kernel 
kernel = np.ones((5, 5), np.uint8) 
# invert the image 
invert = cv2.bitwise_not(binr) 
# erode the image 
erosion = cv2.erode(invert, kernel, iterations=1) 
dilation = cv2.dilate(invert, kernel, iterations=1)
opening = cv2.morphologyEx(binr, cv2.MORPH_OPEN, kernel, iterations=1) 
closing = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, kernel, iterations=1)
#morph_gradient = cv2.morphologyEx(invert, cv2.MORPH_GRADIENT, kernel) 
morph_gradient = cv2.morphologyEx(invert, cv2.MORPH_BLACKHAT, kernel) 

#cv2.imshow("y", filtered_image)
#cv2.imshow("X", filtered_image_x)
cv2.imshow("Original", img)
cv2.imshow("Dilation", morph_gradient)

k = cv2.waitKey(0)
#2. image edge detection -variance filter,  range filter. Robert’s filter, Kirsch’s template filter,Prewitt’s gradient filter,  Sobel’s gradient filter, Gradient filter combined with Gaussian filter, and thinned using Canny’s methods	Erosion
#Dilation
#Opening
#Closing
#Morphological Gradient
#Top hat
#Black hat									