import cv2

img = cv2.imread('img.png')
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow('HSV Image', hsv_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#hue refers to dominant color of family
#saturation is purify(0-255)
#value-brightness of color