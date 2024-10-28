import cv2

#Read an image
image=cv2.imread('./img.png')

#Display the image using Opencv
cv2.imshow('Image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#To check dimnesions of the image
print(image.shape)