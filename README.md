# AI-Enhanced-Engagement-Tracker-for-Young-Learners_Infosys_Internship_Oct2024
Infosys Springboard Internship 5.0 AI-Enhanced Engagement Tracker for Young Learners Project.

# Image processing
Libraries/Frameworks Used:
OpenCV Version-4.10.0.84
Numpy-For array manipulation

Input Image-
![img](https://github.com/user-attachments/assets/30cd01cf-4132-46ea-bb0f-c465d133e0ad)

Developed Logics-
A.Blurimg-
 This applies a Gaussian blur to an image to reduce noise and detail.
Output-![image](https://github.com/user-attachments/assets/8b473979-a4cc-4819-8f65-853dd458cb2b)



 B.Contour-
 This detects contours in a grayscale image using a binary threshold and `cv2.findContours()`. 
 The contours are drawn onto the original image in green.
 Output-[Uploading contourimport cv2

img = cv2.imread('img.png', 0)
_, threshold = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

cv2.imshow('Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#170> white
#
.py…]()

 C.Crop-
 This function extracts a specific region of an image based on pixel range and displays the 
 cropped section.
 Output-[Uploading crop.import cv2

img = cv2.imread('img.png')
cropped = img[50:200, 100:300]

cv2.imshow('Cropped Image', cropped)
cv2.waitKey(0)
cv2.destroyAllWindows()py…]()

 D.Dil_ero-
 This function applies morphological operations, dilation and erosion, to enhance and reduce 
 features in an image, respectively.
 Output-[Uploading dil_erimport cv2
import numpy as np

img = cv2.imread('image1.jpg', 0)
kernel = np.ones((5,5), np.uint8)

dilation = cv2.dilate(img, kernel, iterations=1)
erosion = cv2.erode(img, kernel, iterations=1)

cv2.imshow('Dilated Image', dilation)
cv2.imshow('Eroded Image', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()o.py…]()

 E.Edgedetect-
 This applies the Canny edge detection algorithm to detect edges in a grayscale image.
 Output-[Uploading edgimport cv2

img = cv2.imread('img.png', 0)
edges = cv2.Canny(img, 100, 200)

cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindowsedetect.py…]()

 F.Hist_eq-
 This enhances the contrast of a grayscale image using histogram equalization.
 
 Output-[Uploading himport cv2

img = cv2.imread('image1.jpg', 0)
equalized = cv2.equalizeHist(img)

cv2.imshow('Equalized Image', equalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
ist_eq.py…]()

 G.Hsv-
 This converts a color image from the BGR color space to HSV.
 Output-
 [Uploading hsv.py…]()import cv2

img = cv2.imread('img.png')
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

cv2.imshow('HSV Image', hsv_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#hue refers to dominant color of family
#saturation is purify(0-255)
#value-brightness of color

 H.Imagestack-
 This provides images in horizontal and vertical view in stack.
 
 Output-
 [Uploading imagestimport cv2
import numpy as np

img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

img1 = cv2.resize(img1, (500, 500))
img2 = cv2.resize(img2, (500, 500))

h_concat = np.hstack((img1, img2))
v_concat = np.vstack((img1, img2))

cv2.imshow('Horizontal Concatenation', h_concat)
cv2.imshow('Vertical Concatenation', v_concat)

cv2.waitKey(0)
cv2.destroyAllWindows()ack.py…]()

I.Morphological_transform-
This applies opening and closing morphological operations to a grayscale image to remove noise and fill gaps.

Output-

[Uploadingimport cv2
import numpy as np

img = cv2.imread('image2.jpg', 0)
kernel = np.ones((5,5), np.uint8)

opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Opening (Noise Removal)', opening)
cv2.imshow('Closing (Fill Gaps)', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()
 morphological_transfrom.py…]()

 J.Multivideo-
  This function reads and displays images from a specified folder, printing the dimensions of 
  each image.

  Output-
  [Uploading muimport cv2
import os

folder_path = "C:/Users/tejac/OneDrive/Desktop/teja/"


for filename in os.listdir(folder_path):
   
    file_path = os.path.join(folder_path, filename)

    image = cv2.imread(file_path)
    if image is not None:
       
        cv2.imshow('Image', image)
        cv2.waitKey(0)      
        cv2.destroyAllWindows()
        print(f"{filename} dimensions: {image.shape}")
    else:
        print(f"Failed to load {filename}")

ltivid.py…]()

 K.Read-
  This function reads and displays images from a specified folder, printing the dimensions of 
  each image.

  Output-
  [Uploading reimport cv2

#Read an image
image=cv2.imread('./img.png')

#Display the image using Opencv
cv2.imshow('Image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#To check dimnesions of the image
print(image.shape)ad.py…]()

 L.Read_multiple-
  This function reads and displays images from a specified folder, printing the dimensions of 
  each image.
  
 M.Resize-
 This resizes an image to specified dimensions.

 Output-
 [Uploading resizeimport cv2

img = cv2.imread('img.png')
resized = cv2.resize(img, (300, 300))

cv2.imshow('Resized Image', resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
.py…]()


 N.Rotate-
 This rotates an image by 90 degrees around its center.

 Output-[Uploadinimport cv2

img = cv2.imread('img.png')
(h, w) = img.shape[:2]
center = (w // 2, h // 2)

matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated = cv2.warpAffine(img, matrix, (w, h))

cv2.imshow('Rotated Image', rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()g rotate.py…]()

 
 O.Template-
 This function performs template matching to locate a template image within a larger image.

 Output-
 [Uploading templimport cv2

img = cv2.imread('image1.jpg')
template = cv2.imread('image2.jpg', 0)
result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
top_left = max_loc
h, w = template.shape[:2]
bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)

cv2.imshow('Detected Template', img)
cv2.waitKey(0)
cv2.destroyAllWindows()ate.py…]()



# Video processing
Libraries/Frameworks Used-Opencv
Version-4.10.0.84

Developed Logics-
A.multivid-
This function reads and displays images from a specified folder, printing the dimensions of each image.

B.vid_fps-
This function captures video from the webcam, displays it in real-time, and calculates the FPS.

C.vid_save-
This function captures live video and saves it to a specified output file.

D.vid_stack-
This function reads and resizes two video files, concatenating them horizontally.

E.vid_stream-
This function captures live video from the webcam and displays it in real-time.

Annotaions
Libraries/Frameworks Used-Opencv,LabelImg
Version-4.10.0.84, Version of LabelImg- 1.8.6

Developed Logics-
A.Data_segregate-
This function organizes images and their label files into matched and unmatched directories.

B.Label-
This function draws bounding boxes on images based on annotations in the label files.

C.Label_manipulate-
This function updates class numbers in label files for object detection tasks.


# Face Recognition
Libraries/Frameworks Used-
OpenCV Version-4.10.0.84
face_recognition==1.3.0
dlib==19.24.6
pandas==2.2.3
numpy==2.2.3
imutils== 0.5.4
datetime==5.5

Developed Logics-
A.Face_Recognition-

B.attendence_save
C.Test
D.tools
E.Excel_sc
F.Excel_sc_dt
G.landmark
H.atten_score
I.avg_atten_score
J.vanshika_Face_Recognition
K.vanshika_attendence_save
L.vanshika_Test
M.vanshika_tools
N.vanshika_Excel_sc
O.vanshika_Excel_sc_dt
P.vanshika_landmark
Q.vanshika_atten_score
R.vanshika_avg_atten_score
S.
T.
M.
N.
O.
P.







