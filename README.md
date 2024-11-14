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
 
Output-
![image](https://github.com/user-attachments/assets/8b473979-a4cc-4819-8f65-853dd458cb2b)


B.Contour-
 This detects contours in a grayscale image using a binary threshold and `cv2.findContours()`. 
 The contours are drawn onto the original image in green.
 
 Output-
 ![image](https://github.com/user-attachments/assets/1565b2e4-c451-49e3-bcb1-ca31ff160ad6)


 C.Crop-
 This function extracts a specific region of an image based on pixel range and displays the 
 cropped section.
 
 Output-
 ![image](https://github.com/user-attachments/assets/2d34ce4e-62b0-4c27-84de-f9248475b59e)


 D.Dil_ero-
 This function applies morphological operations, dilation and erosion, to enhance and reduce 
 features in an image, respectively.
 
 Output-
 ![image](https://github.com/user-attachments/assets/44c7e4ad-0b01-42e0-8ff6-534c6bd7e37d)

 E.Edgedetect-
 This applies the Canny edge detection algorithm to detect edges in a grayscale image.
 
 Output-
 ![image](https://github.com/user-attachments/assets/93818a96-fcce-4f4c-8ca0-9f8cdf1f59f1)


 F.Hist_eq-
 This enhances the contrast of a grayscale image using histogram equalization.
 
 Output-
 ![image](https://github.com/user-attachments/assets/5f82b0bf-3eb2-4c56-a581-a52561b18fdd)

 G.Hsv-
 This converts a color image from the BGR color space to HSV.
 
 Output-
 ![image](https://github.com/user-attachments/assets/d7982238-e6ec-413a-8e58-47e567ce4501)

 H.Imagestack-
 This provides images in horizontal and vertical view in stack.
 
 Output-
 ![image](https://github.com/user-attachments/assets/836fe531-94df-44dd-878c-a4fbb3a05371)

 
I.Morphological_transform-
This applies opening and closing morphological operations to a grayscale image to remove noise and fill gaps.

Output-
![image](https://github.com/user-attachments/assets/b7cd84b1-fb83-40a9-a5b4-c283e1a9aeed)

 J.Multivideo-
  This function reads and displays images from a specified folder, printing the dimensions of 
  each image.

  Output-
  ![image](https://github.com/user-attachments/assets/ade1e562-6c58-4d07-a64a-ceb58bc787bf)

  K.Read-
  This function reads and displays images from a specified folder, printing the dimensions of 
  each image.

  Output-
  ![image](https://github.com/user-attachments/assets/6d388c5a-2a25-4e94-ae08-97fd04bd2a32)


 L.Read_multiple-
  This function reads and displays images from a specified folder, printing the dimensions of 
  each image.
  
 M.Resize-
 This resizes an image to specified dimensions.

 Output-
 ![image](https://github.com/user-attachments/assets/f82820fe-5e9a-4c94-ae95-c6a0a4cda3e0)
 
 N.Rotate-
 This rotates an image by 90 degrees around its center.

 Output-
 ![image](https://github.com/user-attachments/assets/7706a87d-e087-4a2b-940f-bd8a8e73e258)

 
 O.Template-
 This function performs template matching to locate a template image within a larger image.

 Output-![image](https://github.com/user-attachments/assets/f9934182-a257-434f-829a-d0d0ab45d78c)



# Video processing
Libraries/Frameworks Used-Opencv
Version-4.10.0.84

Developed Logics-
A.multivid-
This function reads and displays images from a specified folder, printing the dimensions of each image.

Output-
![image](https://github.com/user-attachments/assets/2d6ee27e-a09c-46d2-866d-bfe6aa35171c)


B.vid_fps-
This function captures video from the webcam, displays it in real-time, and calculates the FPS.

Output-
![image](https://github.com/user-attachments/assets/fcfe8c94-5751-4f07-a35b-34ece56a1b79)


C.vid_save-
This function captures live video and saves it to a specified output file.

Output-
![image](https://github.com/user-attachments/assets/09fc1186-a65c-4e38-aa40-5848d150ba60)


D.vid_stack-
This function reads and resizes two video files, concatenating them horizontally.

E.vid_stream-
This function captures live video from the webcam and displays it in real-time.

# Annotaions
Libraries/Frameworks Used-Opencv,LabelImg
Version-4.10.0.84, Version of LabelImg- 1.8.6


Developed Logics-
A. Data_segregate-
This function organizes images and their label files into matched and unmatched directories.
Output-
![image](https://github.com/user-attachments/assets/19fea229-e620-4d0b-8d23-8109ecd83586)


B. Label-
This function draws bounding boxes on images based on annotations in the label files.
Input-
![image](https://github.com/user-attachments/assets/ac9c5e15-7448-48e8-880a-197d18193672)
Output-
![image](https://github.com/user-attachments/assets/fd9e69d7-bb44-4cc0-8b51-ea5476ea8af4)


C. Label_manipulate-
This function updates class numbers in label files for object detection tasks.
Input-
![image](https://github.com/user-attachments/assets/84ee6bb5-f481-4d8f-a232-a83c31477811)
Output-
![image](https://github.com/user-attachments/assets/836781e9-425a-4bc5-bac6-e37bfd3e812c)


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
A.vanshika_Face_Recognition-

Input-
![image](https://github.com/user-attachments/assets/99ea3eac-316e-47bd-b601-6a695ff7a512)
Output-
![image](https://github.com/user-attachments/assets/6cadd48e-f982-4811-8190-e2b67c8ddf36)

B.vanshika_attendence_save-

Output-
![image](https://github.com/user-attachments/assets/6cadd48e-f982-4811-8190-e2b67c8ddf36)
![image](https://github.com/user-attachments/assets/002b9d87-aebb-435b-a18d-c62800db3d4e)



C.vanshika_Test-

![image](https://github.com/user-attachments/assets/101663c5-189b-4275-b21d-34549b49a8b1)


E.vanshika_Excel_sc-
![image](https://github.com/user-attachments/assets/9f6b023f-641c-4282-80b5-0ddcf243c8e3)
![image](https://github.com/user-attachments/assets/ce2ee54d-2bf8-48cb-94d4-27d64a3aef70)

F.vanshika_Excel_sc_dt-
![image](https://github.com/user-attachments/assets/3252f514-245f-4a2a-9535-c2eecbd688cd)
![image](https://github.com/user-attachments/assets/ec55e608-5b36-4d9b-ba33-f93651aea447)

G.vanshika_landmark-
![image](https://github.com/user-attachments/assets/c4bbceab-2913-4e21-848f-f3966d124479)
![image](https://github.com/user-attachments/assets/9e6d9ad1-3037-4fd1-9420-3e45ab3e5120)


H.vanshika_atten_score-
![image](https://github.com/user-attachments/assets/e41197c8-3976-4635-a22c-5f911d1b9d09)
![image](https://github.com/user-attachments/assets/002b01c5-7f03-40a2-ac17-eab43e6f400b)

i.vanshika_avg_atten_score-
![image](https://github.com/user-attachments/assets/9d25555b-9d23-450c-a1d8-a0375aa13f38)
![image](https://github.com/user-attachments/assets/48de0339-9ea7-42f6-8d2f-d823bea8b44b)











