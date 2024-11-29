# import cv2 as cv
# import face_recognition
# import dlib
# import pandas as pd
# import numpy as np
# from datetime import datetime
# import os
# import time
# from imutils import face_utils

# # Initialize dlib's face detector and the facial landmark predictor
# p = r"C:\Users\DELL\Desktop\Internship\face recognition\facial-landmarks-recognition\shape_predictor_68_face_landmarks.dat"
# detector = dlib.get_frontal_face_detector()
# landmark_predictor = dlib.shape_predictor(p)

# # Create a directory to save screenshots
# if not os.path.exists("vanshika_screenshots(8)"):
#     os.makedirs("vanshika_screenshots(8)")

# # Load the known image of Vanshika
# known_image = face_recognition.load_image_file("C:\\Users\\DELL\\Desktop\\Internship\\face recognition\\vanshika face recog\\2.jpeg")
# known_faces = face_recognition.face_encodings(known_image, num_jitters=50, model='large')[0]

# # Create a DataFrame to store recognized face information
# columns = ['Name', 'Date', 'Time', 'Screenshot', 'Attentive', 'Attention Score']
# df = pd.DataFrame(columns=columns)

# # Initialize a list to store attention scores for averaging later
# attention_scores = []

# # Launch the live camera or video
# cam = cv.VideoCapture(0)
# if not cam.isOpened():
#     print("Camera not working")
#     exit()

# # Define the maximum thresholds for yaw and pitch beyond which attentiveness is 0
# MAX_YAW_THRESHOLD = 0.8  # Relaxed threshold
# MAX_PITCH_THRESHOLD = 0.8  # Relaxed threshold

# # Initialize the start time
# start_time = time.time()

# # Corrected head pose calculation function
# def get_head_pose(landmarks):
#     image_points = np.array([
#         landmarks[30],  # Nose tip
#         landmarks[8],   # Chin
#         landmarks[36],  # Left eye left corner
#         landmarks[45],  # Right eye right corner
#         landmarks[48],  # Left mouth corner
#         landmarks[54]   # Right mouth corner
#     ], dtype="double")
    
#     model_points = np.array([
#         (0.0, 0.0, 0.0),             # Nose tip
#         (0.0, -330.0, -65.0),        # Chin
#         (-165.0, 170.0, -135.0),     # Left eye left corner
#         (165.0, 170.0, -135.0),      # Right eye right corner
#         (-150.0, -150.0, -125.0),    # Left mouth corner
#         (150.0, -150.0, -125.0)      # Right mouth corner
#     ])
    
#     focal_length = 320
#     center = (160, 120)  # Adjusted for 320x240 resolution
#     camera_matrix = np.array([
#         [focal_length, 0, center[0]],
#         [0, focal_length, center[1]],
#         [0, 0, 1]], dtype="double")
    
#     dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
#     success, rotation_vector, translation_vector = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    
#     if not success:
#         print("Error: Head pose estimation failed. Defaulting to zeros.")
#         rotation_vector = np.zeros((3, 1))  # Default to zero rotation
#         translation_vector = np.zeros((3, 1))  # Default to zero translation
#     else:
#         print(f"Rotation Vector (Yaw, Pitch, Roll): {rotation_vector.ravel()}")
#         print(f"Translation Vector: {translation_vector.ravel()}")
    
#     return rotation_vector, translation_vector

# # Function to calculate attentiveness score between 0 and 1 based on yaw and pitch
# def calculate_attention_score(yaw, pitch):
#     yaw_score = max(0, 1 - abs(yaw[0]) / MAX_YAW_THRESHOLD)
#     pitch_score = max(0, 1 - abs(pitch[0]) / MAX_PITCH_THRESHOLD)
#     # Overall score as the average of yaw and pitch scores
#     attention_score = (yaw_score + pitch_score) / 2
#     return attention_score

# frame_count = 0
# try:
#     while True:
#         frame_count += 1
#         ret, frame = cam.read()

#         if not ret:
#             print("Can't receive frame")
#             break

#         if frame_count % 2 == 0:  # Skip every other frame
#             continue

#         frame = cv.resize(frame, (320, 240))  # Downscale for faster processing
#         gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

#         # Detect faces in the current frame
#         face_locations = face_recognition.face_locations(frame)
#         if not face_locations:
#             continue

#         # Encode faces for comparison
#         face_encodings = face_recognition.face_encodings(frame, face_locations)

#         for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#             # Calculate distance from known face
#             distance = face_recognition.face_distance([known_faces], face_encoding)[0]

#             if distance < 0.6:
#                 # Recognized as Vanshika
#                 now = datetime.now()
#                 date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
#                 name = 'Vanshika'
                
#                 # Landmark detection for attentiveness analysis
#                 face_landmarks = landmark_predictor(gray, dlib.rectangle(left, top, right, bottom))
#                 landmarks = [(p.x, p.y) for p in face_landmarks.parts()]
                
#                 # Draw green circles on each facial landmark
#                 for (x, y) in landmarks:
#                     cv.circle(frame, (x, y), 2, (0, 255, 0), -1)

#                 # Calculate head pose
#                 rotation_vector, translation_vector = get_head_pose(landmarks)
#                 yaw, pitch, roll = rotation_vector
                
#                 # Debugging print statements for verification
#                 print(f"Yaw: {yaw[0]:.2f}, Pitch: {pitch[0]:.2f}, Roll: {roll[0]:.2f}")
                
#                 # Calculate attentiveness score between 0 and 1
#                 attention_score = calculate_attention_score(yaw, pitch)

#                 # Store attention score in the list for later averaging
#                 attention_scores.append(attention_score)

#                 # Set attentive to 'Yes' if attention score >= 0.5, else 'No'
#                 attentive = 'Yes' if attention_score >= 0.5 else 'No'

#                 # Save screenshot if attentive and display score on frame
#                 screenshot_filename = f"vanshika_screenshots(8)/{name}_{now.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
#                 if attentive == 'Yes':
#                     cv.putText(frame, f'Attentive (Score: {attention_score:.2f})', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
#                 else:
#                     cv.putText(frame, f'Not Attentive (Score: {attention_score:.2f})', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

#                 # Log the recognition event with attentiveness score
#                 cv.imwrite(screenshot_filename, frame)  # Save screenshot
#                 new_entry = pd.DataFrame({
#                     'Name': [name],
#                     'Date': [now.strftime("%Y-%m-%d")],
#                     'Time': [now.strftime("%H:%M:%S")],
#                     'Screenshot': [screenshot_filename],
#                     'Attentive': [attentive],
#                     'Attention Score': [attention_score]  # Store the score in the log
#                 })
#                 df = pd.concat([df, new_entry], ignore_index=True)

#                 # Draw rectangle and label around the face
#                 cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0) if attentive == 'Yes' else (0, 0, 255), 2)
#                 cv.putText(frame, name, (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv.LINE_AA)

#         # Display the frame
#         cv.imshow("Video Stream", frame)
        
#         # Check if 30 seconds have passed and save the DataFrame to Excel
#         if time.time() - start_time >= 30:
#             # Save to Excel every 30 seconds
#             if not df.empty:
#                 df.to_excel('vanshika_attendance_with_attention_and_average(8).xlsx', index=False)
#                 print("Attendance saved to 'vanshika_attendance_with_attention_and_average(8).xlsx'.")
#             start_time = time.time()  # Reset the timer

#         if cv.waitKey(1) == ord('q'):
#             break

# except Exception as e:
#     print(f"An error occurred: {e}")

# finally:
#     if not df.empty:
#         # Calculate the average attention score at the end of the stream
#         average_attention_score = np.mean(attention_scores) if attention_scores else 0
#         df['Average Attention Score'] = average_attention_score
#         df.to_excel('vanshika_attendance_with_attention_and_average(8).xlsx', index=False)
#         print(f"Final average attention score: {average_attention_score:.2f}")
#         print("Attendance log saved to 'vanshika_attendance_with_attention_and_average(8).xlsx'.")

#     cam.release()
#     cv.destroyAllWindows()
 
import cv2 as cv
import face_recognition
import dlib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import time

# Initialize dlib's face detector and the facial landmark predictor
p = r"C:\Users\DELL\Desktop\Internship\face recognition\facial-landmarks-recognition\shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(p)

# Create a directory to save screenshots
if not os.path.exists("vanshika_screenshots"):
    os.makedirs("vanshika_screenshots")

# Load the known image of Vanshika
known_image = face_recognition.load_image_file("C:\\Users\\DELL\\Desktop\\Internship\\face recognition\\vanshika face recog\\2.jpeg")
known_faces = face_recognition.face_encodings(known_image, num_jitters=50, model='large')[0]

# Create a DataFrame to store recognized face information
columns = ['Name', 'Date', 'Time', 'Screenshot', 'Attentive', 'Attention Score', 'Total Attentive Time (s)']
df = pd.DataFrame(columns=columns)

# Initialize variables for tracking attention
attention_scores = []
time_attentive = 0  # Total attentive time
attention_start_time = None  # Start time for attentive duration tracking

# Launch the live camera
cam = cv.VideoCapture(0)
if not cam.isOpened():
    print("Camera not working")
    exit()

# Define thresholds for yaw and pitch beyond which attentiveness is 0
MAX_YAW_THRESHOLD = 0.8
MAX_PITCH_THRESHOLD = 0.8

# Function to calculate attentiveness score based on yaw and pitch
def calculate_attention_score(yaw, pitch):
    yaw_score = max(0, 1 - abs(yaw[0]) / MAX_YAW_THRESHOLD)
    pitch_score = max(0, 1 - abs(pitch[0]) / MAX_PITCH_THRESHOLD)
    return (yaw_score + pitch_score) / 2

frame_count = 0
start_time = time.time()

try:
    while True:
        frame_count += 1
        ret, frame = cam.read()

        if not ret:
            print("Can't receive frame")
            break

        if frame_count % 2 == 0:  # Skip every other frame for efficiency
            continue

        frame = cv.resize(frame, (640, 480))  # Resize frame
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect faces
        face_locations = face_recognition.face_locations(frame)
        if not face_locations:
            continue

        face_encodings = face_recognition.face_encodings(frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Calculate distance from known face
            distance = face_recognition.face_distance([known_faces], face_encoding)[0]

            if distance < 0.6:  # Recognized as Vanshika
                name = "Vanshika"
                now = datetime.now()
                
                # Dynamic attentiveness calculation
                yaw = np.random.uniform(-0.9, 0.9)  # Simulate random yaw for testing
                pitch = np.random.uniform(-0.9, 0.9)  # Simulate random pitch for testing
                attention_score = calculate_attention_score([yaw], [pitch])

                if attention_score >= 0.5:
                    attentive = 'Attentive'
                    color = (0, 255, 0)  # Green for attentive
                else:
                    attentive = 'Not Attentive'
                    color = (0, 0, 255)  # Red for not attentive

                # Update attentive duration tracking
                if attentive == 'Attentive':
                    if attention_start_time is None:
                        attention_start_time = time.time()
                else:
                    if attention_start_time is not None:
                        time_attentive += time.time() - attention_start_time
                        attention_start_time = None

                # Display on the frame
                cv.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv.putText(frame, f"{name}: {attentive} (Score: {attention_score:.2f})", 
                           (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv.LINE_AA)

                # Save screenshot if attentive
                screenshot_filename = f"vanshika_screenshots/{name}_{now.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                cv.imwrite(screenshot_filename, frame)

                # Log to DataFrame
                new_entry = pd.DataFrame({
                    'Name': [name],
                    'Date': [now.strftime("%Y-%m-%d")],
                    'Time': [now.strftime("%H:%M:%S")],
                    'Screenshot': [screenshot_filename],
                    'Attentive': [attentive],
                    'Attention Score': [attention_score],
                    'Total Attentive Time (s)': [time_attentive]
                })
                df = pd.concat([df, new_entry], ignore_index=True)

        # Display the frame
        cv.imshow("Video Stream", frame)

        # Save DataFrame to Excel every 30 seconds
        if time.time() - start_time >= 30:
            if not df.empty:
                df.to_excel('vanshika_attendance_with_attention.xlsx', index=False)
                print("Attendance saved to 'vanshika_attendance_with_attention.xlsx'.")
            start_time = time.time()

        if cv.waitKey(1) == ord('q'):  # Quit on 'q' key press
            break

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    if not df.empty:
        # Save final DataFrame
        df.to_excel('vanshika_attendance_with_attention.xlsx', index=False)
        print("Final attendance log saved.")
    cam.release()
    cv.destroyAllWindows()



