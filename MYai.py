import tensorflow as tf
tf.__version__ = tf.version.VERSION 

import cv2
import json
from deepface import DeepFace

# Load student database
with open("students.json", "r") as f:
    student_data = json.load(f)

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Save frame temporarily for DeepFace
        cv2.imwrite("temp_frame.jpg", frame)

        # Detect and recognize face
        result = DeepFace.find(img_path="temp_frame.jpg", db_path="student_faces/", enforce_detection=False)

        if len(result[0]) > 0:
            # Extract student name from matched face file
            student_name = result[0]["identity"][0].split('/')[-1].split('.')[0]

            if student_name in student_data:
                marks = student_data[student_name]["marks"]
                feedback = "Excellent!" if marks > 90 else "Good job!" if marks > 75 else "Needs Improvement"

                # Display student info
                cv2.putText(frame, f"Name: {student_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Marks: {marks}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Feedback: {feedback}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    except Exception as e:
        print(f"Error: {e}")

    # Show frame
    cv2.imshow("Face Recognition", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

