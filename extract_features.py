'''import cv2
import numpy as np
import mediapipe as mp
import math

mp_pose = mp.solutions.pose

def euclidean_distance(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

def extract_features(image_path):
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize MediaPipe
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            print("No pose landmarks detected.")
            return None

        landmarks = results.pose_landmarks.landmark

        # Get all required landmarks
        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        l_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

        # Normalize distances by shoulder width (chest)
        chest = euclidean_distance(l_shoulder, r_shoulder)

        if chest == 0:
            print("Invalid normalization base (chest width = 0)")
            return None

        features = []

        # Extract normalized measurements
        neck = euclidean_distance(l_shoulder, r_shoulder) * 0.75  # Estimation
        abdomen = euclidean_distance(l_hip, r_hip)
        hip = abdomen
        thigh = euclidean_distance(l_hip, l_knee)
        knee = euclidean_distance(l_knee, l_ankle)
        ankle = 0.05  # dummy value, hard to estimate directly
        biceps = euclidean_distance(l_shoulder, l_elbow)
        forearm = euclidean_distance(l_elbow, l_wrist)
        wrist = 0.03  # dummy value, or you can estimate width from hand landmarks

        # Normalize all except dummy values
        normalized_features = [
            neck / chest,
            chest / chest,
            abdomen / chest,
            hip / chest,
            thigh / chest,
            knee / chest,
            ankle,
            biceps / chest,
            forearm / chest,
            wrist
        ]
        

        return normalized_features

print(extract_features())

'''
import cv2
import numpy as np
import mediapipe as mp
import math
import joblib

mp_pose = mp.solutions.pose

def euclidean_distance(a, b):
    return math.sqrt((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2)

def extract_features(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)

        if not results.pose_landmarks:
            print("No pose landmarks detected.")
            return None

        landmarks = results.pose_landmarks.landmark

        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        l_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
        r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
        l_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
        r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
        r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]



        chest = euclidean_distance(l_shoulder, r_shoulder)

        for i, lm in enumerate(landmarks):
            print(f"{i}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}")


        if chest == 0:
            print("Invalid normalization base (chest width = 0)")
            return None

        neck = chest * 0.75  # estimate
        abdomen = euclidean_distance(l_hip, r_hip)
        hip = abdomen
        thigh = euclidean_distance(l_hip, l_knee)
        biceps = euclidean_distance(l_shoulder, l_elbow)
        forearm = euclidean_distance(l_elbow, l_wrist)
       

        normalized_features = [
            neck / chest,
            chest / chest,
            abdomen / chest,
            hip / chest,
            thigh / chest,
            biceps / chest,
            forearm / chest,
           
        ]

        print("🔍 MediaPipe normalized features:")
        for name, val in zip(
            ["Neck", "Chest", "Abdomen", "Hip", "Thigh", "Knee", "Ankle", "Biceps", "Forearm", "Wrist"],
            normalized_features
        ):
            print(f"{name:<10}: {val:.4f}")

        return normalized_features

# Run prediction when script is executed directly
if __name__ == "__main__":
    # Load your trained model
    model = joblib.load("models/bodyfat_model.pkl")

    # Get user input
    age = int(input("Enter Age: "))
    weight = float(input("Enter Weight (kg): "))
    height = float(input("Enter Height (cm): "))

    image_path = "sample.jpg"  # You said this file is in the same folder

    # Extract features
    features = extract_features(image_path)

    if features:
        input_vector = [age, weight, height] + features
        prediction = model.predict([input_vector])
        print(f"🧠 Predicted Body Fat %: {prediction[0]:.2f}")
    else:
        print("❌ Could not process the image.")
