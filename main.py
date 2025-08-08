# File: main_pro.py
# Author: Tushar R. Islampure
# Purpose: Archery posture evaluation prototype — process videos, compute metrics, create reports
# Date: 2025-08-08
import cv2
import mediapipe as mp
import numpy as np
import os
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

input_folder = "input_videos"
output_folder = "output_frames"
plot_folder = "plots_3d"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)

def calculate_angle(a, b, c):
    """Calculate angle between 3 points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    return np.abs(np.degrees(radians))

def analyze_posture(landmarks, video_name):
    """Analyze archery posture and return feedback string."""
    feedback = []

    # Get landmark
    l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
               landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

    # Elbow angle
    elbow_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
    if elbow_angle < 140:
        feedback.append("Elbow not fully drawn — straighten the draw arm.")
    else:
        feedback.append("Draw elbow angle looks good.")

    # Shoulder alignment
    r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    shoulder_diff = abs(l_shoulder[1] - r_shoulder[1])
    if shoulder_diff > 0.05:
        feedback.append("Shoulders are not level — adjust shoulder posture.")
    else:
        feedback.append("Shoulder alignment is good.")

    return feedback


def save_3d_plot(landmarks, frame_idx, video_name):
    """Save 3D stick figure using pose landmarks."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    zs = [lm.z for lm in landmarks]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, color='blue')
    ax.set_title(f"3D Pose Frame {frame_idx}")
    plt.savefig(f"{plot_folder}/{video_name}_frame{frame_idx}_3d.png")
    plt.close()

def process_video(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    feedback_written = False
    all_feedback = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            feedback = analyze_posture(landmarks, video_name)
            all_feedback.extend(feedback)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            cv2.putText(frame, f"{feedback[0]}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

            if frame_idx % 50 == 0:
                save_3d_plot(landmarks, frame_idx, video_name)

        # Save one frame every 30
        if frame_idx % 30 == 0:
            cv2.imwrite(f"{output_folder}/{video_name}_frame{frame_idx}.jpg", frame)

        frame_idx += 1

    cap.release()

    with open("feedback_report.txt", "a") as f:
        f.write(f"\n\n===== {video_name} =====\n")
        for line in set(all_feedback):
            f.write(f"- {line}\n")

# 🔁 Loop through all videos
for filename in os.listdir(input_folder):
    if filename.endswith(".mp4"):
        process_video(os.path.join(input_folder, filename))

print("✅ Done! Output in 'output_frames', 'plots_3d', and 'feedback_report.txt'")
