# File: main_pro.py
# Author: Tushar R. Islampure
# Purpose: Archery posture evaluation prototype — process videos, compute metrics, create reports
# Date: 2025-08-08

import os
import cv2
import math
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import imageio
import pandas as pd

# ---- Config / paths ----
INPUT_DIR = "input_videos"
OUTPUT_DIR = "outputs_videos"
FRAMES_DIR = "output_framesss"
PLOTS_DIR = "plots_3dss"
REPORTS_DIR = "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ---- MediaPipe setup ----
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ---- Helpers ----
def angle_between(a, b, c):
    """Return angle ABC (degrees). Points are (x,y)."""
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b
    bc = c - b
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm == 0:
        return 0.0
    cosang = np.dot(ba, bc) / norm
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def landmark_to_xy(landmark, frame_w, frame_h):
    return (landmark.x * frame_w, landmark.y * frame_h, landmark.z * frame_w)

def ema(old, new, alpha=0.6):
    """Simple exponential moving average for smoothing landmarks."""
    if old is None:
        return new
    return alpha * np.array(new) + (1 - alpha) * np.array(old)

def compute_center_of_mass(landmarks_xy):
    """Approximate COM using major joints (hips, shoulders)."""
    keys = ['LEFT_HIP','RIGHT_HIP','LEFT_SHOULDER','RIGHT_SHOULDER']
    pts = [landmarks_xy[k] for k in keys if k in landmarks_xy and landmarks_xy[k] is not None]
    if not pts:
        return None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return (sum(xs)/len(xs), sum(ys)/len(ys))

def save_3d_plot_xyz(xs, ys, zs, out_path, title="3D Pose"):
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs)
    # draw simple skeleton connections (selected)
    # note: indices are not used here; user-friendly stick visualization:
    ax.set_title(title)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)

# ---- Core processing per video ----
def process_video(path, pose_model):
    vid_name = os.path.splitext(os.path.basename(path))[0]
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path = os.path.join(OUTPUT_DIR, f"{vid_name}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    frame_idx = 0
    # keep smoothed landmarks across frames
    smoothed = None
    prev_wrist = None
    wrist_speeds = []
    all_feedback = []
    save_plot_frames = []

    # we'll store time-series metrics for the report
    metrics_time = {
        "frame": [],
        "elbow_left_deg": [],
        "elbow_right_deg": [],
        "shoulder_left_deg": [],
        "shoulder_right_deg": [],
        "stance_width_px": [],
        "wrist_speed": []
    }

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc=f"Processing {vid_name}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_model.process(image_rgb)

        landmarks_xy = {}
        landmarks_xyz = {}
        if results.pose_landmarks:
            for lm_name in mp_pose.PoseLandmark:
                lm = results.pose_landmarks.landmark[lm_name.value]
                x, y, z = landmark_to_xy(lm, w, h)
                landmarks_xy[lm_name.name] = (x, y)
                landmarks_xyz[lm_name.name] = (x, y, z)

            # Smooth (EMA) the full landmark dict
            if smoothed is None:
                smoothed = landmarks_xy
            else:
                smoothed = {k: tuple(ema(smoothed.get(k, None), landmarks_xy.get(k))) for k in landmarks_xy.keys()}

            # compute angles
            # elbow left: shoulder-left, elbow-left, wrist-left
            el = elbow_angle_l = None
            try:
                a = smoothed['LEFT_SHOULDER']; b = smoothed['LEFT_ELBOW']; c = smoothed['LEFT_WRIST']
                elbow_angle_l = angle_between(a, b, c)
            except Exception:
                elbow_angle_l = None

            try:
                a = smoothed['RIGHT_SHOULDER']; b = smoothed['RIGHT_ELBOW']; c = smoothed['RIGHT_WRIST']
                elbow_angle_r = angle_between(a, b, c)
            except Exception:
                elbow_angle_r = None

            # shoulder angles (hip-shoulder-elbow)
            try:
                shoulder_left = angle_between(smoothed['LEFT_HIP'], smoothed['LEFT_SHOULDER'], smoothed['LEFT_ELBOW'])
            except Exception:
                shoulder_left = None
            try:
                shoulder_right = angle_between(smoothed['RIGHT_HIP'], smoothed['RIGHT_SHOULDER'], smoothed['RIGHT_ELBOW'])
            except Exception:
                shoulder_right = None

            # stance width (distance between ankles)
            try:
                la = smoothed['LEFT_ANKLE']; ra = smoothed['RIGHT_ANKLE']
                stance_width = math.hypot(la[0]-ra[0], la[1]-ra[1])
            except Exception:
                stance_width = None

            # wrist speed estimate (pixel-per-frame)
            wrist = smoothed.get('LEFT_WRIST') or smoothed.get('RIGHT_WRIST')
            if wrist is not None and prev_wrist is not None:
                ws = math.hypot(wrist[0]-prev_wrist[0], wrist[1]-prev_wrist[1])
            else:
                ws = 0.0
            prev_wrist = wrist
            wrist_speeds.append(ws)

            # compute center of mass
            com = compute_center_of_mass(smoothed)

            # write metrics into time-series
            metrics_time["frame"].append(frame_idx)
            metrics_time["elbow_left_deg"].append(elbow_angle_l or np.nan)
            metrics_time["elbow_right_deg"].append(elbow_angle_r or np.nan)
            metrics_time["shoulder_left_deg"].append(shoulder_left or np.nan)
            metrics_time["shoulder_right_deg"].append(shoulder_right or np.nan)
            metrics_time["stance_width_px"].append(stance_width or np.nan)
            metrics_time["wrist_speed"].append(ws)

            # basic feedback rules (human tone)
            tips = []
            if elbow_angle_l is not None:
                if elbow_angle_l < 140:
                    tips.append("Left elbow seems under-drawn - consider pulling slightly more.")
                else:
                    tips.append("Left elbow draw looks full and steady.")
            if stance_width is not None:
                # compare stance width to frame width as proxy
                if stance_width < w * 0.15:
                    tips.append("Stance might be narrow - widen feet slightly for stability.")
                elif stance_width > w * 0.4:
                    tips.append("Stance seems wide - ensure balance is maintained.")
                else:
                    tips.append("Stance width looks reasonable.")
            if com is not None:
                # check if COM is inside base of stance (approx between ankles)
                try:
                    la = smoothed['LEFT_ANKLE']; ra = smoothed['RIGHT_ANKLE']
                    minx = min(la[0], ra[0]); maxx = max(la[0], ra[0])
                    if com[0] < minx or com[0] > maxx:
                        tips.append("Center of mass is off the base - try centering weight between feet.")
                except Exception:
                    pass

            # detect potential release by wrist speed spike (simple method)
            release_flag = False
            if len(wrist_speeds) > 6:
                # compare recent average vs previous
                recent = np.mean(wrist_speeds[-3:])
                prev = np.mean(wrist_speeds[-6:-3])
                if prev < 1e-3:
                    ratio = 0
                else:
                    ratio = recent / (prev + 1e-6)
                # heuristic threshold — tuned experimentally
                if ratio > 3.5 and recent > 5.0:
                    release_flag = True
                    tips.append("Release detected: look at follow-through and hand reaction.")
            # collect fingerprints for later
            if frame_idx % int(fps*1.5) == 0:
                # save 3D plot image occasionally
                xs = [landmarks_xyz[k][0] for k in landmarks_xyz.keys()]
                ys = [landmarks_xyz[k][1] for k in landmarks_xyz.keys()]
                zs = [landmarks_xyz[k][2] for k in landmarks_xyz.keys()]
                plot_name = os.path.join(PLOTS_DIR, f"{vid_name}_f{frame_idx}_3d.png")
                save_3d_plot_xyz(xs, ys, zs, plot_name, title=f"{vid_name} frame {frame_idx}")
                save_plot_frames.append(plot_name)

            # draw overlay on frame (quick, readable)
            overlay_text = [
                f"Frame: {frame_idx}",
                f"Left Elbow: {int(elbow_angle_l) if elbow_angle_l else 'NA'} deg",
                f"Stance: {int(stance_width) if stance_width else 'NA'} px",
                f"Wrist spd: {ws:.1f}"
            ]
            y0 = 30
            for i, t in enumerate(overlay_text):
                cv2.putText(frame, t, (10, y0 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

            # draw human skeleton
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2))

            # append tips for summary
            all_feedback.extend(tips)

        # write the frame to output video
        writer.write(frame)

        # save periodic frames for the report
        if frame_idx % int(fps*2) == 0:
            fname = os.path.join(FRAMES_DIR, f"{vid_name}_frame{frame_idx}.jpg")
            cv2.imwrite(fname, frame)

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()

    # Post processing: analyze wrist speed time-series for smoothness and consistency
    df = pd.DataFrame(metrics_time)
    smoothness = float(df["wrist_speed"].replace([np.inf, -np.inf], np.nan).dropna().var())
    avg_elbow = float(np.nanmean(df["elbow_left_deg"]))
    avg_stance = float(np.nanmean(df["stance_width_px"]))

    # Create human-friendly summary
    summary_lines = []
    summary_lines.append(f"Video: {vid_name}")
    summary_lines.append(f"Avg draw elbow (left): {avg_elbow:.1f} deg (ideal ~ 140-170 deg)")
    summary_lines.append(f"Stance width (avg): {avg_stance:.1f} px")
    if smoothness < 1.0:
        summary_lines.append("Draw movement is smooth (low wrist variance).")
    else:
        summary_lines.append("Draw/release has variability — practice a consistent draw to reduce noise.")
    # Add repeated tips deduplicated
    uniq_tips = list(dict.fromkeys(all_feedback))[:12]
    summary_lines.extend(["Observations:"] + ["- " + t for t in uniq_tips])

    # Save a short CSV of metrics for reference
    csvpath = os.path.join(REPORTS_DIR, f"{vid_name}_metrics.csv")
    df.to_csv(csvpath, index=False)

    # Save a compact PDF report
    pdfpath = os.path.join(REPORTS_DIR, f"{vid_name}_report.pdf")
    create_pdf_report(vid_name, pdfpath, summary_lines, save_plot_frames, FRAMES_DIR)

    return {
        "video": vid_name,
        "out_video": out_video_path,
        "pdf": pdfpath,
        "metrics_csv": csvpath
    }

# ---- PDF creation (simple and clean) ----
def create_pdf_report(vid_name, out_pdf_path, summary_lines, images_list, frames_dir):
    c = canvas.Canvas(out_pdf_path, pagesize=A4)
    w, h = A4
    margin = 40
    y = h - margin
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, f"Archery Evaluation Report — {vid_name}")
    c.setFont("Helvetica", 10)
    y -= 30
    for line in summary_lines:
        if y < margin + 80:
            c.showPage()
            y = h - margin
        c.drawString(margin, y, line)
        y -= 14

    # place 3 sample frames and 3 plots if available
    inserted = 0
    # prefer frames from frames_dir that match vid_name
    frame_files = sorted([os.path.join(frames_dir,f) for f in os.listdir(frames_dir) if f.startswith(vid_name)])[:3]
    plot_files = images_list[:3]
    y -= 10
    for img_path in frame_files + plot_files:
        if inserted % 2 == 0 and y < 200:
            c.showPage()
            y = h - margin
        try:
            img = ImageReader(img_path)
            iw, ih = img.getSize()
            aspect = ih/iw
            draw_w = (w - 2*margin) / 2
            draw_h = draw_w * aspect
            c.drawImage(img, margin + (inserted%2)*(draw_w+10), y - draw_h, width=draw_w, height=draw_h)
            if inserted % 2 == 1:
                y -= (draw_h + 20)
            inserted += 1
        except Exception:
            # skip broken images
            continue

    c.showPage()
    c.save()

# ---- Entrypoint ----
def main():
    video_files = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.lower().endswith(".mp4")]
    if not video_files:
        print("Put your .mp4 files into the 'input_videos' folder and re-run.")
        return

    with mp_pose.Pose(static_image_mode=False, model_complexity=1,
                      enable_segmentation=False, min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:
        results = []
        for vid in video_files:
            try:
                out = process_video(vid, pose)
                results.append(out)
                print(f"Done: {out['video']}, annotated video: {out['out_video']}, report: {out['pdf']}")
            except Exception as e:
                print(f"Error processing {vid}: {e}")

if __name__ == "__main__":
    main()
