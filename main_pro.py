#File: main_pro.py
# Author: Tushar R. Islampure 
# Purpose: Archery evaluation prototype - Process the 5 supplied archery videos, auto-detect release frames, compute metrics,
# train a simple posture classifier, and create a polished PDF report per-video.
# Notes: I ran quick visual tuning on thresholds for the supplied videos — tune further if needed.
# Date: 2025-08-08

import os, cv2, math, numpy as np, pandas as pd, matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
import mediapipe as mp
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib, seaborn as sns, warnings, json
import subprocess 
warnings.filterwarnings("ignore")

# ------------------ PATHS & CONFIG ------------------
INPUT_DIR = "input_videos"
OUTPUT_VID_DIR = "outputs_videos"
FRAMES_DIR = "output_frames"
PLOTS_DIR = "plots_3d"
REPORTS_DIR = "reports"
METRICS_DIR = "metrics_all"
os.makedirs(OUTPUT_VID_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Processing params
SMOOTH_ALPHA = 0.6  # EMA smoothing
RELEASE_SPIKE_RATIO = 3.5  # heuristic for release detection
RELEASE_RECENT_WINDOW = 3   # frames
CLASSIFIER_WIN_FRAMES = 25  # ~1s @25fps

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ------------------ HELPERS ------------------
def to_xy(lm, w, h):
    return (lm.x * w, lm.y * h, lm.z * w)

def angle_deg(a,b,c):
    a=np.array(a); b=np.array(b); c=np.array(c)
    ba=a-b; bc=c-b
    denom = np.linalg.norm(ba)*np.linalg.norm(bc)
    if denom==0: return np.nan
    cosv = np.dot(ba,bc)/denom
    cosv = np.clip(cosv, -1, 1)
    return float(np.degrees(np.arccos(cosv)))

def ema(prev, new, alpha=SMOOTH_ALPHA):
    if prev is None:
        return new
    return alpha*np.array(new) + (1-alpha)*np.array(prev)

# Cross-correlation-ish release detector
def detect_release_by_corr(wrist_speed, elbow_vel, fs=25):
    ws = np.array(wrist_speed)
    ev = np.array(elbow_vel)
    if len(ws) < 8: return []
    from scipy.signal import savgol_filter
    k = min(11, len(ws)//2*2+1)
    try:
        ws_s = savgol_filter(ws, k, 3)
        ev_s = savgol_filter(ev, k, 3)
    except Exception:
        ws_s = ws; ev_s = ev
    dev = np.abs(np.gradient(ev_s))
    measure = ws_s * dev
    thr = np.nanmean(measure) + 2.2*np.nanstd(measure)
    idx = np.where(measure > thr)[0].tolist()
    # collapse near indices
    events=[]
    if idx:
        cur_group=[idx[0]]
        for i in idx[1:]:
            if i - cur_group[-1] <= int(0.2*fs):
                cur_group.append(i)
            else:
                events.append(int(np.median(cur_group))); cur_group=[i]
        if cur_group: events.append(int(np.median(cur_group)))
    return events

# ------------------ PROCESS VIDEO & SAVE METRICS ------------------
def process_video(video_path):
    vid_name = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Save to a temporary raw-encoded file first
    temp_raw_path = os.path.join(OUTPUT_VID_DIR, vid_name + "_temp.mp4")
    final_path = os.path.join(OUTPUT_VID_DIR, vid_name + "_annotated.mp4")

    writer = cv2.VideoWriter(temp_raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))

    frame_idx = 0
    smoothed = None
    prev_wrist = None
    prev_elbow_angle = None
    wrist_speeds = []
    metrics = {"frame":[], "time":[], "elbow_l":[], "elbow_r":[], "shoulder_l":[], "shoulder_r":[], "stance":[], "wrist_speed":[]}

    with mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        pbar = tqdm(total=total, desc=f"Process {vid_name}")
        while True:
            ok, frame = cap.read()
            if not ok: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            landmarks_xy = {}
            if results.pose_landmarks:
                for lm_enum in mp_pose.PoseLandmark:
                    lm = results.pose_landmarks.landmark[lm_enum.value]
                    x,y,z = to_xy(lm,w,h)
                    landmarks_xy[lm_enum.name] = (x,y)
                if smoothed is None:
                    smoothed = landmarks_xy
                else:
                    smoothed = {k: tuple(ema(smoothed.get(k, None), landmarks_xy.get(k))) for k in landmarks_xy.keys()}
                try:
                    elbow_l = angle_deg(smoothed['LEFT_SHOULDER'], smoothed['LEFT_ELBOW'], smoothed['LEFT_WRIST'])
                except: elbow_l = np.nan
                try:
                    elbow_r = angle_deg(smoothed['RIGHT_SHOULDER'], smoothed['RIGHT_ELBOW'], smoothed['RIGHT_WRIST'])
                except: elbow_r = np.nan
                try:
                    shoulder_l = angle_deg(smoothed['LEFT_HIP'], smoothed['LEFT_SHOULDER'], smoothed['LEFT_ELBOW'])
                except: shoulder_l = np.nan
                try:
                    shoulder_r = angle_deg(smoothed['RIGHT_HIP'], smoothed['RIGHT_SHOULDER'], smoothed['RIGHT_ELBOW'])
                except: shoulder_r = np.nan
                try:
                    la = smoothed['LEFT_ANKLE']; ra = smoothed['RIGHT_ANKLE']
                    stance = math.hypot(la[0]-ra[0], la[1]-ra[1])
                except: stance = np.nan
                wrist = smoothed.get('LEFT_WRIST') or smoothed.get('RIGHT_WRIST')
                if wrist is not None and prev_wrist is not None:
                    ws = math.hypot(wrist[0]-prev_wrist[0], wrist[1]-prev_wrist[1])
                else:
                    ws = 0.0
                prev_wrist = wrist
                if not np.isnan(elbow_l) and prev_elbow_angle is not None:
                    elbow_vel = elbow_l - prev_elbow_angle
                else:
                    elbow_vel = 0.0
                prev_elbow_angle = elbow_l if not np.isnan(elbow_l) else prev_elbow_angle
                wrist_speeds.append(ws)
                metrics["frame"].append(frame_idx)
                metrics["time"].append(frame_idx / fps)
                metrics["elbow_l"].append(elbow_l)
                metrics["elbow_r"].append(elbow_r)
                metrics["shoulder_l"].append(shoulder_l)
                metrics["shoulder_r"].append(shoulder_r)
                metrics["stance"].append(stance)
                metrics["wrist_speed"].append(ws)
                overlay = f"F:{frame_idx} EL:{int(elbow_l) if not np.isnan(elbow_l) else 'NA'} WS:{ws:.1f}"
                cv2.putText(frame, overlay, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          mp_drawing.DrawingSpec((0,255,0),2,2), mp_drawing.DrawingSpec((0,0,255),2,2))
            writer.write(frame)
            if frame_idx % int(fps*2) == 0:
                cv2.imwrite(os.path.join(FRAMES_DIR, f"{vid_name}_frame{frame_idx}.jpg"), frame)
            frame_idx += 1
            pbar.update(1)
        pbar.close()

    writer.release()
    cap.release()

    # 🔹 Convert temp video to guaranteed H.264 + yuv420p for browser compatibility
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-i", temp_raw_path,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            final_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        os.remove(temp_raw_path)
    else:
        # If ffmpeg not found, keep raw temp file but rename
        os.rename(temp_raw_path, final_path)

    # save CSV
    df = pd.DataFrame(metrics)
    csv_path = os.path.join(METRICS_DIR, f"{vid_name}_metrics.csv")
    df.to_csv(csv_path, index=False)
    # detect release candidates
    elbow_grad = np.gradient(df['elbow_l'].fillna(0).values) if len(df)>1 else np.array([0])
    candidates = detect_release_by_corr(df['wrist_speed'].fillna(0).values, elbow_grad, fs=int(fps))
    # save chart
    fig, ax = plt.subplots(3,1,figsize=(9,9))
    ax[0].plot(df['time'], df['elbow_l'], label='Elbow L'); ax[0].legend()
    ax[1].plot(df['time'], df['wrist_speed'], label='Wrist speed'); ax[1].legend()
    ax[2].plot(df['time'], np.gradient(df['elbow_l'].fillna(0)), label='Elbow vel'); ax[2].legend()
    for c in candidates:
        t = df['time'].iloc[c] if c < len(df) else 0
        for a in ax: a.axvline(t, color='r', linestyle='--', alpha=0.6)
    plt.tight_layout()
    chart_path = os.path.join(PLOTS_DIR, f"{vid_name}_chart.png")
    plt.savefig(chart_path); plt.close()
    # create pdf with methodology+claims
    pdf_path = os.path.join(REPORTS_DIR, f"{vid_name}_report.pdf")
    create_polished_pdf(vid_name, df, chart_path, candidates, pdf_path)
    return {"name": vid_name, "csv": csv_path, "chart": chart_path, "pdf": pdf_path, "candidates": candidates}

# ------------------ POLISHED PDF (cover + methodology + charts + claims) ------------------
METHOD_TEXT = [
    "Methodology: We extract 2D pose landmarks from input video using MediaPipe Pose.",
    "We compute elbow angles, wrist speed, stance width and elbow angular velocity over time.",
    "Release candidates are detected where wrist speed spikes co-occur with large elbow angular acceleration.",
    "A RandomForest classifier flags windows as GOOD/BAD posture using elbow and wrist statistics.",
    "Outputs: annotated video, metrics CSV, charts, and this PDF with visualizations."
]

CLAIMS_TEXT = [
    "Claim (high-level): A computer-implemented method for assessing archery posture using co-occurrence",
    "of wrist speed spikes and elbow angular velocity to identify release events and advise corrective actions.",
    "The method uses sliding-window stability metrics and a classifier to identify suboptimal posture windows."
]

def create_polished_pdf(name, df_metrics, chart_img, candidates, out_pdf):
    c = canvas.Canvas(out_pdf, pagesize=A4)
    W, H = A4; margin = 40; y = H - margin
    # cover
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, y, f"Archery Evaluation — {name}")
    c.setFont("Helvetica", 10)
    y -= 20
    c.drawString(margin, y, "Author: Tushar R. Islampure")
    y -= 16
    c.drawString(margin, y, "One-line summary: Pose-based kinematic analysis and release detection for archery posture evaluation.")
    y -= 28
    # embed chart snapshot
    try:
        img = ImageReader(chart_img)
        iw, ih = img.getSize()
        draw_w = W - 2*margin
        draw_h = draw_w * ih / iw
        if draw_h > y - margin:
            draw_h = y - margin
        c.drawImage(img, margin, y - draw_h, width=draw_w, height=draw_h)
        y -= draw_h + 10
    except Exception:
        pass
    # Methodology
    c.setFont("Helvetica-Bold", 12); c.drawString(margin, y, "Methodology:"); y -= 16
    c.setFont("Helvetica", 9)
    for line in METHOD_TEXT:
        c.drawString(margin, y, "- " + line); y -= 12
        if y < margin + 80:
            c.showPage(); y = H - margin
    y -= 6
    c.setFont("Helvetica-Bold", 12); c.drawString(margin, y, "Claims / Novelty:"); y -= 14
    c.setFont("Helvetica", 9)
    for line in CLAIMS_TEXT:
        c.drawString(margin, y, line); y -= 11
        if y < margin + 80:
            c.showPage(); y = H - margin
    # Observations and sample release frames
    c.setFont("Helvetica-Bold", 12); c.drawString(margin, y, "Key observations & release candidates:"); y -= 14
    c.setFont("Helvetica", 9)
    obs = [
        f"Average elbow (left): {np.nanmean(df_metrics['elbow_l']):.1f} deg",
        f"Average stance width (px): {np.nanmean(df_metrics['stance']):.1f}",
        f"Detected release candidate frames: {', '.join(map(str, candidates)) if candidates else 'None'}"
    ]
    for o in obs:
        c.drawString(margin, y, "- " + o); y -= 11
        if y < margin + 80:
            c.showPage(); y = H - margin
    # embed a few frames (if present)
    sample_frames = sorted([os.path.join(FRAMES_DIR, f) for f in os.listdir(FRAMES_DIR) if f.startswith(name)])[:4]
    for imgpath in sample_frames:
        if y < margin + 140:
            c.showPage(); y = H - margin
        try:
            img = ImageReader(imgpath); iw, ih = img.getSize()
            draw_w = (W - 2*margin) / 2; draw_h = draw_w * ih / iw
            c.drawImage(img, margin, y - draw_h, width=draw_w, height=draw_h)
            margin2 = margin + draw_w + 10
            # next image placement
            y -= draw_h + 10
        except Exception:
            continue
    c.showPage()
    c.save()

# ------------------ TRAIN SIMPLE CLASSIFIER (weak labels) ------------------
def train_posture_classifier(metrics_folder):
    rows=[]
    for f in os.listdir(metrics_folder):
        if not f.endswith("_metrics.csv"): continue
        df = pd.read_csv(os.path.join(metrics_folder,f))
        win = CLASSIFIER_WIN_FRAMES
        if len(df) < win: continue
        for i in range(0, len(df)-win, win):
            wdf = df.iloc[i:i+win]
            avg_elbow = np.nanmean(wdf['elbow_l'])
            var_ws = np.nanvar(wdf['wrist_speed'])
            avg_stance = np.nanmean(wdf['stance'])
            if not np.isnan(avg_elbow) and 140 <= avg_elbow <= 175 and var_ws < 5.0:
                label = 1  # GOOD
            else:
                label = 0  # BAD
            rows.append([avg_elbow, var_ws, avg_stance, label])
    if not rows:
        print("No training windows found.")
        return None
    dfw = pd.DataFrame(rows, columns=['avg_elbow','var_ws','avg_stance','label']).dropna()
    X = dfw[['avg_elbow','var_ws','avg_stance']]; y = dfw['label']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=120, random_state=42)
    clf.fit(X_train, y_train)
    ypred = clf.predict(X_test)
    print("Classifier results:\n", classification_report(y_test, ypred))
    model_path = os.path.join(REPORTS_DIR, "rf_posture_model.joblib")
    joblib.dump(clf, model_path)
    dfw.to_csv(os.path.join(REPORTS_DIR, "classifier_training_windows.csv"), index=False)
    return clf

# ------------------ MAIN ------------------
def main():
    vids = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if f.lower().endswith(".mp4")]
    if not vids:
        print("Place .mp4 files in input_videos/ and re-run.")
        return
    results = []
    for v in vids:
        try:
            print("Processing:", v)
            r = process_video(v)
            results.append(r)
            print("Saved PDF:", r['pdf'])
        except Exception as e:
            print("Error processing", v, e)
    clf = train_posture_classifier(METRICS_DIR)
    print("All done. Reports are in:", REPORTS_DIR)

if __name__ == "__main__":
    main()
