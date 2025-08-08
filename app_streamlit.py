# app_streamlit.py
# Author: Tushar R. Islampure
# Purpose: Robust demo to play original and annotated videos, with re-encode fallback.
import streamlit as st
import os, shutil, subprocess, time
import pandas as pd
from pathlib import Path

st.set_page_config(layout="wide", page_title="Archery Evaluation Demo")
st.title("Archery Evaluation — Demo (Tushar)")

INPUT_DIR = "input_videos"
OUTPUT_DIR = "outputs_videos"
METRICS_DIR = "metrics_all"
FRAMES_DIR = "output_frames"

# helper functions
def ffmpeg_available():
    return shutil.which("ffmpeg") is not None

def reencode_to_h264(src_path, dst_path):
    """Transcode src_path -> dst_path using ffmpeg H.264. Returns (ok, message)."""
    if not ffmpeg_available():
        return False, "ffmpeg not found on PATH"
    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        dst_path
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        if proc.returncode == 0 and os.path.exists(dst_path):
            return True, "Re-encode OK"
        else:
            return False, proc.stderr.decode(errors="ignore")[:2000]
    except Exception as e:
        return False, str(e)

def serve_video_bytes(path):
    """Open file as bytes and call st.video on bytes."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        st.video(data)
        st.success(f"Served bytes ({len(data):,} bytes)")
        return True
    except Exception as e:
        st.error(f"Unable to serve bytes: {e}")
        return False

def show_file_debug(path):
    """Show helpful debugging info about the file"""
    p = Path(path)
    st.write("**Path:**", str(p.resolve()))
    if p.exists():
        st.write("**Size:**", f"{p.stat().st_size:,} bytes")
        st.write("**Last modified:**", time.ctime(p.stat().st_mtime))
    else:
        st.write(":warning: File does not exist")

# main UI
video_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(".mp4")]
video_files.sort()
sel = st.selectbox("Choose input video", video_files) if video_files else None

if not sel:
    st.info("Put .mp4 videos inside the 'input_videos/' folder and re-run.")
else:
    col_left, col_right = st.columns([2,1])
    with col_left:
        st.header("Video Player")
        orig_path = os.path.join(INPUT_DIR, sel)
        base = os.path.splitext(sel)[0]
        annotated_name = f"{base}_annotated.mp4"
        ann_path = os.path.join(OUTPUT_DIR, annotated_name)
        tab1, tab2 = st.tabs(["Original", "Annotated"])
        with tab1:
            st.subheader("Original video")
            st.video(orig_path)

with tab2:
    if os.path.exists(ann_path):
        try:
            st.video(ann_path)  # direct path
        except Exception:
            try:
                fixed_path = ann_path.replace("_annotated.mp4", "_fixed.mp4")
                cmd = [
                    "ffmpeg", "-y", "-i", ann_path,
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", fixed_path
                ]
                subprocess.run(cmd, check=True)
                st.video(fixed_path)
            except Exception as e:
                st.error(f"Video failed to play and re-encode failed: {e}")
    else:
        st.warning("Annotated video not yet created. Run main_pro.py first.")

    with col_right:
        st.header("Metrics & Actions")
        csv_path = os.path.join(METRICS_DIR, f"{os.path.splitext(sel)[0]}_metrics.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            st.subheader("Metrics preview")
            st.line_chart(df[['elbow_l','wrist_speed']].fillna(0).rename(columns={'elbow_l':'elbow','wrist_speed':'wrist'}))
            st.dataframe(df.describe().loc[['mean','std']])
            st.markdown("---")
            if os.path.exists(ann_path):
                st.download_button("Download annotated video", data=open(ann_path,'rb').read(), file_name=annotated_name, mime="video/mp4")
            else:
                st.info("Annotated video download will appear here after generation.")
        else:
            st.info("Metrics CSV not found; run main_pro.py to generate metrics.")

