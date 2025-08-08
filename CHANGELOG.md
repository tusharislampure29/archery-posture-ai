# Changelog
## v1.0 — 2025-08-09 - Initial prototype
- Baseline pose extraction (MediaPipe) + angle computation.
- Output: annotated frames and basic CSV metrics.

## v1.1 — 2025-08-09 - Release detection
- Added cross-correlation-style release detection using wrist & elbow dynamics.
- Save candidate frames and charts.
- - Kinematic metrics & release detection

## v1.2 —  2025-08-09 - Classifier & PDFs
- Add RandomForest classifier with weak labels.
- Produce polished PDF per-video with cover page, methodology and claims.
- Annotated video output (H.264 / yuv420p)

## v1.3 — 2025-08-09 - Streamlit demo & packaging
- Add `app_streamlit.py` for quick reviewer demo.
- Prepare repo structure and instructions for submission.
