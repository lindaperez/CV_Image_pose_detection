"""
Generate a stu_id identity grid PDF.
Rows = stu_ids (1-13), Columns = exercise classes.
One representative frame (midpoint) per (stu_id x exercise) cell.
Purpose: visually determine if stu{id} is a consistent person identifier.
"""

import os, re
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages

BASE_DIR = "/Users/lindaperez/Documents/COMPUTER_VISION/Final_project/personal-git/ML_System"
DATA_DIR = os.path.join(BASE_DIR, "datasets/metadata/llsp")
VIDEO_BASE = os.path.join(BASE_DIR, "Data/LLSP/video")
OUTPUT_PDF = os.path.join(BASE_DIR, "reports/stu_identity_grid.pdf")

TYPO_MAP = {
    'squant':'squat','frontraise':'front_raise','benchpressing':'bench_pressing',
    'jumpjacks':'jump_jacks','jump_jack':'jump_jacks','situp':'sit_up',
    'pullups':'pull_up','pushups':'push_up',
}

def load(path, split):
    df = pd.read_csv(path)
    df['split'] = split
    df['type'] = df['type'].map(lambda x: TYPO_MAP.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x)
    return df

print("Loading CSVs...")
all_df = pd.concat([
    load(f'{DATA_DIR}/train.csv', 'train'),
    load(f'{DATA_DIR}/valid.csv', 'valid'),
    load(f'{DATA_DIR}/test.csv',  'test'),
], ignore_index=True)

all_df['subject_id'] = all_df['name'].map(
    lambda n: int(m.group(1)) if (m := re.match(r'^stu(\d+)_', str(n))) else None
)
stu_df = all_df.dropna(subset=['subject_id']).copy()
stu_df['subject_id'] = stu_df['subject_id'].astype(int)

# Pick one representative video per (stu_id, exercise) — first alphabetically
rep = (
    stu_df.sort_values('name')
    .groupby(['subject_id', 'type'])
    .first()
    .reset_index()[['subject_id', 'type', 'name', 'split']]
)

# Layout
STU_IDS    = sorted(rep['subject_id'].unique())
EXERCISES  = sorted(rep['type'].unique())
N_STU      = len(STU_IDS)
N_EX       = len(EXERCISES)

print(f"Grid: {N_STU} stu_ids x {N_EX} exercises = {len(rep)} cells")
print(f"Exercises: {EXERCISES}")

def extract_mid_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total // 2))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize to fixed thumbnail
    h, w = frame.shape[:2]
    target_w = 160
    target_h = int(h * target_w / w)
    return cv2.resize(frame, (target_w, target_h))

print("Extracting frames...")
# Build lookup: (stu_id, exercise) -> frame or None
frame_cache = {}
for _, row in rep.iterrows():
    key = (int(row['subject_id']), row['type'])
    vpath = os.path.join(VIDEO_BASE, row['split'], row['name'])
    if os.path.exists(vpath):
        frame_cache[key] = extract_mid_frame(vpath)
        status = "ok" if frame_cache[key] is not None else "read_fail"
    else:
        frame_cache[key] = None
        status = "missing"
    print(f"  stu{key[0]:2d} {row['type']:20s} [{status}] {row['name']}")

# ── PDF: 4 stu_ids per page for readability ────────────────────────────────
STU_PER_PAGE = 4
CELL_W = 1.8   # inches per cell
CELL_H = 1.55  # inches per cell (frame + labels)
LABEL_H = 0.30 # inches for column header row

PAGE_W = CELL_W * N_EX + 0.6   # left margin for stu label
PAGE_H = (CELL_H * STU_PER_PAGE) + LABEL_H + 0.4

print(f"\nGenerating PDF: {OUTPUT_PDF}")

with PdfPages(OUTPUT_PDF) as pdf:
    stu_chunks = [STU_IDS[i:i+STU_PER_PAGE] for i in range(0, N_STU, STU_PER_PAGE)]

    for chunk_idx, chunk in enumerate(stu_chunks):
        n_rows = len(chunk)
        fig_h = (CELL_H * n_rows) + LABEL_H + 0.4
        fig, axes = plt.subplots(
            n_rows, N_EX,
            figsize=(PAGE_W, fig_h),
            gridspec_kw={'hspace': 0.05, 'wspace': 0.05}
        )
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        fig.patch.set_facecolor('#f8f9fa')

        # Column headers (exercise names) — on first row only via title
        for col_idx, ex in enumerate(EXERCISES):
            ax = axes[0, col_idx]
            short = ex.replace('_', '\n')
            ax.set_title(short, fontsize=7.5, fontweight='bold', color='#1a3a5c',
                         pad=3, wrap=True)

        for row_idx, sid in enumerate(chunk):
            for col_idx, ex in enumerate(EXERCISES):
                ax = axes[row_idx, col_idx]
                frame = frame_cache.get((sid, ex))

                if frame is not None:
                    ax.imshow(frame, aspect='auto')
                else:
                    ax.set_facecolor('#e0e0e0')
                    ax.text(0.5, 0.5, '—', ha='center', va='center',
                            fontsize=12, color='#999', transform=ax.transAxes)

                ax.set_xticks([])
                ax.set_yticks([])

                # Highlight border
                for spine in ax.spines.values():
                    spine.set_edgecolor('#cccccc')
                    spine.set_linewidth(0.5)

            # Row label: stu_id
            axes[row_idx, 0].set_ylabel(
                f'stu{sid}', fontsize=9, fontweight='bold',
                color='#1a3a5c', rotation=0, labelpad=28, va='center'
            )

        page_label = f"Page {chunk_idx+1}/{len(stu_chunks)}  —  stu_id Identity Grid  —  1 frame per (stu_id × exercise)"
        fig.text(0.5, 0.01, page_label, ha='center', fontsize=7, color='#6b7280')
        fig.text(0.5, 0.99,
                 "RepCount-A: Is stu{id} a consistent person identifier?",
                 ha='center', fontsize=9, fontweight='bold', color='#1a3a5c', va='top')

        plt.tight_layout(rect=[0.04, 0.03, 1.0, 0.97])
        pdf.savefig(fig, dpi=120, bbox_inches='tight', facecolor='#f8f9fa')
        plt.close(fig)
        print(f"  Page {chunk_idx+1} done (stu_ids: {[f'stu{s}' for s in chunk]})")

print(f"\nDone: {OUTPUT_PDF}")
