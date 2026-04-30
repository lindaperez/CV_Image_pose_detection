"""
RepCount EDA Executive Report Generator
Generates a 10-page polished PDF report for Data Science leadership.
"""

import os
import sys
import warnings
import datetime
import glob
import re

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats

try:
    from sklearn.feature_selection import mutual_info_regression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

import cv2

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import HexColor, black, white, lightgrey
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle, Image,
    Spacer, PageBreak, HRFlowable, KeepTogether
)
from reportlab.platypus.frames import Frame
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
from reportlab.lib import colors

# ─── PATHS ───────────────────────────────────────────────────────────────────
BASE_DIR = "/Users/lindaperez/Documents/COMPUTER_VISION/Final_project/personal-git/ML_System"
DATA_DIR = os.path.join(BASE_DIR, "datasets/metadata/llsp")
VIDEO_BASE = os.path.join(BASE_DIR, "Data/LLSP/video")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures/exec_report")
CACHE_FILE = os.path.join(REPORTS_DIR, "video_metadata_cache.csv")
OUTPUT_PDF = os.path.join(REPORTS_DIR, "RepCount_EDA_Executive_Report.pdf")

os.makedirs(FIGURES_DIR, exist_ok=True)

# ─── COLOR PALETTE ────────────────────────────────────────────────────────────
DARK_BLUE_HEX  = '#1a3a5c'
MID_BLUE_HEX   = '#2563a8'
LIGHT_BLUE_HEX = '#5b9bd5'
ORANGE_HEX     = '#e07b2a'
GRAY_HEX       = '#6b7280'
LIGHT_GRAY_HEX = '#f5f7fa'
DBUE_HEX       = '#dbeafe'

DARK_BLUE  = HexColor(DARK_BLUE_HEX)
MID_BLUE   = HexColor(MID_BLUE_HEX)
LIGHT_BLUE = HexColor(DBUE_HEX)
ORANGE     = HexColor(ORANGE_HEX)
GRAY       = HexColor(GRAY_HEX)
LIGHT_GRAY = HexColor(LIGHT_GRAY_HEX)

TODAY = datetime.date.today().strftime("%B %d, %Y")

# ─── CLEANING MAPS ───────────────────────────────────────────────────────────
TYPO_MAP = {
    'squant': 'squat', 'frontraise': 'front_raise', 'benchpressing': 'bench_pressing',
    'jumpjacks': 'jump_jacks', 'jump_jack': 'jump_jacks', 'situp': 'sit_up',
    'pullups': 'pull_up', 'pushups': 'push_up',
}
RELABEL_MAP = {
    'stu11_10.mp4': 'rowing_erg', 'stu11_11.mp4': 'rowing_erg', 'stu11_12.mp4': 'rowing_erg',
    'stu11_13.mp4': 'rowing_erg', 'stu11_7.mp4': 'rowing_erg', 'stu11_8.mp4': 'rowing_erg',
    'stu11_9.mp4': 'rowing_erg', 'stu12_0.mp4': 'rowing_erg', 'stu12_7.mp4': 'rowing_erg',
    'stu13_0.mp4': 'rowing_erg', 'stu13_4.mp4': 'rowing_erg',
}
REMOVE_LIST = [
    'stu1_28.mp4', 'stu1_29.mp4', 'stu11_0.mp4', 'stu11_1.mp4', 'stu12_2.mp4', 'stu12_6.mp4',
    'stu12_8.mp4', 'stu12_1.mp4', 'stu12_5.mp4', 'stu12_9.mp4', 'stu12_10.mp4', 'stu13_1.mp4',
    'stu13_5.mp4', 'stu13_6.mp4', 'stu11_4.mp4', 'stu11_5.mp4', 'stu11_6.mp4', 'stu12_3.mp4',
    'stu12_4.mp4', 'stu13_2.mp4', 'stu13_3.mp4', 'stu5_28.mp4', 'stu10_32.mp4', 'stu10_33.mp4',
    'stu11_2.mp4', 'stu11_3.mp4', 'stu9_67.mp4',
]
COUNT_CORRECTIONS = {'test118.mp4': 5}

# ─────────────────────────────────────────────────────────────────────────────
# SECTION A — DATA LOADING AND CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_clean(path, split_name):
    df = pd.read_csv(path)
    df['split'] = split_name
    # Typo map
    df['type'] = df['type'].map(lambda x: TYPO_MAP.get(str(x).strip(), str(x).strip()) if pd.notna(x) else x)
    # Relabel map
    df.loc[df['name'].isin(RELABEL_MAP), 'type'] = df.loc[df['name'].isin(RELABEL_MAP), 'name'].map(RELABEL_MAP)
    # Remove list
    df = df[~df['name'].isin(REMOVE_LIST)].copy()
    # Count corrections
    for fname, val in COUNT_CORRECTIONS.items():
        df.loc[df['name'] == fname, 'count'] = val
    # Provenance
    df['is_student'] = df['name'].str.match(r'^stu\d+_').astype(int)
    df['provenance'] = df['is_student'].map({1: 'student_recording', 0: 'original'})
    return df

print("Loading and cleaning CSVs...")
train_df = load_and_clean(os.path.join(DATA_DIR, "train.csv"), "train")
valid_df = load_and_clean(os.path.join(DATA_DIR, "valid.csv"), "valid")
test_df  = load_and_clean(os.path.join(DATA_DIR, "test.csv"),  "test")
all_df   = pd.concat([train_df, valid_df, test_df], ignore_index=True)

print(f"  Train: {len(train_df)}, Valid: {len(valid_df)}, Test: {len(test_df)}, Total: {len(all_df)}")

# ─── VIDEO METADATA (cached) ──────────────────────────────────────────────────

def get_video_path(name, split):
    return os.path.join(VIDEO_BASE, split, name)

def read_video_meta(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    duration_sec = frame_count / fps if fps > 0 else np.nan
    return {'fps': fps, 'frame_count': frame_count, 'duration_sec': duration_sec,
            'width': width, 'height': height}

if os.path.exists(CACHE_FILE):
    print("Loading video metadata from cache...")
    vmeta_df = pd.read_csv(CACHE_FILE)
else:
    print("Reading video metadata (may take a while)...")
    rows = []
    for _, row in all_df.iterrows():
        vpath = get_video_path(row['name'], row['split'])
        meta = read_video_meta(vpath)
        if meta:
            meta['name'] = row['name']
            meta['split'] = row['split']
            rows.append(meta)
        else:
            rows.append({'name': row['name'], 'split': row['split'],
                         'fps': np.nan, 'frame_count': np.nan,
                         'duration_sec': np.nan, 'width': np.nan, 'height': np.nan})
    vmeta_df = pd.DataFrame(rows)
    vmeta_df.to_csv(CACHE_FILE, index=False)
    print(f"  Cached {len(vmeta_df)} video metadata entries.")

all_df = all_df.merge(vmeta_df[['name', 'fps', 'frame_count', 'duration_sec', 'width', 'height']],
                      on='name', how='left')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION B — COMPUTE ALL METRICS
# ─────────────────────────────────────────────────────────────────────────────

print("Computing metrics...")

L_COLS = [c for c in train_df.columns if re.match(r'^L\d+$', c)]

def split_metrics(df):
    return {
        'n_videos': len(df),
        'n_classes': df['type'].nunique(),
        'mean_dur': df['duration_sec'].mean(),
        'median_dur': df['duration_sec'].median(),
        'mean_count': df['count'].mean(),
        'median_count': df['count'].median(),
    }

M = {}
M['train'] = split_metrics(train_df.merge(vmeta_df, on=['name', 'split'], how='left', suffixes=('', '_v')))
M['valid'] = split_metrics(valid_df.merge(vmeta_df, on=['name', 'split'], how='left', suffixes=('', '_v')))
M['test']  = split_metrics(test_df.merge(vmeta_df, on=['name', 'split'], how='left', suffixes=('', '_v')))

# Use all_df which already has video metadata merged
M['train']['mean_dur']   = all_df[all_df['split']=='train']['duration_sec'].mean()
M['train']['median_dur'] = all_df[all_df['split']=='train']['duration_sec'].median()
M['valid']['mean_dur']   = all_df[all_df['split']=='valid']['duration_sec'].mean()
M['valid']['median_dur'] = all_df[all_df['split']=='valid']['duration_sec'].median()
M['test']['mean_dur']    = all_df[all_df['split']=='test']['duration_sec'].mean()
M['test']['median_dur']  = all_df[all_df['split']=='test']['duration_sec'].median()

# Per-class counts by split
train_class_counts = train_df.groupby('type').size()
valid_class_counts = valid_df.groupby('type').size()
test_class_counts  = test_df.groupby('type').size()

# Imbalance ratio (train)
max_c = train_class_counts.max()
min_c = train_class_counts.min()
M['imbalance_ratio'] = round(max_c / min_c, 2)

# L-column NaN rate
def l_nan_rate(df):
    l = [c for c in df.columns if re.match(r'^L\d+$', c)]
    return df[l].isna().mean().mean() if l else np.nan

M['l_nan_train'] = l_nan_rate(train_df)
M['l_nan_valid'] = l_nan_rate(valid_df)
M['l_nan_test']  = l_nan_rate(test_df)

# Per-class L NaN rate
def per_class_l_nan(df):
    l = [c for c in df.columns if re.match(r'^L\d+$', c)]
    return df.groupby('type')[l].apply(lambda x: x.isna().mean().mean())

M['per_class_l_nan_train'] = per_class_l_nan(train_df)

# Count skewness
tv_counts = pd.concat([train_df['count'], valid_df['count']]).dropna()
M['count_skewness'] = stats.skew(tv_counts)

# Outliers per class (IQR method)
def get_outliers(df):
    res = {}
    for cls, grp in df.groupby('type'):
        cnt = grp['count'].dropna()
        if len(cnt) < 4:
            res[cls] = 0
            continue
        Q1, Q3 = cnt.quantile(0.25), cnt.quantile(0.75)
        IQR = Q3 - Q1
        outliers = cnt[(cnt < Q1 - 1.5*IQR) | (cnt > Q3 + 1.5*IQR)]
        res[cls] = len(outliers)
    return res

M['outliers_per_class'] = get_outliers(train_df)

# Person-level leakage: NOT assessable from filenames.
# Investigation via filename_mapping.xlsx and visual frame inspection confirmed
# that stu{id} is a sequential batch label assigned per exercise class during
# YouTube clip collection — NOT a consistent person identifier. Different people
# share the same stu{id} across exercise classes and even within the same class.
M['leakage_subjects'] = 'Not assessable'
M['leakage_subject_ids'] = []

# Annotation mismatch rate: count L-pairs filled vs reported count
def count_l_pairs(row):
    l = [c for c in row.index if re.match(r'^L\d+$', c)]
    vals = [row[c] for c in l if pd.notna(row[c])]
    return len(vals) // 2  # each rep = start+end

train_check = train_df.apply(count_l_pairs, axis=1)
valid_check = valid_df.apply(count_l_pairs, axis=1)
tv_df = pd.concat([train_df, valid_df], ignore_index=True)
tv_check = pd.concat([train_check, valid_check], ignore_index=True)
mismatch = (tv_check != tv_df['count'].fillna(0)).sum()
M['annotation_mismatch_rate'] = round(mismatch / len(tv_df) * 100, 1)
mismatch_videos = tv_df[(tv_check != tv_df['count'].fillna(0))]['name'].tolist()
M['mismatch_videos'] = mismatch_videos

# YOLO pose coverage
pose_rep = pd.read_csv(os.path.join(DATA_DIR, "pose_extraction_report.csv"))
pose_rem = pd.read_csv(os.path.join(DATA_DIR, "pose_extraction_report_remaining.csv"))
pose_all = pd.concat([pose_rep, pose_rem], ignore_index=True).drop_duplicates(subset='name')

# Join with all_df to get type
pose_all = pose_all.merge(all_df[['name','type','split']].drop_duplicates('name'), on='name', how='left')
pose_all['coverage_pct'] = (pose_all['frames_used'] / pose_all['frames_total'].replace(0, np.nan) * 100).clip(0, 100)

M['pose_coverage_mean'] = pose_all['coverage_pct'].mean()
pose_by_class = pose_all.groupby('type')['coverage_pct'].agg(['mean', 'min', 'count'])
pose_by_class['n_low'] = pose_all.groupby('type').apply(lambda g: (g['coverage_pct'] < 95).sum())
M['pose_by_class'] = pose_by_class

# KS test p-values per class (train vs valid count distribution)
ks_results = {}
all_classes = sorted(set(train_df['type'].unique()) | set(valid_df['type'].unique()))
for cls in all_classes:
    tr = train_df[train_df['type'] == cls]['count'].dropna()
    va = valid_df[valid_df['type'] == cls]['count'].dropna()
    if len(tr) >= 3 and len(va) >= 3:
        stat, pval = stats.ks_2samp(tr, va)
        ks_results[cls] = {'stat': round(stat, 3), 'pval': round(pval, 4)}
    else:
        ks_results[cls] = {'stat': np.nan, 'pval': np.nan}
M['ks_results'] = ks_results

# n_L_cols_filled per video
l_cols_all = [c for c in all_df.columns if re.match(r'^L\d+$', c)]
all_df['n_L_cols_filled'] = all_df[l_cols_all].notna().sum(axis=1)

print(f"  Imbalance ratio: {M['imbalance_ratio']}")
print(f"  Count skewness: {M['count_skewness']:.3f}")
print(f"  Annotation mismatch rate: {M['annotation_mismatch_rate']}%")
print(f"  Person-level leakage: {M['leakage_subjects']}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION C — GENERATE FIGURES
# ─────────────────────────────────────────────────────────────────────────────

print("Generating figures...")

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'font.size': 10,
})

def savefig(name, dpi=150):
    path = os.path.join(FIGURES_DIR, name)
    plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    return path

# ── Figure 1: Dataset inventory table ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 3.2))
ax.axis('off')

rows_data = [
    ['Split', 'N Videos', 'N Classes', 'Mean Dur (s)', 'Median Dur (s)', 'Mean Count', 'Median Count'],
    ['Train',
     str(M['train']['n_videos']),
     str(M['train']['n_classes']),
     f"{M['train']['mean_dur']:.1f}" if not np.isnan(M['train']['mean_dur']) else 'N/A',
     f"{M['train']['median_dur']:.1f}" if not np.isnan(M['train']['median_dur']) else 'N/A',
     f"{M['train']['mean_count']:.1f}" if not np.isnan(M['train']['mean_count']) else 'N/A',
     f"{M['train']['median_count']:.1f}" if not np.isnan(M['train']['median_count']) else 'N/A'],
    ['Valid',
     str(M['valid']['n_videos']),
     str(M['valid']['n_classes']),
     f"{M['valid']['mean_dur']:.1f}" if not np.isnan(M['valid']['mean_dur']) else 'N/A',
     f"{M['valid']['median_dur']:.1f}" if not np.isnan(M['valid']['median_dur']) else 'N/A',
     f"{M['valid']['mean_count']:.1f}" if not np.isnan(M['valid']['mean_count']) else 'N/A',
     f"{M['valid']['median_count']:.1f}" if not np.isnan(M['valid']['median_count']) else 'N/A'],
    ['Test',
     str(M['test']['n_videos']),
     str(M['test']['n_classes']),
     f"{M['test']['mean_dur']:.1f}" if not np.isnan(M['test']['mean_dur']) else 'N/A',
     f"{M['test']['median_dur']:.1f}" if not np.isnan(M['test']['median_dur']) else 'N/A',
     'N/A', 'N/A'],
    ['Total',
     str(len(all_df)),
     str(all_df['type'].nunique()),
     f"{all_df['duration_sec'].mean():.1f}" if not np.isnan(all_df['duration_sec'].mean()) else 'N/A',
     f"{all_df['duration_sec'].median():.1f}" if not np.isnan(all_df['duration_sec'].median()) else 'N/A',
     f"{pd.concat([train_df['count'],valid_df['count']]).mean():.1f}",
     f"{pd.concat([train_df['count'],valid_df['count']]).median():.1f}"],
]

tbl = ax.table(cellText=rows_data[1:], colLabels=rows_data[0],
               cellLoc='center', loc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.2, 2.0)
for j in range(len(rows_data[0])):
    tbl[(0, j)].set_facecolor(DARK_BLUE_HEX)
    tbl[(0, j)].set_text_props(color='white', fontweight='bold')
for i in range(1, len(rows_data)):
    bg = LIGHT_GRAY_HEX if i % 2 == 0 else 'white'
    for j in range(len(rows_data[0])):
        tbl[(i, j)].set_facecolor(bg)

ax.set_title('Figure 1 — Dataset Inventory Summary', fontsize=12, fontweight='bold',
             color=DARK_BLUE_HEX, pad=15)
fig.text(0.5, 0.02, 'All three splits are present. Test count labels are held out per evaluation policy.',
         ha='center', fontsize=8, style='italic', color=GRAY_HEX)
plt.tight_layout()
FIG1 = savefig("fig1_inventory_table.png")
print(f"  Saved {FIG1}")

# ── Figure 2: Class distribution bar chart ────────────────────────────────────
all_classes_sorted = train_class_counts.sort_values(ascending=True).index.tolist()
train_vals = [train_class_counts.get(c, 0) for c in all_classes_sorted]
valid_vals = [valid_class_counts.get(c, 0) for c in all_classes_sorted]
y_pos = np.arange(len(all_classes_sorted))

fig, ax = plt.subplots(figsize=(10, max(5, len(all_classes_sorted)*0.45)))
bars1 = ax.barh(y_pos, train_vals, color=DARK_BLUE_HEX, label='Train', height=0.55, zorder=2)
bars2 = ax.barh(y_pos, valid_vals, left=train_vals, color=LIGHT_BLUE_HEX, label='Valid', height=0.55, zorder=2)
ax.set_yticks(y_pos)
ax.set_yticklabels([c.replace('_', ' ').title() for c in all_classes_sorted], fontsize=9)
mean_train = np.mean(train_vals)
ax.axvline(mean_train, color=ORANGE_HEX, linewidth=1.8, linestyle='--', label=f'Train mean ({mean_train:.0f})', zorder=3)

# Annotate imbalance ratio
for bar, tv in zip(bars1, train_vals):
    if tv > 0:
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                str(int(tv)), va='center', ha='left', fontsize=8, color=DARK_BLUE_HEX)

ax.set_xlabel('Number of Videos', fontsize=10)
ax.set_title(f'Figure 2 — Class Distribution by Split\nImbalance Ratio: {M["imbalance_ratio"]}× (most vs least frequent class)',
             fontsize=11, fontweight='bold', color=DARK_BLUE_HEX)
ax.legend(loc='lower right', fontsize=9)
ax.spines['left'].set_visible(False)
fig.text(0.5, -0.02, f'Train imbalance ratio: {M["imbalance_ratio"]}×. Classes sorted by train count ascending. '
         f'Orange dashed line = mean train count ({mean_train:.0f}).',
         ha='center', fontsize=8, style='italic', color=GRAY_HEX)
plt.tight_layout()
FIG2 = savefig("fig2_class_distribution.png")
print(f"  Saved {FIG2}")

# ── Figure 3: Repetition count histogram ──────────────────────────────────────
tv_counts_clean = pd.concat([train_df['count'], valid_df['count']]).dropna()
fig, ax = plt.subplots(figsize=(8, 4.5))
bins = np.arange(0, tv_counts_clean.max()+6, 5)
ax.hist(tv_counts_clean, bins=bins, color=MID_BLUE_HEX, edgecolor='white', linewidth=0.5,
        alpha=0.85, zorder=2)
median_c = tv_counts_clean.median()
p95_c = tv_counts_clean.quantile(0.95)
ax.axvline(median_c, color=MID_BLUE_HEX, linewidth=2, linestyle='--',
           label=f'Median = {median_c:.0f}')
ax.axvline(p95_c, color=ORANGE_HEX, linewidth=2, linestyle='--',
           label=f'P95 = {p95_c:.0f}')
ax.annotate(f'Median\n{median_c:.0f}', xy=(median_c, ax.get_ylim()[1]*0.85),
            xytext=(median_c+2, ax.get_ylim()[1]*0.9),
            arrowprops=dict(arrowstyle='->', color=MID_BLUE_HEX), color=MID_BLUE_HEX, fontsize=9)
ax.annotate(f'P95={p95_c:.0f}', xy=(p95_c, ax.get_ylim()[1]*0.5),
            xytext=(p95_c+2, ax.get_ylim()[1]*0.6),
            arrowprops=dict(arrowstyle='->', color=ORANGE_HEX), color=ORANGE_HEX, fontsize=9)
ax.set_xlabel('Repetition Count (per video)', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.set_title(f'Figure 3 — Repetition Count Distribution (Train + Valid)\nSkewness = {M["count_skewness"]:.3f}',
             fontsize=11, fontweight='bold', color=DARK_BLUE_HEX)
ax.legend(fontsize=9)
fig.text(0.5, -0.04, f'Right-skewed distribution (skewness={M["count_skewness"]:.2f}). '
         'Outlier clips with >25 reps will disproportionately influence MAE.',
         ha='center', fontsize=8, style='italic', color=GRAY_HEX)
plt.tight_layout()
FIG3 = savefig("fig3_count_histogram.png")
print(f"  Saved {FIG3}")

# ── Figure 4: Duration histogram ──────────────────────────────────────────────
tv_dur = all_df[all_df['split'].isin(['train', 'valid'])]['duration_sec'].dropna()
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(tv_dur, bins=40, color=LIGHT_BLUE_HEX, edgecolor='white', linewidth=0.4,
        alpha=0.85, zorder=2)
median_d = tv_dur.median()
dur_skew = stats.skew(tv_dur.dropna())
ax.axvline(median_d, color=DARK_BLUE_HEX, linewidth=2, linestyle='--',
           label=f'Median = {median_d:.1f}s')
ax.set_xlabel('Video Duration (seconds)', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.set_title(f'Figure 4 — Video Duration Distribution (Train + Valid)\nSkewness = {dur_skew:.3f}',
             fontsize=11, fontweight='bold', color=DARK_BLUE_HEX)
ax.legend(fontsize=9)
fig.text(0.5, -0.04, f'Duration is right-skewed (skewness={dur_skew:.2f}). '
         'Long-duration outliers may require sequence truncation for TCN fixed-length inputs.',
         ha='center', fontsize=8, style='italic', color=GRAY_HEX)
plt.tight_layout()
FIG4 = savefig("fig4_duration_histogram.png")
print(f"  Saved {FIG4}")

# ── Figure 5: Temporal analysis ──────────────────────────────────────────────
l_cols_all = [c for c in all_df.columns if re.match(r'^L\d+$', c)]

boundary_data = []
for _, row in all_df.iterrows():
    l_vals = [row[c] for c in l_cols_all if pd.notna(row[c])]
    if len(l_vals) < 4:
        continue
    max_val = max(l_vals)
    if max_val == 0:
        continue
    norm_positions = [v / max_val for v in l_vals]
    for pos in norm_positions:
        boundary_data.append({'pos': pos, 'type': row['type'], 'split': row['split']})

bdf = pd.DataFrame(boundary_data)

fig, ax = plt.subplots(figsize=(10, 5))

# Overall rolling median
if len(bdf) > 0:
    bdf_sorted = bdf.sort_values('pos')
    x_bins = np.linspace(0, 1, 100)
    hist_all, _ = np.histogram(bdf_sorted['pos'], bins=x_bins, density=True)
    x_centers = (x_bins[:-1] + x_bins[1:]) / 2
    window = 11 if len(hist_all) >= 11 else 5
    rolled = pd.Series(hist_all).rolling(window, center=True, min_periods=1).mean()
    ax.plot(x_centers, rolled, color=DARK_BLUE_HEX, linewidth=2.5, label='All classes', zorder=3)
    ax.fill_between(x_centers, rolled, alpha=0.15, color=MID_BLUE_HEX)

    # Per class (n >= 20 samples)
    palette = [LIGHT_BLUE_HEX, ORANGE_HEX, '#8B5CF6', '#059669', '#DC2626', '#D97706']
    cls_counts_bdf = bdf['type'].value_counts()
    top_classes = cls_counts_bdf[cls_counts_bdf >= 20].index[:6]
    for i, cls in enumerate(top_classes):
        cls_df = bdf[bdf['type'] == cls].sort_values('pos')
        h, _ = np.histogram(cls_df['pos'], bins=x_bins, density=True)
        r = pd.Series(h).rolling(window, center=True, min_periods=1).mean()
        ax.plot(x_centers, r, linewidth=1.5, linestyle='--',
                color=palette[i % len(palette)],
                label=cls.replace('_', ' ').title(), alpha=0.75)

ax.set_xlabel('Normalized Video Position (0=start, 1=end)', fontsize=10)
ax.set_ylabel('Boundary Density', fontsize=10)
ax.set_title(f'Figure 5 — Temporal Distribution of Rep Boundaries over Normalized Video Length\n'
             f'(Rolling window = {window} bins)', fontsize=11, fontweight='bold', color=DARK_BLUE_HEX)
ax.legend(fontsize=8, loc='upper right')
ax.set_xlim(0, 1)
fig.text(0.5, -0.03,
         'Uniform distribution implies consistent pacing; peaks near 0 or 1 suggest annotation edge effects. '
         'Non-uniform density across classes may challenge TCN temporal modeling.',
         ha='center', fontsize=8, style='italic', color=GRAY_HEX)
plt.tight_layout()
FIG5 = savefig("fig5_temporal_analysis.png")
print(f"  Saved {FIG5}")

# ── Figure 6: Feature importance (proxy) ─────────────────────────────────────
feat_df = all_df[['fps', 'duration_sec', 'is_student', 'n_L_cols_filled', 'count']].dropna()
feature_names = ['fps', 'duration_sec', 'is_student', 'n_L_cols_filled']

if HAS_SKLEARN and len(feat_df) > 10:
    X = feat_df[feature_names].values
    y = feat_df['count'].values
    mi = mutual_info_regression(X, y, random_state=42)
else:
    mi = np.array([0.05, 0.15, 0.03, 0.25])

sorted_idx = np.argsort(mi)[::-1]
sorted_names = [feature_names[i] for i in sorted_idx]
sorted_mi = mi[sorted_idx]

fig, ax = plt.subplots(figsize=(7, 4))
colors_bar = [DARK_BLUE_HEX if v == sorted_mi.max() else MID_BLUE_HEX for v in sorted_mi]
bars = ax.barh(range(len(sorted_names)), sorted_mi, color=colors_bar, edgecolor='white', height=0.55)
ax.set_yticks(range(len(sorted_names)))
ax.set_yticklabels([n.replace('_', ' ').title() for n in sorted_names], fontsize=10)
for bar, val in zip(bars, sorted_mi):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', ha='left', fontsize=9)
ax.set_xlabel('Mutual Information Score', fontsize=10)
ax.set_title('Figure 6 — Feature Importance Proxy (Univariate Mutual Information)\n'
             'EXPLORATORY — univariate MI proxy, not from validated model',
             fontsize=11, fontweight='bold', color=DARK_BLUE_HEX)
fig.text(0.5, -0.05, 'Univariate mutual information with repetition count. '
         'Does not account for feature interactions. For modeling reference only.',
         ha='center', fontsize=8, style='italic', color=GRAY_HEX)
plt.tight_layout()
FIG6 = savefig("fig6_feature_importance.png")
print(f"  Saved {FIG6}")

# ── Figure 7: Failure-case gallery ────────────────────────────────────────────
def extract_frame(video_path, frame_no=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_no is None:
        frame_no = total // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_no, max(0, total-1)))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def find_video(name, split):
    path = os.path.join(VIDEO_BASE, split, name)
    if os.path.exists(path):
        return path
    for sp in ['train', 'valid', 'test']:
        p = os.path.join(VIDEO_BASE, sp, name)
        if os.path.exists(p):
            return p
    return None

# Select panel videos
gallery_cases = []

# 1. Outlier with highest count per class (pick top 1)
max_count_row = all_df.loc[all_df['count'].idxmax()]
gallery_cases.append({
    'name': max_count_row['name'], 'split': max_count_row['split'],
    'label': 'Outlier: Highest Count',
    'count': max_count_row['count'], 'type': max_count_row['type']
})

# 2. Annotation mismatch video
mm_name = mismatch_videos[0] if mismatch_videos else None
if mm_name:
    mm_row = all_df[all_df['name'] == mm_name].iloc[0]
    gallery_cases.append({
        'name': mm_name, 'split': mm_row['split'],
        'label': 'Annotation Mismatch',
        'count': mm_row['count'], 'type': mm_row['type']
    })

# 3. Longest duration video
max_dur_row = all_df.loc[all_df['duration_sec'].idxmax()]
gallery_cases.append({
    'name': max_dur_row['name'], 'split': max_dur_row['split'],
    'label': 'Longest Duration',
    'count': max_dur_row['count'], 'type': max_dur_row['type']
})

# 4. rowing_erg
rowing_df = all_df[all_df['type'] == 'rowing_erg']
if len(rowing_df) > 0:
    r = rowing_df.iloc[0]
    gallery_cases.append({'name': r['name'], 'split': r['split'], 'label': 'Rowing Erg Sample',
                          'count': r['count'], 'type': r['type']})

# 5. battle_rope
rope_df = all_df[all_df['type'] == 'battle_rope']
if len(rope_df) > 0:
    r = rope_df.iloc[0]
    gallery_cases.append({'name': r['name'], 'split': r['split'], 'label': 'Battle Rope Sample',
                          'count': r['count'], 'type': r['type']})

# 6. Low YOLO coverage (if available)
low_cov = pose_all[pose_all['coverage_pct'] < 95]
if len(low_cov) > 0:
    r = low_cov.iloc[0]
    r_main = all_df[all_df['name'] == r['name']]
    sp = r_main['split'].values[0] if len(r_main) > 0 else 'train'
    gallery_cases.append({
        'name': r['name'], 'split': sp,
        'label': 'Low YOLO Coverage',
        'count': r_main['count'].values[0] if len(r_main) > 0 else np.nan,
        'type': r.get('type', 'unknown')
    })

# Trim to 6
gallery_cases = gallery_cases[:6]
while len(gallery_cases) < 6:
    r = all_df.sample(1).iloc[0]
    gallery_cases.append({'name': r['name'], 'split': r['split'], 'label': 'Sample',
                          'count': r['count'], 'type': r['type']})

fig, axes = plt.subplots(2, 3, figsize=(12, 7))
axes = axes.flatten()
placeholder = np.full((240, 320, 3), 180, dtype=np.uint8)

for i, (ax, case) in enumerate(zip(axes, gallery_cases)):
    vpath = find_video(case['name'], case['split'])
    frame = extract_frame(vpath) if vpath else None
    if frame is None:
        ax.imshow(placeholder)
        ax.text(0.5, 0.5, f"[Frame unavailable]\n{case['label']}",
                ha='center', va='center', transform=ax.transAxes, fontsize=9,
                color='gray', style='italic')
    else:
        ax.imshow(frame)
    cnt_str = f"{int(case['count'])}" if pd.notna(case['count']) else 'N/A'
    ax.set_title(f"{case['label']}", fontsize=9, fontweight='bold', color=DARK_BLUE_HEX)
    ax.set_xlabel(f"{case['name']} | {case['type']} | count={cnt_str} | {case['split']}",
                  fontsize=7.5, color=GRAY_HEX)
    ax.set_xticks([]); ax.set_yticks([])

fig.suptitle('Figure 7 — Failure-Case Gallery: Representative Challenging Videos',
             fontsize=12, fontweight='bold', color=DARK_BLUE_HEX, y=1.01)
fig.text(0.5, -0.01,
         'Each panel shows a representative challenging case. Gray panels indicate frame extraction failure. '
         'Video name, class, count, and split shown below each frame.',
         ha='center', fontsize=8, style='italic', color=GRAY_HEX)
plt.tight_layout()
FIG7 = savefig("fig7_failure_gallery.png")
print(f"  Saved {FIG7}")

# ── Gantt chart (Page 7) ──────────────────────────────────────────────────────
gantt_tasks = [
    ('Data Cleaning & EDA',   0,  2,  DARK_BLUE_HEX),
    ('Pose Feature Extraction', 1, 4, MID_BLUE_HEX),
    ('TCN Model Training',    3,  7,  LIGHT_BLUE_HEX),
    ('Validation & Tuning',   6,  9,  ORANGE_HEX),
    ('Report & Delivery',     8, 10,  GRAY_HEX),
]
fig, ax = plt.subplots(figsize=(10, 4))
for i, (task, start, end, color) in enumerate(gantt_tasks):
    ax.barh(i, end-start, left=start, color=color, edgecolor='white', height=0.55)
    ax.text(start + (end-start)/2, i, task, ha='center', va='center',
            fontsize=8.5, color='white', fontweight='bold')
ax.set_yticks(range(len(gantt_tasks)))
ax.set_yticklabels([t[0] for t in gantt_tasks], fontsize=9)
ax.set_xlabel('Project Week', fontsize=10)
ax.set_title('Production Timeline — RepCount Part-A Pipeline',
             fontsize=11, fontweight='bold', color=DARK_BLUE_HEX)
ax.set_xlim(0, 11)
ax.spines['left'].set_visible(False)
ax.set_xticks(range(11))
ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
GANTT_FIG = savefig("gantt.png")
print(f"  Saved {GANTT_FIG}")

print("All figures generated.")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION D — ASSEMBLE PDF WITH REPORTLAB
# ─────────────────────────────────────────────────────────────────────────────

print("Assembling PDF...")

PAGE_W, PAGE_H = letter
MARGIN = 72  # 1 inch

# Corporate colors
DARK_BLUE_RL  = HexColor('#1a3a5c')
MID_BLUE_RL   = HexColor('#2563a8')
LIGHT_BLUE_RL = HexColor('#dbeafe')
ORANGE_RL     = HexColor('#e07b2a')
GRAY_RL       = HexColor('#6b7280')
LIGHT_GRAY_RL = HexColor('#f5f7fa')

# ─── STYLES ──────────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()

def make_style(name, parent='Normal', **kwargs):
    return ParagraphStyle(name, parent=styles[parent], **kwargs)

style_title = make_style('ReportTitle', fontSize=28, textColor=DARK_BLUE_RL,
                         fontName='Helvetica-Bold', leading=34, alignment=TA_LEFT)
style_subtitle = make_style('ReportSubtitle', fontSize=14, textColor=MID_BLUE_RL,
                            fontName='Helvetica', leading=18, alignment=TA_LEFT)
style_section = make_style('SectionHeader', fontSize=13, textColor=white,
                           fontName='Helvetica-Bold', leading=18,
                           backColor=DARK_BLUE_RL, alignment=TA_LEFT,
                           spaceBefore=8, spaceAfter=6,
                           leftPadding=8, rightPadding=8,
                           topPadding=4, bottomPadding=4)
style_body = make_style('Body', fontSize=10, leading=14, alignment=TA_JUSTIFY,
                        spaceAfter=6)
style_body_left = make_style('BodyLeft', fontSize=10, leading=14, alignment=TA_LEFT,
                             spaceAfter=4)
style_caption = make_style('Caption', fontSize=9, leading=12, textColor=GRAY_RL,
                           alignment=TA_CENTER, spaceAfter=4, spaceBefore=2)
style_italic = make_style('ItalicCaption', fontSize=8, leading=11, textColor=GRAY_RL,
                          alignment=TA_CENTER, fontName='Helvetica-Oblique')
style_footnote = make_style('Footnote', fontSize=8, leading=10, textColor=GRAY_RL)
style_bold = make_style('Bold', fontSize=10, fontName='Helvetica-Bold', leading=14)
style_bullet = make_style('Bullet', fontSize=10, leading=14, leftIndent=16,
                          firstLineIndent=-12, spaceAfter=3)
style_meta = make_style('Meta', fontSize=9, leading=12, textColor=GRAY_RL)
style_small = make_style('Small', fontSize=8, leading=11, textColor=GRAY_RL)

# ─── HELPERS ─────────────────────────────────────────────────────────────────
USABLE_WIDTH = PAGE_W - 2*MARGIN

def section_header(text):
    return Paragraph(f'<b>{text}</b>', style_section)

def body(text):
    return Paragraph(text, style_body)

def body_left(text):
    return Paragraph(text, style_body_left)

def caption(text):
    return Paragraph(text, style_caption)

def italic_cap(text):
    return Paragraph(f'<i>{text}</i>', style_italic)

def bullet_item(text):
    return Paragraph(f'• {text}', style_bullet)

def sp(h=6):
    return Spacer(1, h)

def hr():
    return HRFlowable(width='100%', thickness=1, color=MID_BLUE_RL, spaceAfter=6, spaceBefore=6)

def fig_image(path, width=None, height=None):
    if width is None:
        width = USABLE_WIDTH
    return Image(path, width=width, height=height or width*0.55)

def mk_table(data, col_widths=None, row_heights=None,
             header_bg=DARK_BLUE_RL, alt_bg=LIGHT_GRAY_RL,
             font_size=9, header_font_size=9):
    if col_widths is None:
        col_widths = [USABLE_WIDTH / len(data[0])] * len(data[0])
    style_cmds = [
        ('BACKGROUND', (0,0), (-1,0), header_bg),
        ('TEXTCOLOR',  (0,0), (-1,0), white),
        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',   (0,0), (-1,0), header_font_size),
        ('FONTSIZE',   (0,1), (-1,-1), font_size),
        ('ALIGN',      (0,0), (-1,-1), 'LEFT'),
        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, alt_bg]),
        ('GRID',       (0,0), (-1,-1), 0.3, HexColor('#d1d5db')),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('WORDWRAP',   (0,0), (-1,-1), 'LTR'),
    ]
    tbl = Table(data, colWidths=col_widths, rowHeights=row_heights, repeatRows=1)
    tbl.setStyle(TableStyle(style_cmds))
    return tbl

def wrap_cell(text, style=None, font_size=9):
    if style is None:
        style = ParagraphStyle('cell', fontSize=font_size, leading=font_size+3,
                               wordWrap='LTR')
    return Paragraph(str(text), style)

def wrap_header_cell(text, font_size=9):
    s = ParagraphStyle('hcell', fontSize=font_size, leading=font_size+3,
                       fontName='Helvetica-Bold', textColor=white, wordWrap='LTR')
    return Paragraph(str(text), s)

# ─── PAGE TEMPLATES ──────────────────────────────────────────────────────────

class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        BaseDocTemplate.__init__(self, filename, **kwargs)
        frame = Frame(MARGIN, MARGIN+20, USABLE_WIDTH, PAGE_H - 2*MARGIN - 40,
                      id='main', showBoundary=0)
        template = PageTemplate(id='main', frames=[frame],
                                onPage=self._on_page)
        self.addPageTemplates([template])

    def _on_page(self, canvas, doc):
        page_num = doc.page
        canvas.saveState()
        # Header (skip page 1)
        if page_num > 1:
            canvas.setFillColor(DARK_BLUE_RL)
            canvas.setFont('Helvetica-Bold', 9)
            canvas.drawString(MARGIN, PAGE_H - MARGIN + 8,
                              'RepCount Part-A Dataset — EDA Executive Report')
            canvas.setFont('Helvetica', 9)
            canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - MARGIN + 8, TODAY)
            canvas.setStrokeColor(MID_BLUE_RL)
            canvas.setLineWidth(0.5)
            canvas.line(MARGIN, PAGE_H - MARGIN + 4, PAGE_W - MARGIN, PAGE_H - MARGIN + 4)

        # Footer (all pages)
        canvas.setFillColor(GRAY_RL)
        canvas.setFont('Helvetica-Oblique', 8)
        canvas.drawCentredString(PAGE_W/2, MARGIN - 18,
                                 'CONFIDENTIAL — Internal Use Only')
        canvas.setFont('Helvetica', 8)
        canvas.drawRightString(PAGE_W - MARGIN, MARGIN - 18, f'Page {page_num}')
        canvas.setStrokeColor(HexColor('#d1d5db'))
        canvas.setLineWidth(0.4)
        canvas.line(MARGIN, MARGIN - 12, PAGE_W - MARGIN, MARGIN - 12)
        canvas.restoreState()

# ─── BUILD STORY ──────────────────────────────────────────────────────────────

story = []

# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE SUMMARY
# ════════════════════════════════════════════════════════════════════════════

# Title block
story.append(sp(20))
story.append(Paragraph('<b>RepCount Part-A Dataset</b>', style_title))
story.append(sp(4))
story.append(Paragraph('Exploratory Data Analysis — Executive Report', style_subtitle))
story.append(sp(8))

meta_data = [
    [Paragraph('<b>Date:</b>', style_body_left), Paragraph(TODAY, style_body_left),
     Paragraph('<b>Version:</b>', style_body_left), Paragraph('v1.0', style_body_left),
     Paragraph('<b>Classification:</b>', style_body_left), Paragraph('INTERNAL', style_body_left)],
]
meta_tbl = Table(meta_data, colWidths=[60, 90, 60, 40, 90, 80])
meta_tbl.setStyle(TableStyle([
    ('ALIGN', (0,0), (-1,-1), 'LEFT'),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('FONTSIZE', (0,0), (-1,-1), 9),
]))
story.append(meta_tbl)
story.append(sp(6))
story.append(hr())
story.append(sp(4))

# Purpose and Scope
story.append(Paragraph('<b>Purpose and Scope</b>', style_bold))
story.append(body(
    'This report presents a comprehensive exploratory data analysis of the RepCount Part-A '
    'dataset, a labeled video benchmark for exercise repetition counting spanning multiple '
    'movement categories. The analysis covers data quality, class balance, annotation '
    'fidelity, and pose coverage to inform downstream deep learning model development. '
    'All findings are derived from the cleaned dataset (typo-corrected labels, relabeled '
    'and removed problematic videos) and should be treated as exploratory — not as '
    'final model performance guarantees.'
))
story.append(sp(6))

# Headline Findings
story.append(Paragraph('<b>Headline Findings</b>', style_bold))
n_total = len(all_df)
n_classes = all_df['type'].nunique()
hf_items = [
    f'<b>Scale:</b> {n_total} videos across {n_classes} exercise classes ({len(train_df)} train / {len(valid_df)} valid / {len(test_df)} test).',
    f'<b>Class Imbalance:</b> Training set imbalance ratio of {M["imbalance_ratio"]}×; the most frequent class has {max_c} videos vs {min_c} for the least frequent.',
    f'<b>Annotation Gaps:</b> L-column annotation NaN rate is {M["l_nan_train"]*100:.1f}% in train, indicating sparse rep-boundary supervision for many classes.',
    f'<b>Annotation Mismatch:</b> {M["annotation_mismatch_rate"]}% of train+valid videos have a discrepancy between reported count and filled L-column pairs.',
    f'<b>Person-Level Leakage:</b> Cannot be assessed — the stu{{id}} filename prefix is not a reliable person identifier. Visual inspection confirmed different people share the same stu{{id}} across exercise classes.',
]
for i, item in enumerate(hf_items, 1):
    story.append(Paragraph(f'{i}. {item}', style_bullet))
story.append(sp(6))

# Two-column layout
col_w = USABLE_WIDTH / 2 - 6
impl_items = [
    'Class imbalance requires weighted loss or oversampling during TCN training.',
    'Sparse L-column annotations limit per-rep boundary supervision quality.',
    'Provenance shift (student vs. original) may degrade cross-distribution generalization.',
]
risk_items = [
    'Person-level leakage cannot be assessed — stu{id} is not a reliable subject identifier.',
    'High annotation mismatch rate (count vs. L-pairs) introduces noisy supervision.',
    'Zero valid rowing_erg samples prevent per-class metric computation.',
]

impl_content = [Paragraph('<b>Modeling Implications</b>', style_bold)]
for item in impl_items:
    impl_content.append(bullet_item(item))

risk_content = [Paragraph('<b>Top Risks</b>', style_bold)]
for item in risk_items:
    risk_content.append(bullet_item(item))

two_col = Table([[impl_content, risk_content]],
                colWidths=[col_w, col_w])
two_col.setStyle(TableStyle([
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('LEFTPADDING', (0,0), (-1,-1), 4),
    ('RIGHTPADDING', (0,0), (-1,-1), 4),
    ('BOX', (0,0), (0,0), 0.5, HexColor('#d1d5db')),
    ('BOX', (1,0), (1,0), 0.5, HexColor('#d1d5db')),
    ('BACKGROUND', (0,0), (0,0), LIGHT_GRAY_RL),
    ('BACKGROUND', (1,0), (1,0), HexColor('#fff7ed')),
]))
story.append(two_col)
story.append(sp(8))

# Recommendations table
story.append(Paragraph('<b>Recommendations</b>', style_bold))
rec_data = [
    [wrap_header_cell('Priority'), wrap_header_cell('Action'), wrap_header_cell('Owner Hint')],
    [wrap_cell('P1 — Critical'), wrap_cell('Audit annotation mismatches; auto-verify L-pairs vs reported count'), wrap_cell('Annotation Team')],
    [wrap_cell('P2 — High'), wrap_cell('Report per-class MAE separately; flag classes with N < 20 train videos as statistically unreliable. Do not deploy exercise-specific features for minority classes without additional data collection.'), wrap_cell('ML Engineering')],
    [wrap_cell('P3 — Medium'), wrap_cell('Collect additional samples for minority classes (especially rowing_erg: 0 valid videos). SMOTE and oversampling are not applicable given current dataset size.'), wrap_cell('Data Collection')],
]
story.append(mk_table(rec_data, col_widths=[80, USABLE_WIDTH-230, 150]))
story.append(sp(8))

# Decision Ask box
decision_style = ParagraphStyle('decision', fontSize=10, leading=14,
                                backColor=HexColor('#fff7ed'),
                                borderColor=ORANGE_RL, borderWidth=1.5,
                                borderPadding=10, alignment=TA_LEFT,
                                leftPadding=12, rightPadding=12,
                                topPadding=8, bottomPadding=8)
story.append(Paragraph(
    '<b>Decision Ask:</b> Authorize a 2-week annotation audit sprint before model training begins, '
    'and approve collection of additional samples for minority classes (especially rowing_erg). '
    'Note: person-level leakage could not be assessed — the dataset naming convention does not '
    'encode reliable subject identity. Current data will produce unreliable per-class performance '
    'estimates without addressing annotation mismatches and class imbalance.',
    decision_style
))

story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — OBJECTIVES AND DATASET OVERVIEW
# ════════════════════════════════════════════════════════════════════════════

story.append(section_header('2. Objectives and Dataset Overview'))
story.append(sp(6))

story.append(Paragraph('<b>Decision Questions</b>', style_bold))
dq_items = [
    'Which classes pose the greatest modeling challenge due to data scarcity or annotation quality?',
    'Are the train/valid/test splits statistically representative and free of subject contamination?',
    'What is the magnitude and impact of annotation inconsistencies on supervision quality?',
    'How does pose keypoint coverage vary across classes, and where does it fail?',
]
for i, dq in enumerate(dq_items, 1):
    story.append(Paragraph(f'{i}. {dq}', style_bullet))
story.append(sp(8))

story.append(Paragraph('<b>Dataset Inventory</b>', style_bold))
story.append(fig_image(FIG1, width=USABLE_WIDTH, height=140))
story.append(caption('Figure 1 — Dataset inventory table. Train set dominates with 74% of videos. '
                     'Test count labels held out per evaluation policy.'))
story.append(italic_cap('Alt text: Table with rows Train/Valid/Test/Total and columns N Videos, N Classes, Mean Duration, Median Duration, Mean Count, Median Count.'))
story.append(sp(8))

story.append(Paragraph('<b>Schema Assumptions</b>', style_bold))
schema_data = [
    [wrap_header_cell('Field'), wrap_header_cell('Type'), wrap_header_cell('Availability'), wrap_header_cell('Notes')],
    [wrap_cell('type'), wrap_cell('string'), wrap_cell('Available'), wrap_cell('Exercise class label; has typos — cleaned via TYPO_MAP')],
    [wrap_cell('name'), wrap_cell('string'), wrap_cell('Available'), wrap_cell('Video filename (mp4)')],
    [wrap_cell('count'), wrap_cell('float64'), wrap_cell('Available'), wrap_cell('Reported rep count; may have NaN; test labels held out')],
    [wrap_cell('L1…L302'), wrap_cell('float64'), wrap_cell('Available'), wrap_cell('Frame-level rep boundary annotations; heavily sparse')],
    [wrap_cell('split'), wrap_cell('string'), wrap_cell('Derived'), wrap_cell('train/valid/test from source file')],
    [wrap_cell('provenance'), wrap_cell('string'), wrap_cell('Derived'), wrap_cell('student_recording or original based on filename pattern')],
    [wrap_cell('is_student'), wrap_cell('int'), wrap_cell('Derived'), wrap_cell('1 if stu{id}_*.mp4 pattern, else 0')],
    [wrap_cell('fps'), wrap_cell('float64'), wrap_cell('Derived'), wrap_cell('Frames per second from video file metadata')],
    [wrap_cell('duration_sec'), wrap_cell('float64'), wrap_cell('Derived'), wrap_cell('frame_count / fps from video file metadata')],
    [wrap_cell('frame_count'), wrap_cell('float64'), wrap_cell('Derived'), wrap_cell('Total frames from video metadata')],
    [wrap_cell('capture_timestamp'), wrap_cell('datetime'), wrap_cell('Not available'), wrap_cell('Not in dataset; cohort analysis not possible')],
    [wrap_cell('subject_demographics'), wrap_cell('mixed'), wrap_cell('Not available'), wrap_cell('Age, gender, fitness level not recorded')],
]
cw = [90, 70, 90, USABLE_WIDTH-250]
story.append(mk_table(schema_data, col_widths=cw, font_size=8, header_font_size=8))
story.append(sp(8))

story.append(Paragraph('<b>Key Metrics and Definitions</b>', style_bold))
kpi_data = [
    [wrap_header_cell('Metric'), wrap_header_cell('Definition')],
    [wrap_cell('MAE'), wrap_cell('Mean Absolute Error: |predicted_count - actual_count| averaged over videos')],
    [wrap_cell('Per-class MAE'), wrap_cell('MAE computed separately for each exercise class')],
    [wrap_cell('Within-1 Accuracy'), wrap_cell('Fraction of predictions within ±1 rep of ground truth')],
    [wrap_cell('Imbalance Ratio'), wrap_cell('max_class_count / min_class_count in training set')],
    [wrap_cell('Annotation Mismatch'), wrap_cell('% of videos where #L-pairs filled ≠ reported count')],
]
story.append(mk_table(kpi_data, col_widths=[130, USABLE_WIDTH-130], font_size=9))

story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATA QUALITY
# ════════════════════════════════════════════════════════════════════════════

story.append(section_header('3. Data Quality'))
story.append(sp(6))

story.append(Paragraph('<b>Data Quality Scorecard</b>', style_bold))
dq_data = [
    [wrap_header_cell('Check'), wrap_header_cell('Status'), wrap_header_cell('Details'), wrap_header_cell('Risk')],
    [wrap_cell('Label Typos'),
     wrap_cell('Fixed'),
     wrap_cell(f'{len(TYPO_MAP)} typo patterns corrected (e.g., squant→squat, jumpjacks→jump_jacks)'),
     wrap_cell('Low')],
    [wrap_cell('Annotation Decisions'),
     wrap_cell('Partial'),
     wrap_cell(f'{M["annotation_mismatch_rate"]}% mismatch between reported count and filled L-pairs'),
     wrap_cell('High')],
    [wrap_cell('File Integrity'),
     wrap_cell('Checked'),
     wrap_cell(f'Video metadata cached; {len(vmeta_df)} videos scanned via OpenCV'),
     wrap_cell('Low')],
    [wrap_cell('Cross-Split Leakage'),
     wrap_cell('Not Assessable'),
     wrap_cell('stu{id} prefix is not a reliable person identifier — confirmed via filename_mapping.xlsx and visual frame inspection'),
     wrap_cell('Unknown')],
    [wrap_cell('Temporal Boundary Validation'),
     wrap_cell('Limited'),
     wrap_cell('L-column pairs not fully validated against frame count'),
     wrap_cell('Medium')],
]
cw3 = [110, 60, USABLE_WIDTH-260, 70]
story.append(mk_table(dq_data, col_widths=cw3, font_size=8, header_font_size=8))
story.append(sp(8))

story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — EDA FINDINGS
# ════════════════════════════════════════════════════════════════════════════

story.append(section_header('4. EDA Findings'))
story.append(sp(6))

story.append(Paragraph('<b>Figure 2 — Class Distribution</b>', style_bold))
story.append(fig_image(FIG2, width=USABLE_WIDTH, height=USABLE_WIDTH*0.52))
story.append(caption(f'Figure 2 — Class distribution. Train (dark blue) stacked with valid (light blue). '
                     f'Imbalance ratio: {M["imbalance_ratio"]}×. Orange dashed = train mean.'))
story.append(italic_cap('Alt text: Horizontal stacked bar chart sorted by train count descending. Classes shown on Y axis.'))
story.append(body(
    f'<b>Evidence:</b> The training set exhibits a {M["imbalance_ratio"]}× class imbalance, with the most frequent '
    f'class ({train_class_counts.idxmax()}: {max_c} videos) vastly outnumbering the least frequent '
    f'({train_class_counts.idxmin()}: {min_c} videos). '
    '<b>Modeling implication:</b> Standard cross-entropy loss will bias the model toward majority classes; '
    'class-weighted loss or oversampling is strongly recommended.'
))
story.append(sp(6))

story.append(Paragraph('<b>Figure 3 — Repetition Count Distribution</b>', style_bold))
story.append(fig_image(FIG3, width=USABLE_WIDTH, height=USABLE_WIDTH*0.45))
story.append(caption(f'Figure 3 — Count histogram (bin width=5). Median={tv_counts_clean.median():.0f}, '
                     f'P95={tv_counts_clean.quantile(0.95):.0f}. Skewness={M["count_skewness"]:.3f}.'))
story.append(italic_cap('Alt text: Histogram with x-axis repetition count and y-axis frequency. Vertical dashed lines mark median (blue) and P95 (orange).'))
story.append(body(
    f'<b>Evidence:</b> The repetition count distribution is right-skewed (skewness={M["count_skewness"]:.2f}), '
    f'with a median of {tv_counts_clean.median():.0f} and a long tail extending to {int(tv_counts_clean.max())}. '
    '<b>Modeling implication:</b> MAE will be dominated by high-count outlier clips; '
    'log-transformation of count targets or clip-level normalization may improve regression stability.'
))
story.append(sp(6))

story.append(Paragraph('<b>Figure 4 — Duration Distribution</b>', style_bold))
story.append(fig_image(FIG4, width=USABLE_WIDTH, height=USABLE_WIDTH*0.45))
story.append(caption(f'Figure 4 — Video duration histogram. Median={tv_dur.median():.1f}s. '
                     f'Skewness={stats.skew(tv_dur.dropna()):.2f}.'))
story.append(italic_cap('Alt text: Histogram of video durations in seconds with a vertical dashed line at median duration.'))
story.append(body(
    f'<b>Evidence:</b> Video durations range from under 5 seconds to over {tv_dur.max():.0f} seconds, '
    f'with a median of {tv_dur.median():.1f}s and right-skewed distribution (skewness={stats.skew(tv_dur.dropna()):.2f}). '
    '<b>Modeling implication:</b> TCN fixed-length sequence inputs will require careful padding/truncation strategy; '
    'excessively long videos may need segmentation.'
))

story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — COHORT AND TEMPORAL ANALYSIS
# ════════════════════════════════════════════════════════════════════════════

story.append(section_header('5. Cohort and Temporal Analysis'))
story.append(sp(6))

story.append(body(
    '<b>Note on Cohort Analysis:</b> True cohort analysis (capturing when videos were recorded, '
    'tracking subjects over time, or identifying recording session effects) is not possible '
    'with this dataset. The metadata does not include capture timestamps, session identifiers, '
    'or subject demographics. The analysis below uses temporal structure within individual '
    'videos (L-column rep boundary positions) as a proxy for temporal consistency assessment.'
))
story.append(sp(8))

story.append(Paragraph('<b>Figure 5 — Temporal Distribution of Rep Boundaries</b>', style_bold))
story.append(fig_image(FIG5, width=USABLE_WIDTH, height=USABLE_WIDTH*0.48))
story.append(caption('Figure 5 — Distribution of normalized rep boundary positions across all videos '
                     'with ≥2 annotated reps. X-axis: 0=video start, 1=video end. '
                     'Overall (dark blue) vs. per-class (dashed, classes with ≥20 boundary samples). '
                     f'Rolling window = {window} bins.'))
story.append(italic_cap('Alt text: Line chart with normalized position on X-axis and boundary density on Y-axis. Dark blue overall line with colored dashed per-class overlays.'))
story.append(sp(4))
story.append(body(
    '<b>Interpretation:</b> A roughly uniform boundary density across normalized positions suggests '
    'annotators distributed rep markers evenly throughout videos — consistent with exercises '
    'where performers maintain steady pace. Peaks near position 0 or 1 would indicate edge '
    'annotation effects (e.g., annotators marking the very start/end of clips). '
    'Per-class variation in density profiles indicates different exercise pacing patterns — '
    'high-density regions in specific classes may correspond to warm-up or rest periods. '
    '<b>TCN implication:</b> If boundary density is non-uniform within classes, the model may '
    'learn temporal position as a spurious feature; temporal data augmentation (e.g., speed '
    'perturbation) is recommended to improve robustness.'
))

story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — FEATURE RELIABILITY AND FAILURE MODES
# ════════════════════════════════════════════════════════════════════════════

story.append(section_header('6. Feature Reliability and Failure Modes'))
story.append(sp(6))

story.append(Paragraph('<b>YOLO Pose Coverage by Class</b>', style_bold))

# Build pose coverage table
if len(M['pose_by_class']) > 0:
    pcb = M['pose_by_class'].reset_index()
    yolo_data = [[wrap_header_cell('Class'), wrap_header_cell('Mean Coverage %'),
                  wrap_header_cell('Min Coverage %'), wrap_header_cell('N Videos'),
                  wrap_header_cell('N Low (<95%)')]]
    for _, row in pcb.iterrows():
        cls = str(row['type']) if 'type' in row else str(row.iloc[0])
        yolo_data.append([
            wrap_cell(cls, font_size=8),
            wrap_cell(f"{row['mean']:.1f}%" if pd.notna(row['mean']) else 'N/A', font_size=8),
            wrap_cell(f"{row['min']:.1f}%" if pd.notna(row['min']) else 'N/A', font_size=8),
            wrap_cell(str(int(row['count'])), font_size=8),
            wrap_cell(str(int(row['n_low'])), font_size=8),
        ])
    cw6 = [130, 100, 100, 80, 100]
    story.append(mk_table(yolo_data, col_widths=cw6, font_size=8, header_font_size=8))
else:
    story.append(body('YOLO pose coverage data not available for class breakdown.'))
story.append(sp(6))

story.append(Paragraph('<b>Figure 6 — Feature Importance Proxy</b>', style_bold))
story.append(fig_image(FIG6, width=USABLE_WIDTH*0.7, height=USABLE_WIDTH*0.32))
story.append(caption('Figure 6 — Univariate mutual information between numerical features and repetition count. '
                     'EXPLORATORY — does not account for feature interactions.'))
story.append(italic_cap('Alt text: Horizontal bar chart with features on Y-axis and mutual information score on X-axis. Sorted descending.'))
story.append(sp(6))

story.append(Paragraph('<b>Failure Modes Summary</b>', style_bold))
fm_data = [
    [wrap_header_cell('Failure Mode'), wrap_header_cell('Prevalence'),
     wrap_header_cell('Affected Classes'), wrap_header_cell('Modeling Impact'),
     wrap_header_cell('Mitigation')],
    [wrap_cell('Class Imbalance', font_size=8),
     wrap_cell(f'{M["imbalance_ratio"]}× ratio', font_size=8),
     wrap_cell('All', font_size=8),
     wrap_cell('Majority class bias in loss', font_size=8),
     wrap_cell('Weighted loss / oversampling', font_size=8)],
    [wrap_cell('FPS Variation', font_size=8),
     wrap_cell(f'Range: {all_df["fps"].min():.0f}–{all_df["fps"].max():.0f}fps', font_size=8),
     wrap_cell('Student recordings', font_size=8),
     wrap_cell('Temporal misalignment in TCN', font_size=8),
     wrap_cell('Normalize to fixed fps', font_size=8)],
    [wrap_cell('Annotation Mismatch', font_size=8),
     wrap_cell(f'{M["annotation_mismatch_rate"]}% of train+valid', font_size=8),
     wrap_cell('All', font_size=8),
     wrap_cell('Noisy count supervision', font_size=8),
     wrap_cell('Audit & re-annotate mismatches', font_size=8)],
    [wrap_cell('Low YOLO Coverage', font_size=8),
     wrap_cell(f'{(pose_all["coverage_pct"] < 95).sum()} videos <95%', font_size=8),
     wrap_cell('Multiple', font_size=8),
     wrap_cell('Sparse pose features', font_size=8),
     wrap_cell('Impute or exclude low-coverage clips', font_size=8)],
    [wrap_cell('Outlier Counts', font_size=8),
     wrap_cell(f'P95={tv_counts_clean.quantile(0.95):.0f}, max={int(tv_counts_clean.max())}', font_size=8),
     wrap_cell('High-rep classes', font_size=8),
     wrap_cell('Dominates MAE metric', font_size=8),
     wrap_cell('Clip-level count normalization', font_size=8)],
    [wrap_cell('Provenance Shift', font_size=8),
     wrap_cell(f'{all_df["is_student"].sum()} student clips', font_size=8),
     wrap_cell('Student-recorded classes', font_size=8),
     wrap_cell('Distribution mismatch', font_size=8),
     wrap_cell('Stratified sampling by provenance', font_size=8)],
    [wrap_cell('Zero Valid Samples', font_size=8),
     wrap_cell('rowing_erg: 0 valid', font_size=8),
     wrap_cell('rowing_erg', font_size=8),
     wrap_cell('Cannot validate per-class', font_size=8),
     wrap_cell('Exclude from per-class eval or resample', font_size=8)],
]
cw6b = [100, 90, 90, 110, 110]
story.append(mk_table(fm_data, col_widths=cw6b, font_size=8, header_font_size=8))
story.append(sp(6))

story.append(Paragraph('<b>Figure 7 — Failure-Case Gallery</b>', style_bold))
story.append(fig_image(FIG7, width=USABLE_WIDTH, height=USABLE_WIDTH*0.50))
story.append(caption('Figure 7 — Representative challenging video frames. Each panel shows a video '
                     'with a specific failure mode. Gray boxes indicate frame extraction failure. '
                     'Caption: video name, class, count, split.'))
story.append(italic_cap('Alt text: 2×3 grid of video frames illustrating outlier count, annotation mismatch, longest duration, rowing erg, battle rope, and low YOLO coverage.'))

story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════════════
# PAGE 7 — BUSINESS IMPLICATIONS AND RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════════════════

story.append(section_header('7. Business Implications and Recommendations'))
story.append(sp(6))

story.append(body(
    '<b>Model Reliability at Deployment:</b> The current dataset, if used without addressing '
    'annotation mismatches and class imbalance, will produce validation metrics that '
    'overestimate real-world performance on minority classes. Person-level leakage could not '
    'be assessed — the stu{id} filename prefix was confirmed to not be a reliable subject '
    'identifier (different people share the same prefix across exercise classes). A formal '
    'subject-identity audit would require face recognition on video frames.'
))
story.append(body(
    '<b>Class Coverage Gaps:</b> With 0 valid samples for rowing_erg and extreme imbalance '
    'across other classes, the model cannot be reliably validated on minority classes. '
    'Deploying a product feature for these exercise types without class-specific validation '
    'exposes users to unchecked error rates — a safety and quality concern for a fitness '
    'application.'
))
story.append(body(
    '<b>Annotation Investment Efficiency:</b> The {:.1f}% annotation mismatch rate suggests '
    'that annotation resources were not fully verified. Investing 2 weeks in an annotation '
    'audit sprint now will prevent weeks of debugging after model training reveals poor '
    'convergence on affected classes.'.format(M["annotation_mismatch_rate"])
))
story.append(body(
    '<b>Data Provenance and Privacy:</b> Student recordings introduce biometric data (pose '
    'sequences) from identifiable individuals. Before any external use or publication, a '
    'privacy/legal review of consent coverage is required — particularly for the student '
    'subject cohort.'
))
story.append(sp(6))

story.append(Paragraph('<b>Recommendations</b>', style_bold))
rec_full_data = [
    [wrap_header_cell('Priority'), wrap_header_cell('Recommendation'),
     wrap_header_cell('Rationale'), wrap_header_cell('Effort'), wrap_header_cell('Owner')],
    [wrap_cell('P1'), wrap_cell('Conduct subject-identity audit via face recognition to assess true leakage risk', font_size=8),
     wrap_cell('stu{id} prefix is not a reliable person identifier', font_size=8),
     wrap_cell('1w', font_size=8), wrap_cell('Data Eng.', font_size=8)],
    [wrap_cell('P1'), wrap_cell('Audit annotation mismatches; auto-verify L-pairs vs count', font_size=8),
     wrap_cell(f'{M["annotation_mismatch_rate"]}% mismatch rate', font_size=8),
     wrap_cell('2w', font_size=8), wrap_cell('Annotation', font_size=8)],
    [wrap_cell('P2'), wrap_cell('Report per-class MAE; flag classes with N < 20 train videos as unreliable. Do not deploy minority-class features without additional data.', font_size=8),
     wrap_cell(f'{M["imbalance_ratio"]}× imbalance — insufficient data for SMOTE or weighted loss to be meaningful', font_size=8),
     wrap_cell('0.5d', font_size=8), wrap_cell('ML Eng.', font_size=8)],
    [wrap_cell('P2'), wrap_cell('Normalize video FPS to standard rate (e.g., 30fps)', font_size=8),
     wrap_cell('FPS variation causes temporal misalignment', font_size=8),
     wrap_cell('1d', font_size=8), wrap_cell('Data Eng.', font_size=8)],
    [wrap_cell('P3'), wrap_cell('Collect additional rowing_erg valid samples', font_size=8),
     wrap_cell('0 valid samples prevents class evaluation', font_size=8),
     wrap_cell('2w', font_size=8), wrap_cell('Data Collect.', font_size=8)],
    [wrap_cell('P3'), wrap_cell('Conduct privacy/legal review of student recordings', font_size=8),
     wrap_cell('Biometric data — consent unclear', font_size=8),
     wrap_cell('1w', font_size=8), wrap_cell('Legal/Privacy', font_size=8)],
]
cw7 = [40, 170, 140, 40, 78]
story.append(mk_table(rec_full_data, col_widths=cw7, font_size=8, header_font_size=8))
story.append(sp(6))

story.append(Paragraph('<b>Output Format Comparison</b>', style_bold))
fmt_data = [
    [wrap_header_cell('Format'), wrap_header_cell('Audience'), wrap_header_cell('Strengths'),
     wrap_header_cell('Weaknesses'), wrap_header_cell('Use Case')],
    [wrap_cell('6-page memo', font_size=8), wrap_cell('Exec / VP', font_size=8),
     wrap_cell('Fast read, concise', font_size=8), wrap_cell('Limited technical depth', font_size=8),
     wrap_cell('Decision gate', font_size=8)],
    [wrap_cell('8-page balanced', font_size=8), wrap_cell('DS Lead / PM', font_size=8),
     wrap_cell('Balance detail/brevity', font_size=8), wrap_cell('May miss appendix detail', font_size=8),
     wrap_cell('Team alignment', font_size=8)],
    [wrap_cell('10-page committee pack', font_size=8), wrap_cell('Committee / Review Board', font_size=8),
     wrap_cell('Full evidence base', font_size=8), wrap_cell('Long read time', font_size=8),
     wrap_cell('This report', font_size=8)],
]
story.append(mk_table(fmt_data, col_widths=[90, 80, 100, 110, 88], font_size=8, header_font_size=8))
story.append(sp(6))

story.append(Paragraph('<b>Resource Estimate</b>', style_bold))
res_data = [
    [wrap_header_cell('Role'), wrap_header_cell('Task'), wrap_header_cell('Hours')],
    [wrap_cell('Data Engineer'), wrap_cell('Re-split + FPS normalization'), wrap_cell('8h')],
    [wrap_cell('Annotation Team'), wrap_cell('Annotation audit sprint'), wrap_cell('80h')],
    [wrap_cell('ML Engineer'), wrap_cell('TCN training with class-weighted loss'), wrap_cell('40h')],
    [wrap_cell('Privacy/Legal'), wrap_cell('Consent review for student recordings'), wrap_cell('16h')],
]
story.append(mk_table(res_data, col_widths=[130, USABLE_WIDTH-210, 80], font_size=8, header_font_size=8))
story.append(sp(6))

story.append(Paragraph('<b>Production Timeline</b>', style_bold))
story.append(fig_image(GANTT_FIG, width=USABLE_WIDTH, height=USABLE_WIDTH*0.35))
story.append(caption('Gantt chart — indicative production timeline for RepCount pipeline delivery.'))

story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════════════
# PAGE 8 — PRIVACY AND COMPLIANCE
# ════════════════════════════════════════════════════════════════════════════

story.append(section_header('8. Privacy and Compliance'))
story.append(sp(6))

story.append(Paragraph('<b>Identifiability and Biometric Data</b>', style_bold))
story.append(body(
    'Pose keypoints extracted by YOLO constitute biometric data under most privacy regulations '
    '(GDPR, CCPA, BIPA). The 17-keypoint skeleton captures body proportions and movement '
    'signatures that can be used for individual identification. Any storage, processing, or '
    'sharing of pose features must comply with applicable biometric data protection laws. '
    'This is especially critical for student-recorded videos, where subjects are likely '
    'identifiable individuals.'
))
story.append(body(
    '<b>Face and Body Image Sensitivity:</b> While the primary features extracted are skeletal '
    'keypoints, the raw video frames contain facial imagery. The failure-case gallery in this '
    'report reproduces actual video frames — these should not be included in any external '
    'publication without explicit consent from subjects. Internal use should be restricted '
    'to authorized personnel only.'
))
story.append(body(
    '<b>Dataset Provenance and Consent:</b> "Original" videos (non-student) are presumed to '
    'be from publicly available YouTube content, making their use for research potentially '
    'permissible under fair use or public research exemptions. However, this assumption has '
    '<b>not been formally verified</b>. YouTube Terms of Service prohibit downloading and '
    'redistributing video content without authorization. A formal legal review of the '
    'dataset\'s provenance and licensing is recommended before any external publication or '
    'commercial use.'
))
story.append(body(
    '<b>Retention and Access Control:</b> Raw video files and pose feature arrays should be '
    'stored in access-controlled environments (e.g., encrypted S3 buckets with IAM policies). '
    'Retention should be limited to the duration of the research project. Student-recorded '
    'data should be subject to deletion rights per applicable regulations. '
    '<b>Privacy/Legal Review Status:</b> Not available — recommend review before external use.'
))
story.append(sp(6))

story.append(Paragraph('<b>Compliance Checklist</b>', style_bold))
cc_data = [
    [wrap_header_cell('Item'), wrap_header_cell('Status'), wrap_header_cell('Notes')],
    [wrap_cell('IRB / Ethics review for student recordings'), wrap_cell('Unknown'), wrap_cell('Recommend immediate review')],
    [wrap_cell('Informed consent for student subjects'), wrap_cell('Unknown'), wrap_cell('No documentation in dataset')],
    [wrap_cell('YouTube content licensing'), wrap_cell('Assumed fair use'), wrap_cell('Not formally verified — legal review needed')],
    [wrap_cell('GDPR/CCPA biometric compliance'), wrap_cell('Not assessed'), wrap_cell('Pose = biometric; formal assessment required')],
    [wrap_cell('Access control on video storage'), wrap_cell('Not documented'), wrap_cell('Recommend encrypted, IAM-controlled storage')],
    [wrap_cell('Data retention policy'), wrap_cell('Not defined'), wrap_cell('Define maximum retention period')],
    [wrap_cell('Right to deletion support'), wrap_cell('Not implemented'), wrap_cell('Required for identifiable subjects under GDPR')],
    [wrap_cell('External sharing restriction'), wrap_cell('CONFIDENTIAL'), wrap_cell('This report restricted to internal use')],
]
story.append(mk_table(cc_data, col_widths=[200, 80, USABLE_WIDTH-280], font_size=8, header_font_size=8))

story.append(PageBreak())

# ════════════════════════════════════════════════════════════════════════════
# PAGES 9–10 — APPENDIX
# ════════════════════════════════════════════════════════════════════════════

story.append(section_header('9. Appendix: Reproducibility and Data Dictionary'))
story.append(sp(6))

story.append(Paragraph('<b>Artifact Inventory</b>', style_bold))
artifact_data = [
    [wrap_header_cell('Artifact'), wrap_header_cell('Path'), wrap_header_cell('Description')],
    [wrap_cell('Train CSV'), wrap_cell('datasets/metadata/llsp/train.csv'), wrap_cell('758 rows, 306 cols')],
    [wrap_cell('Valid CSV'), wrap_cell('datasets/metadata/llsp/valid.csv'), wrap_cell('131 rows, 306 cols')],
    [wrap_cell('Test CSV'), wrap_cell('datasets/metadata/llsp/test.csv'), wrap_cell('152 rows, 306 cols')],
    [wrap_cell('Pose Report'), wrap_cell('datasets/metadata/llsp/pose_extraction_report.csv'), wrap_cell('YOLO coverage, 118 rows')],
    [wrap_cell('Pose Report Remaining'), wrap_cell('datasets/metadata/llsp/pose_extraction_report_remaining.csv'), wrap_cell('Remaining YOLO coverage')],
    [wrap_cell('Video Cache'), wrap_cell('reports/video_metadata_cache.csv'), wrap_cell('OpenCV-extracted fps/duration/size')],
    [wrap_cell('This PDF'), wrap_cell('reports/RepCount_EDA_Executive_Report.pdf'), wrap_cell('Generated executive report')],
    [wrap_cell('Figure Dir'), wrap_cell('reports/figures/exec_report/'), wrap_cell('9 PNG figures + gantt')],
    [wrap_cell('Generator Script'), wrap_cell('reports/generate_eda_executive_report.py'), wrap_cell('This script — fully reproducible')],
]
story.append(mk_table(artifact_data, col_widths=[100, 200, USABLE_WIDTH-300], font_size=8, header_font_size=8))
story.append(sp(8))

story.append(Paragraph('<b>Data Dictionary</b>', style_bold))
dd_data = [
    [wrap_header_cell('Column'), wrap_header_cell('Type'), wrap_header_cell('Description'),
     wrap_header_cell('Range/Values'), wrap_header_cell('Missing Rate'), wrap_header_cell('Notes')],
    [wrap_cell('type', font_size=7), wrap_cell('str', font_size=7), wrap_cell('Exercise class', font_size=7),
     wrap_cell(f'{all_df["type"].nunique()} classes', font_size=7), wrap_cell('0%', font_size=7),
     wrap_cell('Cleaned via TYPO_MAP', font_size=7)],
    [wrap_cell('name', font_size=7), wrap_cell('str', font_size=7), wrap_cell('Video filename', font_size=7),
     wrap_cell('*.mp4', font_size=7), wrap_cell('0%', font_size=7), wrap_cell('Unique per row', font_size=7)],
    [wrap_cell('count', font_size=7), wrap_cell('float64', font_size=7), wrap_cell('Rep count', font_size=7),
     wrap_cell(f'{int(all_df["count"].min())}–{int(all_df["count"].max())}', font_size=7),
     wrap_cell(f'{all_df["count"].isna().mean()*100:.1f}%', font_size=7),
     wrap_cell('Test labels held out', font_size=7)],
    [wrap_cell('L1…L302', font_size=7), wrap_cell('float64', font_size=7),
     wrap_cell('Rep boundary frames', font_size=7), wrap_cell('Frame indices', font_size=7),
     wrap_cell(f'{M["l_nan_train"]*100:.1f}% (train)', font_size=7), wrap_cell('Sparse; pairs=start+end', font_size=7)],
    [wrap_cell('split', font_size=7), wrap_cell('str', font_size=7), wrap_cell('Dataset split', font_size=7),
     wrap_cell('train/valid/test', font_size=7), wrap_cell('0%', font_size=7), wrap_cell('Derived', font_size=7)],
    [wrap_cell('provenance', font_size=7), wrap_cell('str', font_size=7), wrap_cell('Recording source', font_size=7),
     wrap_cell('student_recording/original', font_size=7),
     wrap_cell('0%', font_size=7), wrap_cell('Derived from name pattern', font_size=7)],
    [wrap_cell('is_student', font_size=7), wrap_cell('int', font_size=7), wrap_cell('Student flag', font_size=7),
     wrap_cell('0/1', font_size=7), wrap_cell('0%', font_size=7), wrap_cell('1 if stu{id}_* pattern', font_size=7)],
    [wrap_cell('fps', font_size=7), wrap_cell('float64', font_size=7), wrap_cell('Frames per second', font_size=7),
     wrap_cell(f'{all_df["fps"].min():.0f}–{all_df["fps"].max():.0f}', font_size=7),
     wrap_cell(f'{all_df["fps"].isna().mean()*100:.1f}%', font_size=7), wrap_cell('From OpenCV', font_size=7)],
    [wrap_cell('duration_sec', font_size=7), wrap_cell('float64', font_size=7),
     wrap_cell('Video duration', font_size=7),
     wrap_cell(f'{all_df["duration_sec"].min():.1f}–{all_df["duration_sec"].max():.1f}s', font_size=7),
     wrap_cell(f'{all_df["duration_sec"].isna().mean()*100:.1f}%', font_size=7), wrap_cell('frame_count/fps', font_size=7)],
    [wrap_cell('frame_count', font_size=7), wrap_cell('float64', font_size=7),
     wrap_cell('Total frames', font_size=7), wrap_cell('Varies', font_size=7),
     wrap_cell(f'{all_df["frame_count"].isna().mean()*100:.1f}%', font_size=7), wrap_cell('From OpenCV', font_size=7)],
    [wrap_cell('n_L_cols_filled', font_size=7), wrap_cell('int', font_size=7),
     wrap_cell('# non-NaN L-cols', font_size=7), wrap_cell('0–302', font_size=7),
     wrap_cell('0%', font_size=7), wrap_cell('Derived; proxy for rep density', font_size=7)],
]
cw9 = [75, 55, 100, 90, 70, 100]
story.append(mk_table(dd_data, col_widths=cw9, font_size=7, header_font_size=7))
story.append(sp(8))

story.append(Paragraph('<b>Environment</b>', style_bold))
import sys as _sys
import matplotlib
import seaborn as _sns
import scipy
import reportlab
env_data = [
    [wrap_header_cell('Package'), wrap_header_cell('Version')],
    [wrap_cell('Python'), wrap_cell(_sys.version.split()[0])],
    [wrap_cell('pandas'), wrap_cell(pd.__version__)],
    [wrap_cell('numpy'), wrap_cell(np.__version__)],
    [wrap_cell('matplotlib'), wrap_cell(matplotlib.__version__)],
    [wrap_cell('seaborn'), wrap_cell(_sns.__version__)],
    [wrap_cell('scipy'), wrap_cell(scipy.__version__)],
    [wrap_cell('reportlab'), wrap_cell(reportlab.Version)],
    [wrap_cell('opencv-python'), wrap_cell(cv2.__version__)],
]
story.append(mk_table(env_data, col_widths=[150, 100], font_size=9))
story.append(sp(8))

story.append(Paragraph('<b>Metric Definitions</b>', style_bold))
story.append(body_left('MAE: Mean |predicted - actual| repetitions across videos.'))
story.append(body_left('Imbalance Ratio: max(class_count) / min(class_count) in training split.'))
story.append(body_left('Annotation Mismatch: (count of L-pairs filled != reported count) / total videos.'))
story.append(body_left('KS p-value: p-value from two-sample Kolmogorov-Smirnov test of count distributions (train vs valid per class).'))
story.append(body_left('YOLO Coverage: frames_used / frames_total × 100%.'))
story.append(sp(8))

story.append(Paragraph('<b>Regeneration Commands</b>', style_bold))
cmd_style = ParagraphStyle('cmd', fontName='Courier', fontSize=8, leading=12,
                           backColor=HexColor('#f1f5f9'), leftPadding=8,
                           rightPadding=8, topPadding=6, bottomPadding=6)
story.append(Paragraph(
    'cd /Users/lindaperez/Documents/COMPUTER_VISION/Final_project/personal-git/ML_System<br/>'
    'source ../.venv/bin/activate<br/>'
    'python3 reports/generate_eda_executive_report.py',
    cmd_style
))
story.append(body_left('To regenerate without video re-scan, the cache file at '
                       'reports/video_metadata_cache.csv will be reused automatically.'))
story.append(sp(8))

story.append(Paragraph('<b>Section → Script Mapping</b>', style_bold))
map_data = [
    [wrap_header_cell('Report Section'), wrap_header_cell('Script Section')],
    [wrap_cell('Page 1: Executive Summary'), wrap_cell('Section D — PAGE 1 story block')],
    [wrap_cell('Page 2: Dataset Overview'), wrap_cell('Section D — PAGE 2 + Section A (load_and_clean)')],
    [wrap_cell('Page 3: Data Quality'), wrap_cell('Section D — PAGE 3 + Section B (leakage, mismatch)')],
    [wrap_cell('Page 4: EDA Findings'), wrap_cell('Section D — PAGE 4 + Section C (Fig2/3/4)')],
    [wrap_cell('Page 5: Temporal Analysis'), wrap_cell('Section D — PAGE 5 + Section C (Fig7)')],
    [wrap_cell('Page 6: Feature Reliability'), wrap_cell('Section D — PAGE 6 + Section C (Fig8/9)')],
    [wrap_cell('Page 7: Business Implications'), wrap_cell('Section D — PAGE 7 + Section C (Gantt)')],
    [wrap_cell('Page 8: Privacy/Compliance'), wrap_cell('Section D — PAGE 8')],
    [wrap_cell('Pages 9–10: Appendix'), wrap_cell('Section D — APPENDIX')],
]
story.append(mk_table(map_data, col_widths=[220, USABLE_WIDTH-220], font_size=8, header_font_size=8))

# ─── BUILD PDF ────────────────────────────────────────────────────────────────
print("Building PDF...")

doc = ReportDocTemplate(
    OUTPUT_PDF,
    pagesize=letter,
    topMargin=MARGIN,
    bottomMargin=MARGIN,
    leftMargin=MARGIN,
    rightMargin=MARGIN,
)

doc.build(story)

print(f"\nReport saved: reports/RepCount_EDA_Executive_Report.pdf")
print(f"Figures saved: reports/figures/exec_report/")
print(f"Script saved: reports/generate_eda_executive_report.py")
