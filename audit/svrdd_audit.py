# ======================================================================
# SVRDD data audit (used in the progress report, Section IV-A)
#
# Usage in Colab (CPU is enough), after notebook sections 1, 2 and 2b:
#     %run -i {PROJECT}/audit/svrdd_audit.py
# (-i lets the script see DRIVE_DIR from the notebook.)
#
# Checks: label validity, per-class/split/district counts, box sizes
# (COCO bins at the 640 px training input), cross-split near-duplicates
# (64-bit dHash). Writes CSV/LaTeX tables and PNG figures to <DRIVE_DIR>/audit.
# ======================================================================
import os, glob, json, collections, itertools
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

YOLO_ROOT = globals().get('YOLO_ROOT', '/content/datasets/svrdd/SVRDD_YOLO')
SPLIT_DIR = globals().get('SPLIT_DIR', '/content/datasets/svrdd/splits')     # written by 2b
OUT       = globals().get('AUDIT_OUT', f'{DRIVE_DIR}/audit' if os.path.isdir(DRIVE_DIR) else '/content/audit')
INPUT_PX  = 640                       # training input size (what the model sees)
HASH_THR  = 4                         # dHash Hamming distance counted as "near-duplicate"
NAMES = ['longitudinal_crack', 'transverse_crack', 'alligator_crack', 'pothole',
         'manhole_cover', 'longitudinal_patch', 'transverse_patch']
SHORT = ['L-crack', 'T-crack', 'Alligator', 'Pothole', 'Manhole', 'L-patch', 'T-patch']
SPLITS = ['train', 'val', 'test']
os.makedirs(OUT, exist_ok=True)
print('Saving audit outputs to:', OUT)

# ---------- 1) Read every image + label of every split ----------------
rows, issues, imgs = [], [], []
for split in SPLITS:
    paths = [l.strip() for l in open(f'{SPLIT_DIR}/{split}.txt') if l.strip()]
    for p in paths:
        district = p.split('/')[-3]
        lbl = p.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
        try:
            W, H = Image.open(p).size                       # reads only the header
        except Exception as e:
            issues.append(dict(split=split, image=p, issue=f'unreadable image: {e}')); continue
        n_box = 0
        if not os.path.exists(lbl):
            issues.append(dict(split=split, image=p, issue='missing label file'))
        else:
            seen = set()
            for k, line in enumerate(open(lbl)):
                f = line.split()
                if not f: continue
                if len(f) != 5:
                    issues.append(dict(split=split, image=p, issue=f'line {k}: {len(f)} fields')); continue
                c, x, y, w, h = int(float(f[0])), *map(float, f[1:])
                prob = []
                if not 0 <= c < len(NAMES): prob.append(f'class id {c} out of range')
                if w <= 0 or h <= 0:        prob.append('zero/negative size')
                if not (0 <= x <= 1 and 0 <= y <= 1): prob.append('center outside image')
                if x - w/2 < -1e-3 or x + w/2 > 1+1e-3 or y - h/2 < -1e-3 or y + h/2 > 1+1e-3:
                    prob.append('box extends outside image')
                key = (c, round(x, 4), round(y, 4), round(w, 4), round(h, 4))
                if key in seen: prob.append('duplicate box')
                seen.add(key)
                for pr in prob:
                    issues.append(dict(split=split, image=p, issue=f'line {k}: {pr}'))
                if 0 <= c < len(NAMES):
                    rows.append(dict(split=split, district=district, image=os.path.basename(p), cls=c,
                                     x=x, y=y, w=w, h=h, W=W, H=H))
                    n_box += 1
        imgs.append(dict(split=split, district=district, path=p, image=os.path.basename(p), W=W, H=H, n_box=n_box))

boxes, imgs, issues = pd.DataFrame(rows), pd.DataFrame(imgs), pd.DataFrame(issues, columns=['split', 'image', 'issue'])
boxes['name'] = boxes.cls.map(dict(enumerate(SHORT)))
boxes['area_frac'] = boxes.w * boxes.h                                   # share of the image
boxes['side_px'] = np.sqrt(boxes.area_frac) * INPUT_PX                   # equivalent square side at 640 px
boxes['size_bin'] = pd.cut(boxes.side_px**2, [0, 32**2, 96**2, np.inf], labels=['small', 'medium', 'large'])
print(f'Images: {len(imgs)}   boxes: {len(boxes)}   images without boxes: {(imgs.n_box == 0).sum()}')
print('Image sizes:', imgs.groupby(['W', 'H']).size().to_dict())
print(f'Label issues: {len(issues)}')
if len(issues):
    print(issues.issue.str.replace(r'line \d+: ', '', regex=True).value_counts().to_string())
issues.to_csv(f'{OUT}/label_issues.csv', index=False)

# ---------- 2) Class counts per split ---------------------------------
cnt = boxes.pivot_table(index='name', columns='split', values='cls', aggfunc='size', fill_value=0)
cnt = cnt.reindex(SHORT)[SPLITS]
cnt['total'] = cnt.sum(axis=1)
cnt['share_%'] = (100 * cnt.total / cnt.total.sum()).round(1)
cnt['imgs_with_class'] = boxes.groupby('name').image.nunique().reindex(SHORT)
cnt['train_imbalance'] = (cnt.train.max() / cnt.train).round(2)    # most frequent / this class
print('\n=== Boxes per class and split ===\n', cnt.to_string())

# ---------- 3) Box size per class -------------------------------------
size = boxes.groupby('name').agg(median_area_pct=('area_frac', lambda s: 100 * s.median()),
                                 median_side_px=('side_px', 'median'),
                                 pct_small=('size_bin', lambda s: 100 * (s == 'small').mean()),
                                 pct_medium=('size_bin', lambda s: 100 * (s == 'medium').mean()),
                                 pct_large=('size_bin', lambda s: 100 * (s == 'large').mean()),
                                 median_y_center=('y', 'median')).reindex(SHORT).round(2)
print(f'\n=== Box size per class (side_px = sqrt(area) at {INPUT_PX} px input; COCO bins) ===\n', size.to_string())

per_class = cnt.join(size)
per_class.to_csv(f'{OUT}/per_class_stats.csv')
with open(f'{OUT}/per_class_stats.tex', 'w') as f:
    f.write(per_class[['train', 'val', 'test', 'share_%', 'median_area_pct', 'median_side_px', 'pct_small']]
            .rename(columns={'share_%': 'share (\\%)', 'median_area_pct': 'med. area (\\%)',
                             'median_side_px': 'med. side (px)', 'pct_small': 'small (\\%)'})
            .to_latex(float_format='%.2f', caption='Per-class statistics of SVRDD (authors\' split).',
                      label='tab:classstats'))

# ---------- 4) District x class ---------------------------------------
dist = boxes.pivot_table(index='district', columns='name', values='cls', aggfunc='size', fill_value=0)[SHORT]
dist['images'] = imgs.groupby('district').size()
dist.to_csv(f'{OUT}/district_by_class.csv')
print('\n=== Boxes per district and class ===\n', dist.to_string())

# ---------- 5) Panorama / near-duplicate leakage across splits ---------
imgs['pano_id'] = imgs.image.str.split('_').str[0]
pano = imgs.groupby('pano_id').split.agg(lambda s: tuple(sorted(set(s))))
cross_pano = pano[pano.map(len) > 1]
print(f'\nFile-name prefixes (panorama id candidates): {pano.size} unique for {len(imgs)} images; '
      f'{len(cross_pano)} prefixes appear in more than one split')

def dhash(path, n=8):
    im = Image.open(path); im.draft('L', (64, 64))          # fast JPEG decode at low resolution
    g = np.asarray(im.convert('L').resize((n + 1, n), Image.BILINEAR), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(''.join('1' if b else '0' for b in bits), 2)

print('Computing perceptual hashes (a few minutes)...')
imgs['dhash'] = [dhash(p) for p in imgs.path]
h = np.array(imgs.dhash.tolist(), dtype=np.uint64)
def popcount(a):
    a = a.copy(); c = np.zeros(a.shape, dtype=np.uint8)
    while a.any():
        c += (a & np.uint64(1)).astype(np.uint8); a >>= np.uint64(1)
    return c
pairs = []
tr = np.where(imgs.split.values == 'train')[0]
for s in ['val', 'test']:
    for i in np.where(imgs.split.values == s)[0]:
        d = popcount(h[tr] ^ h[i])
        for j in tr[d <= HASH_THR]:
            pairs.append(dict(eval_split=s, eval_image=imgs.path[i], other_split='train', other_image=imgs.path[j],
                              hamming=int(d[tr == j][0])))
va, te = np.where(imgs.split.values == 'val')[0], np.where(imgs.split.values == 'test')[0]
for i in va:
    d = popcount(h[te] ^ h[i])
    for j in te[d <= HASH_THR]:
        pairs.append(dict(eval_split='val', eval_image=imgs.path[i], other_split='test', other_image=imgs.path[j],
                          hamming=int(d[te == j][0])))
pairs = pd.DataFrame(pairs, columns=['eval_split', 'eval_image', 'other_split', 'other_image', 'hamming'])
pairs.to_csv(f'{OUT}/near_duplicate_pairs.csv', index=False)
print(f'Near-duplicate pairs across splits (dHash distance <= {HASH_THR}):',
      (pairs.eval_split + '-' + pairs.other_split).value_counts().to_dict() if len(pairs) else 0,
      '| exact (distance 0):', int((pairs.hamming == 0).sum()) if len(pairs) else 0)

# ---------- 6) Figures -------------------------------------------------
plt.rcParams.update({'font.size': 9, 'figure.dpi': 150})
colors = {'train': '#4C72B0', 'val': '#DD8452', 'test': '#55A868'}

fig, ax = plt.subplots(figsize=(6, 3))
xs = np.arange(len(SHORT)); bw = 0.27
for k, s in enumerate(SPLITS):
    ax.bar(xs + (k - 1) * bw, cnt[s], bw, label=s, color=colors[s])
ax.set_xticks(xs, SHORT, rotation=30, ha='right'); ax.set_ylabel('boxes'); ax.legend(frameon=False)
ax.set_title('Instances per class and split'); fig.tight_layout(); fig.savefig(f'{OUT}/fig_class_counts.png'); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 3))
data = [boxes.loc[boxes.name == n, 'side_px'] for n in SHORT]
ax.boxplot(data, showfliers=False); ax.set_yscale('log')
ax.set_xticks(range(1, len(SHORT) + 1), SHORT, rotation=30, ha='right')
for t, lab in [(32, 'COCO small/medium'), (96, 'medium/large')]:
    ax.axhline(t, ls='--', lw=0.8, color='gray'); ax.text(len(SHORT) + 0.45, t, lab, fontsize=7, va='bottom', ha='right')
ax.set_ylabel(f'sqrt(box area) at {INPUT_PX}px input'); ax.set_title('Object size per class')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_box_size.png'); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 2.8))
share = dist[SHORT].div(dist[SHORT].sum(axis=0), axis=1) * 100
im = ax.imshow(share.values, cmap='Blues', aspect='auto')
ax.set_xticks(range(len(SHORT)), SHORT, rotation=30, ha='right'); ax.set_yticks(range(len(share)), share.index)
for (r, c), v in np.ndenumerate(share.values):
    ax.text(c, r, f'{v:.0f}', ha='center', va='center', fontsize=7, color='white' if v > 35 else 'black')
ax.set_title('Share of each class coming from each district (%)'); fig.colorbar(im, ax=ax, shrink=0.8)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_district_class.png'); plt.close(fig)

fig, axes = plt.subplots(1, len(SHORT), figsize=(10, 1.9), sharey=True)
for ax, n in zip(axes, SHORT):
    b = boxes[boxes.name == n]
    ax.hist2d(b.x, b.y, bins=20, range=[[0, 1], [0, 1]], cmap='magma')
    ax.set_title(n, fontsize=8); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
axes[0].set_ylabel('image top -> bottom', fontsize=7)
fig.suptitle('Where box centres lie in the image', fontsize=9); fig.tight_layout()
fig.savefig(f'{OUT}/fig_box_location.png'); plt.close(fig)

if len(pairs):
    show = pairs.sort_values('hamming').head(4)
    fig, axes = plt.subplots(len(show), 2, figsize=(5, 2.4 * len(show)), squeeze=False)
    for r, (_, p) in enumerate(show.iterrows()):
        for c, (path, lab) in enumerate([(p.eval_image, p.eval_split), (p.other_image, p.other_split)]):
            axes[r, c].imshow(Image.open(path).convert('RGB')); axes[r, c].axis('off')
            axes[r, c].set_title(f'{lab}  d={p.hamming}', fontsize=7)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_near_duplicates.png'); plt.close(fig)

# ---------- 7) Summary -------------------------------------------------
summary = dict(images=len(imgs), boxes=len(boxes), images_without_boxes=int((imgs.n_box == 0).sum()),
               image_sizes={f'{w}x{hh}': int(n) for (w, hh), n in imgs.groupby(['W', 'H']).size().items()},
               label_issues=len(issues), prefixes_in_multiple_splits=int(len(cross_pano)),
               near_duplicate_pairs=int(len(pairs)), rarest_class=cnt.total.idxmin(),
               smallest_class=size.median_side_px.idxmin())
json.dump(summary, open(f'{OUT}/audit_summary.json', 'w'), indent=2)
print('\n=== SUMMARY ===\n', json.dumps(summary, indent=2))
print('\nFiles written:', sorted(os.listdir(OUT)))

# ---------- 8) Near-duplicate follow-up: how many eval images are affected? ----
print('\nPairs by Hamming distance:\n', pairs.hamming.value_counts().sort_index().to_string())
for s in ['val', 'test']:
    sub = pairs[(pairs.eval_split == s) & (pairs.other_split == 'train')]
    n_eval = int((imgs.split == s).sum())
    print(f'{s}: {sub.eval_image.nunique()} of {n_eval} images have >=1 near-duplicate in train '
          f'(exact: {sub[sub.hamming == 0].eval_image.nunique()})')
# Flagged validation images, for the "clean subset" sensitivity check
flag = pairs[(pairs.eval_split == 'val') & (pairs.other_split == 'train')].eval_image.drop_duplicates()
flag.to_csv(f'{OUT}/val_flagged_near_duplicates.txt', index=False, header=False)

# Random borderline pairs (largest distance) for visual inspection
border = pairs[pairs.hamming == HASH_THR]
if len(border):
    show = border.sample(min(3, len(border)), random_state=0)
    fig, axes = plt.subplots(len(show), 2, figsize=(5, 2.4 * len(show)), squeeze=False)
    for r, (_, p) in enumerate(show.iterrows()):
        for c, (path, lab) in enumerate([(p.eval_image, p.eval_split), (p.other_image, p.other_split)]):
            axes[r, c].imshow(Image.open(path).convert('RGB')); axes[r, c].axis('off')
            axes[r, c].set_title(f'{lab}  d={p.hamming}', fontsize=7)
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_near_duplicates_borderline.png'); plt.close(fig)
print('Files written:', sorted(os.listdir(OUT)))
