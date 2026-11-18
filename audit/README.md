# Data audit

`svrdd_audit.py` produces the dataset analysis used in the progress report (Section IV-A).

**Run in Colab** (CPU is enough), after notebook sections 1, 2 and 2b:

```python
%run -i {PROJECT}/audit/svrdd_audit.py
```

**Outputs** (written to `<DRIVE_DIR>/audit`):

| File | Content |
|---|---|
| `label_issues.csv` | invalid / duplicate boxes, missing label files |
| `per_class_stats.csv`, `.tex` | boxes per class and split, imbalance ratio, box size (COCO bins at 640 px) |
| `district_by_class.csv` | boxes per district and class |
| `near_duplicate_pairs.csv` | val/test images with a dHash distance <= 4 to another split |
| `val_flagged_near_duplicates.txt` | flagged validation images (for the clean-subset check) |
| `fig_*.png` | class counts, box sizes, district heatmap, box locations, near-duplicate examples |
| `audit_summary.json` | headline numbers |

**Main findings (official split):** clean labels (1 duplicate box); 5.1:1 imbalance (L-crack vs pothole);
pothole is rare *and* small, manhole is small but frequent, alligator is rare but large;
72% of potholes come from Fengtai; exact near-duplicates of training frames: 2 val and 7 test images
(consecutive frames of the same drive).
