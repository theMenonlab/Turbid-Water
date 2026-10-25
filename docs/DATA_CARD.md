# NRL water-imaging dataset

The dataset provides the frozen, model-ready inputs used in the current restoration and five-category recognition study. A telescope views displayed targets through nearby water under clear, motor-driven, bubbly, and turbid acquisition conditions. Categories are apparel, artwork, cars, dishes, and furniture, with integer labels 0 through 4 in that order.

The paired restoration manifest contains 3,342 intervals: 2,648 training, 365 tuning, and 329 evaluation. The classification manifest contains 3,475 intervals: 2,758 training, 372 tuning, and 345 evaluation. Both cohorts use 580, 76, and 72 identities in those roles. The 133 additional classification intervals are excluded from paired restoration. Identity roles and original content hashes are preserved.

## Files and arrays

| Path | Meaning |
|---|---|
| `data/RESTORATION_PAIRED_MANIFEST_v1.json` | Authoritative paired restoration records and provenance |
| `data/CLASSIFICATION_MANIFEST_v1.json` | Authoritative recognition records, including category-only intervals |
| `data/NATURAL_IMAGE_ROSTER.json` | 128 natural-image crops, source URLs, attribution, and roles |
| `data/objects/*.npy` | Deduplicated arrays named by their file SHA-256 |
| `metadata/*.csv` | Compact, browsable record indexes and natural-image attribution |
| `metadata/SUMMARY.json` | Counts by role and category |
| `metadata/D5_CONFIRMED_CONDITION_OVERRIDE_v1_20261013.json` | Confirmed physical interpretation of the original D5 condition label |
| `assets/physical24/*.npy` | 24 physical-object inputs for qualitative inspection |
| `synthetic_source_originals/*.jpg` | Original natural images retained for source provenance |
| `preview/` | Deterministic first, middle, and last examples from each paired split |

Input arrays are uint8 encoded RGB with shape `(8, 256, 256, 3)`. Paired targets and natural-image crops are uint8 RGB with shape `(256, 256, 3)`. Divide by 255 to obtain the model's input scale. The eight input frames retain their sampled temporal order. The single-frame control uses slot 4; mean-input models use the unrounded float32 mean of all eight slots.

The original JSON manifests retain source-content hashes, frame bounds, identity admission evidence, geometry, and historical provenance. Fields beginning `provenance:` describe source lineage and are not local file dependencies. Runtime paths under `data/objects/` resolve from the dataset package root. CSV files simplify browsing; the frozen JSON records govern admission and roles.

## Conditions, references, and reuse

D5, the June 25 high-motor group, is confirmed as clear water plus bubbles and motor. D2, the June 24 high-motor group, remains high motor. The source-linked override preserves historical keys and row IDs. Acquisition conditions and categories are not fully crossed.

Digital targets represent the displayed source images after the archived preprocessing. They are distinct from independent clear-camera references and do not define a calibrated optical-resolution ceiling. The 24 physical inputs carry neither paired ground truth nor recognition labels. The raw video archive and external UIEB imagery are outside this model-ready dataset package.

The natural-image roster has 96 train, 16 tune, and 16 diagnostic-test crops. Only the 96 training crops enter synthetic updates. Original COCO metadata identifies these photos as CC BY 2.0; attribution, source URLs, and transformations are preserved per image. Project-captured data and displayed-image content retain existing rights. No new blanket license is applied.
