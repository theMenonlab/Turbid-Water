Three independently trained joint-RGB NAFNet controls on Legion

Workspace: /home/al/Documents/Codex/2026-10-15/task-2
No manuscript, Menon delivery, historical checkpoint, or source dataset is modified.

Source and architecture
Official megvii-research/NAFNet, commit 2b4af71ebe098a92a75910c233a3965a3e93ede4.
https://github.com/megvii-research/NAFNet/tree/2b4af71ebe098a92a75910c233a3965a3e93ede4
The three vendored architecture/dependency files are byte-identical to this
checkout's Git HEAD blobs. The official LICENSE and GoPro width32 recipe are
included. We use NAFNet itself; NAFNetLocal's large-image pooling conversion is
not applied. Official width32 topology: enc [1,1,1,28], middle 1, dec [1,1,1,1],
DW/FFN expansion 2, dropout 0. Original RGB graph: 17,111,907 parameters.

Minimal burst adaptation: replace only intro Conv2d(3,32,3,padding=1) with
Conv2d(24,32,3,padding=1); keep RGB ending convolution; replace the residual
addition with the eight-slot RGB arithmetic mean. All NAFBlocks, downsampling,
upsampling and skip connections are the official implementation. There is no
recurrent module. campaign_models.py follows the official forward operations
with that input and residual adaptation. An RGB repeated-input equivalence
check against the unmodified official graph passed at max absolute error
7.15e-7 with folded intro weights (a test only, not training initialization).
All three trainable graphs have 17,117,955 parameters, an increase of 6,048.

Input modes use identical capacity and fresh seed1337 initial weights:
 single: fifth frame (zero-based slot4) repeated into all8 RGB slots;
 mean: unrounded float32 eight-frame arithmetic mean repeated into all8 slots;
 burst: all8 measured RGB slots in their frozen order, flattened frame-major.
One fresh scratch initializer is recorded in assets/nafnet_initializer.pth and
its hash in config/CAMPAIGN_v1.json. No older NAFNet checkpoint is loaded.
The older local recurrent width24 model is a different architecture.

Frozen data and budget
Parent release: /media/al/ingold_20260515/nrl_final_paper_v3_20261014
Parent protocol SHA256: 652e2e1d7283f37730e8b95367d0c75a0a8bcb447cee4c1b5a6597a637a5d772
The existing compatible data-root is final_paper_local_backup_20261014.
All three local manifests match the downloaded final release's hashes exactly.
Restoration manifest: 12cedc7d910c922c642693959dae31e962ce84c4aa0c09dafc97a5f76c36cb05
Counts:2648train/365tune/329eval paired intervals;580/76/72 identities.
All3342 input/target content hashes, uint8 dtypes and shapes were checked.
The260 validated paired recovery intervals are included. The133 additional
category-only records are excluded. Frozen roles and illustration/reservation
exclusions remain intact. Some local object files have historical expansion
paths; content is admitted and selected solely by the final paired manifest,
not an older classification-only manifest. No dataset copies are made.
The96 licensed natural training crops and TRAIN-only generator statistics and
draw stream are unchanged. Synthetic tuning/diagnostic crops do not train.

Each fit has65536 updates, batch1,256x256,FP32, seed1337, no augmentation,
49152 measured +16384 synthetic updates, every fourth update synthetic.
Exact parent measured/synthetic draws and generator code/statistics preserved.
AdamW lr0.0002, weight_decay0, default betas(0.9,0.999), eps1e-8,
no scheduler, no gradient clipping, resets, mixed precision, or fallback.
Objective:0.1 Charbonnier +1*(1-SSIM) +0.05 spatial-gradient L1.
Deterministic cuDNN; benchmark/TF32 disabled, two CPU threads.
35 tuning milestones:0,256,1024 and every2048 through65536.
Selection: earliest maximum of measured tuning SSIM averaged across recordings
within identity/condition, then conditions within identity, then identities.
No test-set checkpoint selection. Full matched budget is not a convergence claim.

Execution and monitoring
Existing /tmp/socket-ts.1000 queue has capacity1. The original global
gpu_guard.py owns the lock for the complete serial workflow and checks actual
compute PIDs before admission. Existing jobs are never stopped or resubmitted.
128-update gates per mode (96measured+32synthetic) include actual full step0
tuning and real data loading. Their saved weights,664 Adam states, exact draws
and tuning selection must verify before full continuation. Gate continuation
restores optimizer, counters and Python/NumPy/Torch/CUDA RNG from resume.pth.
All65536 updates include gradient, weight and Adam first/second-moment finite
checks before/after optimizer steps; any error stops without changing protocol.
NUMERIC_PROGRESS.json and finite_guard.jsonl record every64 updates; training
curves record all updates. Protocol/code/data assets are hash-bound in
 audit/SCIENCE_FILE_SHA256.json.

Only best.pth and resume.pth are retained per fit; no step checkpoint pile.
resume.pth at65536 is the final weights+optimizer+RNG checkpoint. Atomic save
uses a temporary .writing file before replacement. Best/final finite state,
all65536 exact draws, all35x365 tuning records and earliest selection verify
on CPU before ALL_THREE_SELECTIONS_FROZEN.json is written. Only then does one
serial heldout evaluation per selected model compute the existing329-record
paired SSIM/PSNR/MAE support and save float32 arrays, fixed first/middle/last
examples and an independent CPU metric replay. No recognition fits are launched.
This is the historically reused evaluation cohort, not a fresh external test.

Useful records
 config/CAMPAIGN_v1.json -- exact three-fit specification and all hashes
 audit/ARCHITECTURE.json -- architecture/count/parity and initializer evidence
 audit/DATA_PREFLIGHT.json -- complete paired-data validation
 audit/ALL_THREE_GATES_PASSED.json -- measured admission gates and ETA
 audit/WORKFLOW_STATUS.json -- active run and phase
 results/restoration/<run>/NUMERIC_PROGRESS.json -- finite progress and rate
 results/restoration/<run>/best.pth -- tuning-selected weights
 results/restoration/<run>/resume.pth -- final or current full training state
 results/restoration/<run>/VERIFIED_065536.json -- final verification
 results/ALL_THREE_SELECTIONS_FROZEN.json -- freeze before heldout inference
 results/evaluation/PAIRED_SUMMARY.csv -- final paired metrics
 audit/ALL_THREE_COMPLETE.json -- complete verified campaign+evaluation
 audit/WORKFLOW_STOPPED.json -- actionable failure; never silently retry
