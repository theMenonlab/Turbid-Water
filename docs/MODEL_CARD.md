# NRL water-imaging restoration models

The release contains ten primary seed-1337 restorers and two completed seed-4099 burst Restormers. The primary models cover compact EDVR, joint-RGB Restormer, and an adaptation of official width-32 NAFNet, each with single, mean, and burst inputs, plus a measured-only Restormer burst control.

Mixed fits use 49,152 measured and 16,384 synthetic updates, with every fourth update synthetic. Measured-only fits use 65,536 measured updates. AdamW uses a constant learning rate of 0.0002, batch size one, and FP32. The fidelity objective combines Charbonnier error, SSIM loss, and spatial-gradient error with weights 0.1, 1, and 0.05. The earliest maximum of measured tuning SSIM selects the checkpoint; evaluation scores never select weights.

The compact EDVR graph has 900,019 parameters and uses Torchvision deformable convolutions. Restormer has 26,135,716 parameters and a 24-channel input stem. The NAFNet adaptation expands the official RGB input stem to 24 channels and uses the eight-slot RGB mean as its output residual. Its 17,117,955-parameter graph follows the official width-32 GoPro topology at commit `2b4af71ebe098a92a75910c233a3965a3e93ede4`. Exact architecture files and their upstream notices are included.

The seed-1337 single Restormer is the completed, corrected model selected at update 53,248. The release excludes its superseded earlier checkpoint, older recurrent NAFNet variants, unfinished seed-2027 fits, optimizer states, and intermediate training weights. Two seed-4099 checkpoints support the completed two-seed burst comparison. The original post-training amendment and incomplete-fit disclosure are retained with the evidence tables.

The models support reconstruction experiments under the recorded near-telescope water conditions. The study's evaluation identities were historically reused. Digital display targets and approximate acquisition geometry support dataset-specific comparisons; restored detail does not establish recovery of true fine structure in an unrelated scene. The two-seed comparison covers burst Restormer, and does not establish seed robustness for all architecture/input combinations.

Each `model-index.json` entry provides the family, input mode, mixture, seed, completed and selected updates, parameter count, original checkpoint SHA-256, and selection-receipt location. Original selected checkpoint bytes are preserved. Recognition classifiers are trained by the archived reproduction code; this curated weights release contains restoration models and training initializers.
