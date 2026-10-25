# Using the released restorers

All models accept eight RGB input slots and return one RGB image. Each input comparison uses a separately trained model with the same capacity within its family.

- **Single:** repeat the fifth sampled frame, index 4, into all eight slots.
- **Mean:** repeat the unrounded float32 arithmetic mean of the eight frames.
- **Burst:** retain all eight frames in their recorded order.

The wrapper performs these operations inside each model. Pass the original burst without computing a rounded mean first. Encoded RGB values are divided by 255; no radiometric conversion, fitted alignment, or brightness correction is applied. The release uses FP32 inference.

`infer.py` accepts a still image or NumPy burst. Still images are converted to RGB and resized with Lanczos when their longest dimension exceeds `--max-side`. NumPy bursts retain their original geometry. Spatial dimensions must be at least 32 pixels. Reflect padding extends the bottom and right edges to multiples of 32 when needed; the prediction is cropped back to the original geometry. Released 256-pixel benchmark arrays need no padding.

The returned `.npy` contains unclipped float32 RGB. The PNG uses clipping to the unit interval, multiplication by 255, and NumPy ties-to-even rounding. Benchmark metrics clip predictions to the unit interval without quantizing them.

## Python interface

```python
import numpy as np
from nrl_water import load_model, predict

net = load_model("restormer_RGB_mean_mixed_seed1337", "/path/to/kaggle-model-weights")
burst = np.load("/path/to/eight_frames.npy", allow_pickle=False)
prediction = predict(net, burst)
```

`load_model` checks the original checkpoint hash, loads the complete state dictionary strictly, and verifies the model's parameter count. The loader uses the archived architecture implementations. Training initializer files are unnecessary for inference.

CPU execution is supported. EDVR requires the compiled Torchvision deformable-convolution operator, so PyTorch and Torchvision must be compatible. Large-image inference can require substantial memory; begin with the default 256-pixel geometry.
