# ComfyUI Flux2 Fun ControlNet

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

ComfyUI implementation of [FLUX.2-dev-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union) from Alibaba's VideoX-Fun.

A **unified ControlNet** that supports multiple control modes with a single checkpoint — no mode switching required.

## Supported Control Types

| Control Type | Description                  |
|--------------|------------------------------|
| **Pose**     | OpenPose skeleton            |
| **Canny**    | Edge detection               |
| **Depth**    | Depth maps                   |
| **HED**      | Soft edge detection          |
| **MLSD**     | Line segment detection       |
| **Tile**     | Upscaling/detail enhancement |

The model automatically detects the control type from your input image.

## Installation

### Method 1: Git Clone

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jadervasque/comfyui-flux2fun-controlnet.git
```

### Method 2: Download ZIP

1. Download this repository as ZIP.
2. Extract it to `ComfyUI/custom_nodes/comfyui-flux2fun-controlnet`.

### Download Model

Download the ControlNet checkpoint and place it in `ComfyUI/models/controlnet/`:

- [FLUX.2-dev-Fun-Controlnet-Union.safetensors](https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union/tree/main) (~8.3 GB)

### Requirements

- Current ComfyUI with FLUX.2 support
- FLUX.2-dev base model
- `flux2-vae.safetensors` or compatible VAE
- Python 3.10+
- PyTorch 2.0+

## Nodes

### Load Flux2 Fun ControlNet

Loads the ControlNet checkpoint.

| Input           | Type     | Description            |
|-----------------|----------|------------------------|
| controlnet_name | dropdown | Select checkpoint file |

| Output     | Type                 | Description  |
|------------|----------------------|--------------|
| controlnet | FLUX2_FUN_CONTROLNET | Loaded model |

### Apply Flux2 Fun ControlNet

Applies ControlNet to conditioning.

| Input         | Type                 | Description                  |
|---------------|----------------------|------------------------------|
| conditioning  | CONDITIONING         | Text conditioning from CLIP  |
| controlnet    | FLUX2_FUN_CONTROLNET | Loaded ControlNet            |
| vae           | VAE                  | FLUX VAE                     |
| strength      | FLOAT                | Control strength (0.0–2.0)   |
| control_image | IMAGE (optional)     | Control signal               |
| mask          | MASK (optional)      | Inpaint mask                 |
| inpaint_image | IMAGE (optional)     | Image to inpaint             |

| Output       | Type         | Description           |
|--------------|--------------|-----------------------|
| conditioning | CONDITIONING | Modified conditioning |

## Usage

### Control Mode

For pose, canny, depth, HED, MLSD, or tile control:

1. Load the control image.
2. Connect it to `control_image`.
3. Set strength to approximately **0.65–0.80**.
4. Leave `mask` and `inpaint_image` disconnected.

### Control + Inpaint Mode

For regional regeneration with structural guidance:

1. Connect `control_image`.
2. Connect `mask`, where white identifies the regeneration area.
3. Connect `inpaint_image`.
4. Set strength to approximately **0.25–0.40**.

This is mask-guided regional regeneration rather than traditional context-aware inpainting. A dedicated inpaint model may be preferable for general inpainting.

## Recommended Settings

| Mode                       | Strength    | Steps | CFG     | Notes            |
|----------------------------|-------------|-------|---------|------------------|
| Control (pose/canny/depth) | 0.65–0.80   | 25–50 | 3.5–4.5 | Primary use case |
| Control + Inpaint          | 0.25–0.40   | 25–50 | 3.5–4.5 | Experimental     |

## Example Workflows

See the [examples](examples/) folder for ready-to-use workflows.

## Technical Details

This implementation:

- Uses ComfyUI's composable per-generation `patches_replace["dit"]` interface.
- Does **not** replace `Flux.forward_orig` or modify ComfyUI core files.
- Generates ControlNet hints once from the initial double-stream inputs.
- Injects hints after FLUX double-stream blocks 0, 2, 4, and 6.
- Composes with an existing DIT replacement rather than overwriting it.
- Supports reference latents, including `ref_latents_method="index_timestep_zero"`.
- Uses the main-image timestep modulation for ControlNet while ComfyUI retains mixed modulation for reference tokens.
- Applies hints only to main image tokens, never to appended reference-latent tokens.
- Supports chained Flux2 Fun ControlNets with independent strengths.
- Uses a 260-channel control context: 128 control + 4 mask + 128 inpaint.

## Troubleshooting

### Module not found

Restart ComfyUI after installation or update.

### Black output or no effect

- Confirm that strength is greater than zero.
- Verify that the VAE is connected.
- Verify that the control image is valid and has compatible dimensions.

### Out of memory

- Reduce image resolution.
- Use ComfyUI low-VRAM mode.
- Close other GPU applications.

## Credits

- **Original Model:** [alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/FLUX.2-dev-Fun-Controlnet-Union)
- **Reference Implementation:** [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun.git)

## License

[Apache 2.0](LICENSE) — same as VideoX-Fun.

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
