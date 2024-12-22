# ComfyUI Seamless Clone Node

A custom node for ComfyUI that implements OpenCV's seamless cloning functionality, allowing you to blend images naturally using Poisson blending techniques.

## Features

- Seamless image blending using OpenCV's 

- Three blending modes:
  - NORMAL_CLONE: Standard seamless 

  - MIXED_CLONE: Mixed seamless cloning that preserves gradients

  - MONOCHROME_TRANSFER: Monochrome transfer mode

- Automatic or manual center point selection
- Compatible with ComfyUI's image processing pipeline

## Installation

1. Navigate to your ComfyUI custom nodes directory:
```sh
cd ComfyUI/custom_nodes/
```

2. Clone this repository:
```sh
git clone https://github.com/Aksaz/comfyui-seamless-clone
```

3. Install the required dependencies:
```sh
pip install -r requirements.txt
```

## Usage

The node accepts the following inputs:

- source_image: The image to be cloned (foreground)
- destination_image: The target image (background)
- mask_image: A binary mask defining the region to be cloned
- blend_mode: Choose between NORMAL_CLONE, MIXED_CLONE, or MONOCHROME_TRANSFER


- center_x: X-coordinate of the clone center (optional)
- center_y: Y-coordinate of the clone center (optional)

Output:
- `cloned_image`: The resulting seamlessly blended image

## Requirements

- numpy==2.2.0
- opencv_python==4.10.0.84
- torch==2.5.1

## License

See the LICENSE file for details.

## Credits

This node utilizes OpenCV's seamless cloning implementation based on the paper "Seamless Image Cloning and Editing" by Patrick Pérez, Michel Gangnet, and Andrew Blake.
