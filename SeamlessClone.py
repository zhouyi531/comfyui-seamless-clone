import torch
import cv2
import numpy as np

class SeamlessClone:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "destination_image": ("IMAGE",),
                "mask_image": ("MASK",),
                "blend_mode": (["NORMAL_CLONE", "MIXED_CLONE", "MONOCHROME_TRANSFER"],),
                "center_x": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "center_y": ("INT", {"default": 0, "min": 0, "max": 8192}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cloned_image",)
    CATEGORY = "Image Processing"
    FUNCTION = "seamless_clone"

    def seamless_clone(self, source_image, destination_image, mask_image, blend_mode, center_x=None, center_y=None):

        # Ensure batch size is 1 for simplicity
        if source_image.shape[0] != 1 or destination_image.shape[0] != 1 or mask_image.shape[0] != 1:
            raise ValueError("Batch size greater than 1 is not supported.")

        # Convert images to numpy arrays and scale to [0, 255]
        source_image_np = (source_image[0].cpu().numpy() * 255).astype(np.uint8)
        destination_image_np = (destination_image[0].cpu().numpy() * 255).astype(np.uint8)

        # Mask is in [0,1] range (since it's a torch.Tensor)
        mask_np = mask_image[0].cpu().numpy()

        # Resize source image and mask to match destination image size
        dest_h, dest_w = destination_image_np.shape[:2]
        source_image_np = cv2.resize(
            source_image_np, (dest_w, dest_h), interpolation=cv2.INTER_LINEAR
        )
        mask_np = cv2.resize(
            mask_np, (dest_w, dest_h), interpolation=cv2.INTER_LINEAR
        )

        # Ensure mask is single-channel
        if mask_np.ndim == 3 and mask_np.shape[2] > 1:
            mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
        elif mask_np.ndim == 3:
            mask_np = mask_np[:, :, 0]

        # Ensure mask is binary (0 or 255)
        mask_np = (mask_np > 0.5).astype(np.uint8) * 255

        # Check if mask is empty
        if np.count_nonzero(mask_np) == 0:
            # Mask is empty; cannot proceed
            raise ValueError("Mask is empty after processing.")

        # Convert images to BGR for OpenCV
        source_image_cv = cv2.cvtColor(source_image_np, cv2.COLOR_RGB2BGR)
        destination_image_cv = cv2.cvtColor(destination_image_np, cv2.COLOR_RGB2BGR)

        # Calculate the center of the mask if center_x and center_y are not provided
        if center_x == 0 and center_y == 0:
            mask_indices = np.argwhere(mask_np > 0)
            mask_center = mask_indices.mean(axis=0).astype(int)
            center_x, center_y = mask_center[1], mask_center[0]  # (x, y) format

        # Use the calculated or provided center coordinates
        clone_center = (center_x, center_y)

        print(f"clone_center: {clone_center}")
        
        # Map blend_mode string to OpenCV constant
        blend_mode_dict = {
            "NORMAL_CLONE": cv2.NORMAL_CLONE,
            "MIXED_CLONE": cv2.MIXED_CLONE,
            "MONOCHROME_TRANSFER": cv2.MONOCHROME_TRANSFER,
        }
        mode = blend_mode_dict.get(blend_mode, cv2.NORMAL_CLONE)

        # Perform seamless cloning
        output_cv = cv2.seamlessClone(
            source_image_cv, destination_image_cv, mask_np, clone_center, mode
        )

        # Convert output to RGB
        output_rgb = cv2.cvtColor(output_cv, cv2.COLOR_BGR2RGB)

        # Convert to torch.Tensor and scale to [0, 1]
        output_tensor = torch.from_numpy(output_rgb.astype(np.float32) / 255.0)

        # Add batch dimension
        output_tensor = output_tensor.unsqueeze(0)  # Shape [1, H, W, C]

        return (output_tensor,)