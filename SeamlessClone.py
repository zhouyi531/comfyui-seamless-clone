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

        # Get destination image dimensions (H, W)
        dest_h, dest_w = destination_image_np.shape[:2]

        # Convert images to BGR for OpenCV
        source_image_cv = cv2.cvtColor(source_image_np, cv2.COLOR_RGB2BGR)
        destination_image_cv = cv2.cvtColor(destination_image_np, cv2.COLOR_RGB2BGR)

        # Calculate the actual width and height of the mask's content
        mask_indices_y, mask_indices_x = np.where(mask_np > 0)
        min_mask_x = np.min(mask_indices_x)
        max_mask_x = np.max(mask_indices_x)
        min_mask_y = np.min(mask_indices_y)
        max_mask_y = np.max(mask_indices_y)
        mask_actual_width = max_mask_x - min_mask_x + 1
        mask_actual_height = max_mask_y - min_mask_y + 1

        # Determine the desired center for cloning.
        # If input center_x, center_y are default (0,0), calculate from mask's bounding box center.
        # Otherwise, use the user-provided center_x, center_y.
        desired_cx = center_x
        desired_cy = center_y
        if center_x == 0 and center_y == 0: # Default value check from INPUT_TYPES
            desired_cx = min_mask_x + (mask_actual_width - 1) // 2
            desired_cy = min_mask_y + (mask_actual_height - 1) // 2
        
        # Clamp the desired center coordinates to ensure the entire mask bounding box
        # fits within the destination image when centered at the clamped coordinates.
        
        # Offset from the center of the mask's bounding box to its left/top edge.
        offset_x_to_left_edge = (mask_actual_width - 1) // 2
        # Offset from the center of the mask's bounding box to its right/bottom edge.
        # (mask_actual_width - 1 - offset_x_to_left_edge) is also valid for offset_x_to_right_edge
        offset_x_to_right_edge = mask_actual_width - 1 - offset_x_to_left_edge

        offset_y_to_top_edge = (mask_actual_height - 1) // 2
        offset_y_to_bottom_edge = mask_actual_height - 1 - offset_y_to_top_edge

        # Min/max allowable center points for the clone operation
        # cx must be >= offset_x_to_left_edge
        # cx must be <= dest_w - 1 - offset_x_to_right_edge
        min_allowable_cx = offset_x_to_left_edge
        max_allowable_cx = dest_w - 1 - offset_x_to_right_edge
        
        min_allowable_cy = offset_y_to_top_edge
        max_allowable_cy = dest_h - 1 - offset_y_to_bottom_edge

        # Clamp the desired center to the allowable range.
        # Ensure that min_allowable <= max_allowable, which is true if mask_actual_width <= dest_w.
        # This condition holds because mask_np is resized to destination dimensions.
        clamped_cx = np.clip(desired_cx, min_allowable_cx, max_allowable_cx).astype(int)
        clamped_cy = np.clip(desired_cy, min_allowable_cy, max_allowable_cy).astype(int)

        # Use the clamped center coordinates
        clone_center = (clamped_cx, clamped_cy)

        print(f"Original desired_center: ({desired_cx},{desired_cy}), Clamped clone_center: {clone_center}, MaskBBox: {mask_actual_width}x{mask_actual_height}, DestSize: {dest_w}x{dest_h}")
        
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