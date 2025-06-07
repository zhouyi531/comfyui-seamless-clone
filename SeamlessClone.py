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

        # Convert images to numpy arrays and scale to [0, 255] with proper clipping
        source_image_np = np.clip((source_image[0].cpu().numpy() * 255.0), 0, 255).astype(np.uint8)
        destination_image_np = np.clip((destination_image[0].cpu().numpy() * 255.0), 0, 255).astype(np.uint8)

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

        # Ensure mask is binary (0 or 255) with proper thresholding
        mask_np = np.clip(mask_np, 0, 1)
        mask_np = (mask_np > 0.5).astype(np.uint8) * 255

        # Check if mask is empty
        if np.count_nonzero(mask_np) == 0:
            # Mask is empty; cannot proceed
            raise ValueError("Mask is empty after processing.")

        # Get destination image dimensions (H, W)
        dest_h, dest_w = destination_image_np.shape[:2]

        # Ensure images are in correct format for OpenCV
        # ComfyUI images are RGB, OpenCV expects BGR
        if source_image_np.ndim == 3 and source_image_np.shape[2] == 3:
            source_image_cv = cv2.cvtColor(source_image_np, cv2.COLOR_RGB2BGR)
        else:
            source_image_cv = source_image_np
            
        if destination_image_np.ndim == 3 and destination_image_np.shape[2] == 3:
            destination_image_cv = cv2.cvtColor(destination_image_np, cv2.COLOR_RGB2BGR)
        else:
            destination_image_cv = destination_image_np

        # Calculate the center of the mask if center_x and center_y are not provided
        if center_x == 0 and center_y == 0:
            mask_indices = np.argwhere(mask_np > 0)
            if len(mask_indices) > 0:
                mask_center = mask_indices.mean(axis=0).astype(int)
                center_x, center_y = mask_center[1], mask_center[0]  # (x, y) format
            else:
                # Fallback to image center if mask is invalid
                center_x, center_y = dest_w // 2, dest_h // 2

        # Use the calculated or provided center coordinates
        clone_center = (center_x, center_y)
        print(f"Initial clone_center: {clone_center}")
        
        # Check if mask touches image boundaries and adjust if necessary
        mask_adjusted = self._adjust_mask_for_boundaries(mask_np, dest_h, dest_w)
        
        # If mask was adjusted, recalculate center if it was auto-calculated
        if not np.array_equal(mask_np, mask_adjusted):
            print("Mask was adjusted to avoid boundary issues")
            if center_x == 0 and center_y == 0:
                # Recalculate center for adjusted mask
                if np.count_nonzero(mask_adjusted) > 0:
                    mask_indices = np.argwhere(mask_adjusted > 0)
                    mask_center = mask_indices.mean(axis=0).astype(int)
                    center_x, center_y = mask_center[1], mask_center[0]
                    clone_center = (center_x, center_y)
                    print(f"Adjusted clone_center: {clone_center}")
                else:
                    raise ValueError("Mask became empty after boundary adjustment.")
        
        # Map blend_mode string to OpenCV constant
        blend_mode_dict = {
            "NORMAL_CLONE": cv2.NORMAL_CLONE,
            "MIXED_CLONE": cv2.MIXED_CLONE,
            "MONOCHROME_TRANSFER": cv2.MONOCHROME_TRANSFER,
        }
        mode = blend_mode_dict.get(blend_mode, cv2.NORMAL_CLONE)

        try:
            # Perform seamless cloning with adjusted mask
            output_cv = cv2.seamlessClone(
                source_image_cv, destination_image_cv, mask_adjusted, clone_center, mode
            )
        except cv2.error as e:
            # If still failing, try with a more conservative approach
            print(f"OpenCV error occurred: {e}")
            print("Attempting fallback with further mask adjustment...")
            
            try:
                # More aggressive mask adjustment
                mask_conservative = self._create_conservative_mask(mask_adjusted, dest_h, dest_w, clone_center)
                
                if np.count_nonzero(mask_conservative) == 0:
                    raise ValueError("Unable to create a valid mask for seamless cloning. The mask region may be too close to image boundaries.")
                
                output_cv = cv2.seamlessClone(
                    source_image_cv, destination_image_cv, mask_conservative, clone_center, mode
                )
            except cv2.error as e2:
                print(f"Conservative approach also failed: {e2}")
                # Final fallback: return destination image
                output_cv = destination_image_cv

        # Convert output back to RGB format (from BGR)
        if output_cv.ndim == 3 and output_cv.shape[2] == 3:
            output_rgb = cv2.cvtColor(output_cv, cv2.COLOR_BGR2RGB)
        else:
            output_rgb = output_cv

        # Convert to torch.Tensor and scale to [0, 1] with proper clipping
        output_tensor = torch.from_numpy(np.clip(output_rgb.astype(np.float32) / 255.0, 0.0, 1.0))

        # Add batch dimension
        output_tensor = output_tensor.unsqueeze(0)  # Shape [1, H, W, C]

        return (output_tensor,)
    
    def _adjust_mask_for_boundaries(self, mask, height, width):
        """
        Adjust mask to avoid boundary issues with seamlessClone.
        Remove mask pixels that are too close to image edges.
        """
        adjusted_mask = mask.copy()
        
        # Define minimum distance from edges (in pixels)
        edge_margin = 3
        
        # Set edge regions to 0
        adjusted_mask[:edge_margin, :] = 0  # Top edge
        adjusted_mask[-edge_margin:, :] = 0  # Bottom edge
        adjusted_mask[:, :edge_margin] = 0  # Left edge
        adjusted_mask[:, -edge_margin:] = 0  # Right edge
        
        return adjusted_mask
    
    def _create_conservative_mask(self, mask, height, width, center):
        """
        Create a more conservative mask by eroding the mask and ensuring
        it doesn't touch boundaries.
        """
        # Apply morphological erosion to shrink the mask
        kernel = np.ones((5, 5), np.uint8)
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
        
        # Apply larger edge margin
        edge_margin = 10
        conservative_mask = eroded_mask.copy()
        conservative_mask[:edge_margin, :] = 0
        conservative_mask[-edge_margin:, :] = 0
        conservative_mask[:, :edge_margin] = 0
        conservative_mask[:, -edge_margin:] = 0
        
        return conservative_mask
