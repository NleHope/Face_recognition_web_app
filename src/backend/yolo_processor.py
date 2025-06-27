
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import time
from typing import Union, List, Tuple

class YOLOProcessor:
    """
    A class to encapsulate the YOLOv8 model loading, configuration,
    and frame processing logic for real-time object detection.

    This class applies Object-Oriented Programming (OOP) principles like:
    - Encapsulation: Bundling data (model, device, thresholds) and methods
      that operate on that data within a single unit.
    - Abstraction: Hiding the complex internal details of model loading and
      frame processing, exposing only necessary methods for interaction.
    """

    def __init__(self, model_path: str = 'models/yolov8n.pt',
                 initial_confidence_threshold: float = 0.25,
                 img_size: Tuple[int, int] = (640, 640)):
        """
        Constructor for the YOLOProcessor class.
        Initializes the YOLOv8 model, sets up the device, and defines initial parameters.

        Args:
            model_path (str): Path to the YOLOv8 model weights (e.g., 'yolov8n.pt').
            initial_confidence_threshold (float): The default confidence threshold for detections.
                                                  Detections with confidence below this will be filtered out.
            img_size (Tuple[int, int]): The target image size (width, height) for model inference.
                                        YOLO models typically use square inputs (e.g., 640, 640).
                                        Frames will be resized to this dimension before being fed to the model.
        """
        # Determine the device (GPU or CPU) for model inference
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected'}")

        print(f"Loading YOLOv8 model from {model_path} on device: {self.device}...")
        # Load the YOLO model. The .to(self.device) moves it to the specified device.
        self.model = YOLO(model_path)
        
        # If using CUDA, convert the model to half-precision (FP16) for faster inference
        # and reduced memory usage, especially on compatible GPUs.
        if self.device == 'cuda':
            self.model.half()
        
        # Store the confidence threshold and image size as private attributes.
        # Using '_' prefix is a Python convention to indicate these are intended for
        # internal use or accessed via properties.
        self._confidence_threshold = initial_confidence_threshold
        self._img_size = img_size
        
        # Set the initial confidence threshold directly on the YOLO model.
        # This ensures the model's internal filtering uses the specified value.
        self.model.conf = self._confidence_threshold
        
        print("YOLOv8 model loaded and initialized.")

    def _pad_to_divisible(self, image: np.ndarray, stride: int = 32) -> Tuple[np.ndarray, int, int]:
        """
        A private helper method to pad an image.
        Ensures that the image dimensions are divisible by the given stride,
        which is often a requirement for deep learning models to avoid issues
        with convolutional layers or pooling operations.

        Args:
            image (np.ndarray): The input image as a NumPy array (H, W, C).
            stride (int): The stride value (e.g., 32 for YOLOv8).

        Returns:
            Tuple[np.ndarray, int, int]: A tuple containing:
                - padded_image (np.ndarray): The image with added padding.
                - pad_h (int): The amount of padding added to the height.
                - pad_w (int): The amount of padding added to the width.
        """
        h, w = image.shape[:2]
        # Calculate new dimensions that are multiples of the stride
        new_h = ((h + stride - 1) // stride) * stride
        new_w = ((w + stride - 1) // stride) * stride
        
        # Calculate the amount of padding needed
        pad_h = new_h - h
        pad_w = new_w - w

        # Apply padding using OpenCV's copyMakeBorder.
        # BORDER_CONSTANT fills the border with a constant value (0, 0, 0 for black).
        padded_image = cv2.copyMakeBorder(
            image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        return padded_image, pad_h, pad_w

    def set_confidence_threshold(self, threshold: float):
        """
        Public method to set (update) the confidence threshold for object detection.
        This allows external components (like your Gradio app) to dynamically
        change the model's behavior.

        Args:
            threshold (float): The new confidence threshold (must be between 0.0 and 1.0).
        
        Raises:
            ValueError: If the provided threshold is outside the valid range.
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("Confidence threshold must be between 0.0 and 1.0")
        
        self._confidence_threshold = threshold
        self.model.conf = threshold # Update the model's internal confidence attribute
        print(f"YOLO confidence threshold updated to: {self._confidence_threshold:.2f}")

    def get_confidence_threshold(self) -> float:
        """
        Public method to get the current confidence threshold.
        """
        return self._confidence_threshold

    def set_image_size(self, img_size: Tuple[int, int]):
        """
        Public method to set (update) the target image size for model inference.
        This can be useful if you want to change the input resolution of the model.

        Args:
            img_size (Tuple[int, int]): New target image size (width, height).
        
        Raises:
            ValueError: If the img_size is not a valid tuple of positive integers.
        """
        if not isinstance(img_size, tuple) or len(img_size) != 2 or not all(isinstance(x, int) and x > 0 for x in img_size):
            raise ValueError("img_size must be a tuple of two positive integers (width, height).")
        self._img_size = img_size
        print(f"YOLO inference image size updated to: {self._img_size}")

    def get_image_size(self) -> Tuple[int, int]:
        """
        Public method to get the current inference image size.
        """
        return self._img_size

    @torch.inference_mode()  # Decorator to disable gradient calculation, crucial for inference performance.
    def process_frames(self, frames: Union[np.ndarray, List[np.ndarray]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Main public method to process a single frame or a list of frames using the YOLOv8 model.
        It orchestrates the preprocessing, actual model inference, and post-processing
        (drawing bounding boxes and labels).

        Args:
            frames (Union[np.ndarray, List[np.ndarray]]): A single image (H, W, C) as a NumPy array
                                                            or a list of images (each H, W, C).

        Returns:
            Union[np.ndarray, List[np.ndarray]]: A single processed image (with detections drawn)
                                                  if the input was a single frame,
                                                  or a list of processed images if the input was a list.
                                                  Returns original frames on error.
        """
        try:
            total_start_time = time.time() # Start timing for the entire process
            pre_start = time.time()      # Start timing for preprocessing

            # Determine if the input is a single frame or a list of frames
            is_batch_input = isinstance(frames, list)
            frames_list = frames if is_batch_input else [frames]

            processed_frames_for_batch = [] # List to hold preprocessed frames (tensors)
            original_frame_sizes = []       # List to store original dimensions for post-processing scaling

            # --- Preprocessing Loop ---
            for frame in frames_list:
                original_frame_sizes.append((frame.shape[1], frame.shape[0])) # Store original (width, height)

                # Resize frame to the model's target inference size (self._img_size)
                # This ensures consistent input dimensions for the YOLO model.
                if self._img_size and (frame.shape[1] != self._img_size[0] or frame.shape[0] != self._img_size[1]):
                    frame_resized = cv2.resize(frame, self._img_size, interpolation=cv2.INTER_AREA)
                else:
                    frame_resized = frame # No resizing needed if already at target size

                # Pad the resized frame to be divisible by stride (e.g., 32)
                frame_padded, _, _ = self._pad_to_divisible(frame_resized, stride=32)

                # Convert NumPy array to PyTorch tensor
                # - permute(2, 0, 1): Changes HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
                # - .contiguous(): Ensures tensor is stored contiguously in memory for performance
                # - .float(): Converts data type to float32
                # - .to(self.device): Moves tensor to GPU or CPU
                # - / 255.0: Normalizes pixel values from 0-255 to 0-1
                frame_tensor = torch.from_numpy(frame_padded).permute(2, 0, 1).contiguous().float().to(self.device) / 255.0
                
                # Add a batch dimension (e.g., from (3, 640, 640) to (1, 3, 640, 640))
                # Convert to half-precision (FP16) if on GPU for further optimization.
                frame_tensor = frame_tensor.unsqueeze(0).half() if self.device == 'cuda' else frame_tensor.unsqueeze(0)
                
                processed_frames_for_batch.append(frame_tensor)

            # Concatenate all preprocessed tensors to form a single batch tensor for efficient inference
            batch_tensor = torch.cat(processed_frames_for_batch, dim=0)
            preprocess_time = time.time() - pre_start # End timing for preprocessing

            # --- Inference ---
            infer_start = time.time() # Start timing for inference
            # Call the YOLO model's prediction method.
            # - imgsz: Specifies the input image size for the model.
            # - device: Ensures inference happens on the correct device.
            # - verbose=False: Suppresses verbose output from YOLO.
            # - stream=False: Important for batch processing; ensures all results are collected before returning.
            results = self.model(batch_tensor, imgsz=self._img_size[0], device=self.device, verbose=False, stream=False)
            inference_time = time.time() - infer_start # End timing for inference
            
            # Get class names from the loaded YOLO model
            class_names = self.model.names

            # --- Post-processing Loop ---
            post_start = time.time() # Start timing for post-processing
            output_frames = [] # List to store frames with drawn detections

            for idx, result in enumerate(results):
                original_w, original_h = original_frame_sizes[idx]
                # Create a copy of the original frame to draw on.
                # This prevents modifying the input frame directly, which might be needed elsewhere.
                current_frame = frames_list[idx].copy() 

                # Extract bounding box coordinates, class IDs, and confidences from YOLO results
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                # Calculate scaling factors to map detection coordinates (from img_size)
                # back to the original frame dimensions.
                scale_x = original_w / self._img_size[0] if self._img_size[0] > 0 else 1
                scale_y = original_h / self._img_size[1] if self._img_size[1] > 0 else 1

                # Iterate through each detected object
                for box, cls, conf in zip(boxes, classes, confidences):
                    # Convert bounding box coordinates to integers
                    x1, y1, x2, y2 = map(int, box[:4])
                    
                    # Rescale coordinates to match the original frame's resolution
                    x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

                    # Clip coordinates to ensure they stay within the boundaries of the original image
                    x1, x2 = np.clip([x1, x2], 0, original_w)
                    y1, y2 = np.clip([y1, y2], 0, original_h)

                    # Create the label string (class name and confidence score)
                    label = f"{class_names[int(cls)]} {conf:.2f}"

                    # Draw the bounding box rectangle on the current frame
                    cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Adjust text position to ensure it's visible and doesn't go off-screen
                    text_y_pos = y1 - 10 if y1 - 10 > 0 else y1 + 20
                    cv2.putText(current_frame, label, (x1, text_y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                output_frames.append(current_frame) # Add the processed frame to the output list

            postprocess_time = time.time() - post_start # End timing for post-processing
            total_time = time.time() - total_start_time # Total time for processing this batch

            # Optional: Print timing information for debugging/performance monitoring
            # print(f"[TIME] Pre: {preprocess_time:.3f}s | Infer: {inference_time:.3f}s | Post: {postprocess_time:.3f}s | Total: {total_time:.3f}s")

            # Return a single frame if the input was a single frame, otherwise return a list of frames
            return output_frames[0] if not is_batch_input else output_frames

        except Exception as e:
            # Error handling: Print error and traceback, then return original frames
            # This prevents the application from crashing and provides visual feedback.
            print(f"Error in YOLOProcessor.process_frames: {e}")
            import traceback
            traceback.print_exc()
            
            # Return the original frames to avoid breaking the client stream
            if is_batch_input:
                return frames_list
            else:
                return frames[0] if isinstance(frames, list) else frames

# Example usage (for testing purposes, this part would typically not be in the yolo_processor.py file
# when used as a module, but helps demonstrate how to use the class).
if __name__ == "__main__":
    print("--- Testing YOLOProcessor Class ---")
    
    # 1. Create an instance of the processor
    # This will load the YOLO model once.
    try:
        yolo_detector = YOLOProcessor(model_path='yolov8n.pt', 
                                      initial_confidence_threshold=0.5,
                                      img_size=(640, 640))
    except Exception as e:
        print(f"Failed to initialize YOLOProcessor: {e}")
        print("Please ensure 'models/yolov8n.pt' exists and PyTorch/CUDA are correctly set up.")
        exit()

    # 2. Simulate a dummy frame (e.g., a black image)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8) # Black image 640x480 (W, H, C)

    print("\n--- Processing a single dummy frame ---")
    processed_single_frame = yolo_detector.process_frames(dummy_frame)
    if processed_single_frame is not None and processed_single_frame.shape == dummy_frame.shape:
        print(f"Successfully processed single frame. Shape: {processed_single_frame.shape}")
    else:
        print("Failed to process single frame or shape mismatch.")

    # 3. Simulate a batch of dummy frames
    print("\n--- Processing a batch of dummy frames ---")
    # Create 3 random images for the batch
    dummy_frames_batch = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(3)]
    processed_batch_frames = yolo_detector.process_frames(dummy_frames_batch)
    
    if processed_batch_frames is not None and isinstance(processed_batch_frames, list) and len(processed_batch_frames) == 3:
        print(f"Successfully processed batch of frames. Count: {len(processed_batch_frames)}")
        print(f"Shape of first processed frame in batch: {processed_batch_frames[0].shape}")
    else:
        print("Failed to process batch of frames or output format mismatch.")

    # 4. Test updating confidence threshold
    print("\n--- Testing confidence threshold update ---")
    current_threshold = yolo_detector.get_confidence_threshold()
    print(f"Current confidence threshold: {current_threshold}")
    
    try:
        yolo_detector.set_confidence_threshold(0.7)
        print(f"New confidence threshold: {yolo_detector.get_confidence_threshold()}")
        yolo_detector.set_confidence_threshold(1.1) # This should raise an error
    except ValueError as e:
        print(f"Caught expected error when setting invalid threshold: {e}")

    # 5. Test updating image size
    print("\n--- Testing image size update ---")
    current_img_size = yolo_detector.get_image_size()
    print(f"Current inference image size: {current_img_size}")
    
    try:
        yolo_detector.set_image_size((320, 320))
        print(f"New inference image size: {yolo_detector.get_image_size()}")
        yolo_detector.set_image_size((100,)) # This should raise an error
    except ValueError as e:
        print(f"Caught expected error when setting invalid image size: {e}")