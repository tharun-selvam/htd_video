import cv2
import torch
import os
import torch.nn as nn
import ffmpeg
import shutil
import numpy as np
import matplotlib.pyplot as plt
import signal
import sys
import psutil
import time
from sklearn.manifold import TSNE
from torchvision.models import resnet50, ResNet50_Weights
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from torchvision.utils import flow_to_image
from _htd_tree import process_scene_with_tree
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates

# --- GLOBAL CONTROLLER: Choose your flow algorithm ---
# Options: "RAFT" or "FARNEBACK"
FLOW_METHOD = "FARNEBACK"
SMALL_RAFT = 1
BIG_RAFT = 2

# Global variables for graceful shutdown
processing_interrupted = False

def load_vlm_model(model_path, device="cuda"):
    """
    Loads the FastVLM (LLaVA-based) model, tokenizer, and image processor.
    This is based on the logic from the official predict.py.
    """
    print(f"\n[INFO] Loading VLM model from: {model_path}...")
    
    model_base = None 
    model_name = get_model_name_from_path(model_path)
    
    try:
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, 
            model_base, 
            model_name, 
            device=device  # Use the script's device, not a hardcoded "mps"
        )
    except RuntimeError as e:
        if "mps" in str(e):
            print("\n[FATAL ERROR] Your PyTorch environment is not configured for your GPU.")
            print("You are likely running the Apple-provided code on an NVIDIA GPU without fixing the hardcoded 'mps' device flags.")
            print("Please fix the predict.py file in the llava library or your environment.\n")
        raise e

    model.eval()
    print("[INFO] FastVLM loaded successfully.")
    return model, tokenizer, image_processor


def annotate_complex_nodes_with_vlm(complex_nodes, video_path, model, tokenizer, image_processor, device):
    """
    Generates textual descriptions for a list of complex leaf nodes using FastVLM.
    It samples ONE representative frame (the middle frame) from each complex clip.
    """
    if not complex_nodes:
        print("[INFO] No complex nodes to annotate.")
        return

    print(f"\n[INFO] Annotating {len(complex_nodes)} complex leaf nodes with VLM...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file for VLM annotation: {video_path}")
        return

    for i, node in enumerate(complex_nodes):
        global processing_interrupted
        if processing_interrupted:
            print("  -> VLM annotation interrupted.")
            break

        print(f"  -> Annotating node {i+1}/{len(complex_nodes)}: {node.node_id}")
        
        # --- 1. Sample the MIDDLE frame from the clip ---
        middle_frame_idx = node.start_frame + (node.frame_count // 2)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"    -> Could not sample middle frame for node {node.node_id}. Skipping.")
            continue
            
        # Convert frame from BGR (OpenCV) to RGB and then to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        image_size = pil_image.size

        # --- 2. Process the Image (using FastVLM/LLaVA's processor) ---
        image_tensor = process_images([pil_image], image_processor, model.config)[0]
        image_tensor = image_tensor.unsqueeze(0).to(device, dtype=torch.float16)

        # # --- 3. Build the VLM Prompt (using the required conversation template) ---
        # prompt_text = "Describe the primary action in this scene in one detailed sentence."
        
        # # FastVLM uses the Qwen2 conv template ("qwen_2")
        # conv = conv_templates["qwen_2"].copy()
        
        # prompt_with_tokens = DEFAULT_IMAGE_TOKEN + '\n' + prompt_text
        # conv.append_message(conv.roles[0], prompt_with_tokens)
        # conv.append_message(conv.roles[1], None)
        # prompt_inputs = conv.get_prompt()

        # # Tokenize the final prompt string
        # input_ids = tokenizer_image_token(prompt_inputs, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)

        # # --- 4. Generate Description with VLM ---
        # try:
        #     with torch.inference_mode():
        #         output_ids = model.generate(
        #             input_ids,
        #             images=image_tensor,
        #             image_sizes=[image_size],
        #             do_sample=False,  # Set to False for deterministic descriptions
        #             temperature=0.0,
        #             max_new_tokens=150,
        #             use_cache=True
        #         )

        #     description = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
        #     # --- 5. Store the Description directly in the Tree Node ---
        #     node.vlm_labels = {"description": description}
        #     print(f"    -> VLM Description: \"{description}\"")

        # except Exception as e:
        #     print(f"    -> [ERROR] Failed to generate VLM description for node {node.node_id}. Reason: {e}")
        #     node.vlm_labels = {"description": "Error during generation."}
        #     if "out of memory" in str(e).lower():
        #         print("    -> CUDA Out of Memory. This can happen if VRAM is fragmented. Stopping VLM stage.")
        #         torch.cuda.empty_cache()
        #         break # Stop trying to annotate

        # Construct prompt
        prompt_text = "Describe the primary action in this scene in one detailed sentence."
        qs = prompt_text
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Set the pad token id for generation
        model.generation_config.pad_token_id = tokenizer.pad_token_id

        # Tokenize prompt
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(torch.device("cuda"))

        # Run inference
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half(),
                image_sizes=[image_size],
                do_sample=True,
                temperature=0.2,
                top_p=None,
                num_beams=1,
                max_new_tokens=256,
                use_cache=True)

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

            node.vlm_labels = {"description": outputs}
            print(f"    -> VLM Description: \"{outputs}\"")
        #     print(f"    -> VLM Description: \"{description}\"")

    cap.release()
    print("[INFO] VLM annotation complete.")




def signal_handler(sig, frame):
    """Handle interruption signals gracefully"""
    global processing_interrupted
    print(f"\n[WARNING] Received signal {sig}. Initiating graceful shutdown...")
    processing_interrupted = True

def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB

def check_memory_pressure():
    """Check if system is under memory pressure"""
    try:
        # Get system memory info
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        percent_used = memory.percent

        # Get GPU memory if available
        gpu_memory_info = ""
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_memory_reserved = torch.cuda.memory_reserved() / (1024**3)
            gpu_memory_info = f" GPU: {gpu_memory_allocated:.2f}GB allocated, {gpu_memory_reserved:.2f}GB reserved"

        print(f"    -> Memory: {available_gb:.1f}GB available ({100-percent_used:.1f}% free){gpu_memory_info}")

        # Return True if we're running low on memory
        return available_gb < 2.0 or percent_used > 85
    except:
        return False

def find_scenes(video_path, threshold=27.0):
    """Uses PySceneDetect to find scenes in a video."""
    print("[INFO] Detecting scenes with PySceneDetect...")
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    
    # Run the detection
    video_manager.set_downscale_factor()
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)
    
    scene_list = scene_manager.get_scene_list()
    print(f"[INFO] Detected {len(scene_list)} scenes.")
    video_manager.release()
    return scene_list


def calculate_flow_farneback(leaf_frames, scene_index=0):
    """
    Calculates DENSE optical flow for all frame pairs using the CPU-based
    Farneback algorithm. This is much faster than RAFT and uses no VRAM.
    
    It outputs a single tensor on the CPU, matching the output format
    of the calculate_flow_batched function.
    """
    
    num_pairs = len(leaf_frames) - 1
    print(f"  -> Processing {num_pairs} frame pairs using Farneback (CPU)...")

    if num_pairs < 1:
        return None

    # We need a grayscale version of the first frame to start
    try:
        prev_gray = cv2.cvtColor(leaf_frames[0]['frame_data'], cv2.COLOR_BGR2GRAY)
    except cv2.error as e:
        print(f"    [ERROR] Failed to convert frame to grayscale. Skipping scene. Error: {e}")
        return None
    
    all_flow_tensors = []
    
    # Loop from the second frame onwards
    for i in range(1, len(leaf_frames)):
        # Get the current frame in grayscale
        curr_gray = cv2.cvtColor(leaf_frames[i]['frame_data'], cv2.COLOR_BGR2GRAY)
        
        # --- Calculate Optical Flow ---
        # This is the core Farneback function. It outputs a NumPy array [H, W, 2]
        flow_numpy = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 
            0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Convert the [H, W, 2] NumPy array to a [2, H, W] PyTorch tensor (on CPU)
        flow_tensor = torch.from_numpy(flow_numpy).permute(2, 0, 1)
        all_flow_tensors.append(flow_tensor)
        
        # Set the current frame as the "previous" frame for the next iteration
        prev_gray = curr_gray

        if i%5 :
            print(f"    -> Progress: {i}/{num_pairs} pairs")

    if not all_flow_tensors:
        return None
        
    # Stack all [2, H, W] CPU tensors into a single [N-1, 2, H, W] CPU tensor
    return torch.stack(all_flow_tensors, dim=0)


def calculate_flow_batched(scene_frames, model, transforms, device, scene_index, batch_size=4):
    global processing_interrupted

    if len(scene_frames) < 2:
        print(f"  -> Scene {scene_index+1}: Skipping (less than 2 frames)")
        return None

    all_flow_outputs = []
    num_pairs = len(scene_frames) - 1
    print(f"  -> Scene {scene_index+1}: Processing {num_pairs} frame pairs for optical flow")

    # Start with conservative batch size, reduce if OOM occurs
    current_batch_size = min(batch_size, 4)  # Cap initial batch size
    processed_pairs = 0
    consecutive_failures = 0

    i = 0
    while i < num_pairs and not processing_interrupted:
        # Check memory pressure and interrupt state
        if i % 5 == 0:  # Check every 5 batches
            if check_memory_pressure():
                print(f"    -> High memory pressure detected, reducing batch size")
                current_batch_size = max(1, current_batch_size // 2)

            if processing_interrupted:
                print(f"    -> Processing interrupted, stopping optical flow calculation")
                break

        # Clear cache before each batch
        if device == "cuda":
            torch.cuda.empty_cache()

        try:
            # Calculate actual batch size for this iteration
            actual_batch_size = min(current_batch_size, num_pairs - i)
            if actual_batch_size <= 0:
                break

            current_batch_pairs = scene_frames[i : i + actual_batch_size + 1]

            # Process frame pairs more memory efficiently
            batch_img1_cpu_list = []
            batch_img2_cpu_list = []

            for j in range(len(current_batch_pairs) - 1):
                frame1_bgr = current_batch_pairs[j]['frame_data']
                frame2_bgr = current_batch_pairs[j+1]['frame_data']

                # Convert and normalize on CPU to save GPU memory
                frame1_rgb = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2RGB)
                frame2_rgb = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2RGB)

                img1_tensor = torch.from_numpy(frame1_rgb).permute(2, 0, 1).float()
                img2_tensor = torch.from_numpy(frame2_rgb).permute(2, 0, 1).float()

                batch_img1_cpu_list.append(img1_tensor)
                batch_img2_cpu_list.append(img2_tensor)

            if not batch_img1_cpu_list:
                i += actual_batch_size
                continue

            batch_img1_cpu = torch.stack(batch_img1_cpu_list)
            batch_img2_cpu = torch.stack(batch_img2_cpu_list)
            del batch_img1_cpu_list, batch_img2_cpu_list

            # Apply transforms on CPU if possible, or in smaller chunks
            img1_transformed, img2_transformed = transforms(batch_img1_cpu, batch_img2_cpu)
            del batch_img1_cpu, batch_img2_cpu

            # Move to GPU only when needed
            img1_gpu = img1_transformed.to(device, non_blocking=True)
            img2_gpu = img2_transformed.to(device, non_blocking=True)
            del img1_transformed, img2_transformed

            # Process optical flow
            with torch.no_grad():
                list_of_flows = model(img1_gpu, img2_gpu)
                predicted_flow_batch = list_of_flows[-1].cpu()

            del img1_gpu, img2_gpu, list_of_flows
            if device == "cuda":
                torch.cuda.empty_cache()

            all_flow_outputs.append(predicted_flow_batch)
            processed_pairs += predicted_flow_batch.shape[0]
            progress_pct = (processed_pairs / num_pairs) * 100
            print(f"    -> Progress: {processed_pairs}/{num_pairs} pairs ({progress_pct:.1f}%) - Batch size: {current_batch_size}")

            # Move to next batch
            i += actual_batch_size
            consecutive_failures = 0

            # Gradually increase batch size if we're doing well
            if consecutive_failures == 0 and current_batch_size < batch_size:
                current_batch_size = min(current_batch_size + 1, batch_size)

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            consecutive_failures += 1

            if "out of memory" in str(e).lower():
                print(f"    -> OOM with batch size {current_batch_size}, reducing to {max(1, current_batch_size//2)}")
            else:
                print(f"    -> Runtime error with batch size {current_batch_size}: {e}")

            # Aggressively reduce batch size
            current_batch_size = max(1, current_batch_size // 2)

            if device == "cuda":
                torch.cuda.empty_cache()

            # If we fail too many times consecutively, give up
            if consecutive_failures > 5 or current_batch_size == 0:
                print(f"    -> Too many consecutive failures. Stopping optical flow processing.")
                break

            # Don't increment i, retry this batch with smaller size
            continue

    if not all_flow_outputs:
        return None

    scene_flow_tensor = torch.cat(all_flow_outputs, dim=0)
    print(f"  -> Scene {scene_index+1}: Optical flow complete! Generated {scene_flow_tensor.shape[0]} flow fields")
    return scene_flow_tensor

def load_embedding_model(device):
    """
    Loads a pre-trained ResNet-50 and modifies it for feature extraction.
    """
    # 1. Load the pre-trained model and its required weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    
    # 2. Get the specific normalization transforms for this model
    transforms = weights.transforms()

    # 3. Modify the model to output an embedding (a 2048-dim vector)
    # We replace the final classification layer (model.fc) with an "Identity"
    # layer that just passes through the 2048-dim vector from the layer before it.
    model.fc = nn.Identity()

    # 4. Set model to evaluation mode and move to GPU
    model.eval()
    model.to(device)
    
    print("[INFO] ResNet-50 embedding model loaded.")
    return model, transforms


def get_embeddings_from_flow(scene_flow_tensor, model, transforms, device, scene_index=0):
    """
    Converts a stack of flow tensors into a stack of embedding vectors.
    
    Input: scene_flow_tensor (Shape: [N, 2, H, W])
    Output: scene_embeddings (Shape: [N, 2048])
    """
    
    # We must process the flow images one by one or in small batches
    # because flow_to_image and the transforms are not designed for large batches.
    # We'll do it one by one for simplicity.
    
    embedding_list = []
    num_flows = scene_flow_tensor.shape[0]
    print(f"  -> Scene {scene_index+1}: Converting {num_flows} flow fields to embeddings")
    
    # The input tensor has shape [N, 2, H, W], where N is (num_frames - 1)
    # We loop over the N dimension.
    for idx, flow_tensor in enumerate(scene_flow_tensor):
        if (idx + 1) % 10 == 0 or idx == 0 or idx == num_flows - 1:
            progress_pct = ((idx + 1) / num_flows) * 100
            print(f"    -> Embedding progress: {idx+1}/{num_flows} ({progress_pct:.1f}%)")
        # flow_tensor shape is [2, H, W]
        
        # 1. Add a batch dimension -> [1, 2, H, W]
        flow_batch = flow_tensor.unsqueeze(0) 
        
        # 2. Convert to a color image (on GPU) -> [1, 3, H, W]
        # flow_to_image expects a float tensor and outputs a uint8 tensor [0-255]
        flow_image_batch = flow_to_image(flow_batch.to(device))
        
        # 3. Apply the ImageNet transforms (normalization, etc.)
        # The transform function expects a uint8 tensor.
        normalized_batch = transforms(flow_image_batch) # Shape: [1, 3, H_resized, W_resized]
        
        # 4. Get the embedding
        with torch.no_grad():
            # embedding shape will be [1, 2048]
            embedding = model(normalized_batch) 
            
        # Add the embedding (removing the batch dim) to our list
        embedding_list.append(embedding.cpu())

    # Stack all [1, 2048] embeddings into a single [N, 2048] tensor
    scene_embeddings = torch.cat(embedding_list, dim=0)
    print(f"  -> Scene {scene_index+1}: Embedding generation complete! Shape: {scene_embeddings.shape}")
    return scene_embeddings


def save_scenes_to_files(video_path, scene_list, output_dir="video_scenes"):
    """
    Saves a list of detected scenes as individual video clips using FFmpeg.
    
    This function performs a codec copy (no re-encoding), so it is very fast
    and produces lossless-quality clips.
    
    Args:
        video_path (str): Path to the original source video.
        scene_list (list): The list of (start_FrameTimecode, end_FrameTimecode) tuples
                           provided by PySceneDetect.
        output_dir (str): The folder where the scene clips will be saved.
    """
    if not scene_list:
        print("[INFO] No scenes found in the list. Nothing to save.")
        return
    
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    else:
        # Loop over all the items in the directory
        for item_name in os.listdir(output_dir):
            item_path = os.path.join(output_dir, item_name)
            
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    # This is a file or a symlink, delete it
                    os.unlink(item_path) 
                elif os.path.isdir(item_path):
                    # This is a sub-directory, delete it and all its contents
                    shutil.rmtree(item_path) 
            except Exception as e:
                print(f"Failed to delete {item_path}. Reason: {e}")
            
    print(f"[INFO] Saving {len(scene_list)} scenes to '{output_dir}/'...")

    for i, (start_time, end_time) in enumerate(scene_list):
        scene_num = i + 1
        output_filename = os.path.join(output_dir, f"scene_{scene_num:03d}.mp4")
        
        # Get start time and duration in seconds
        start_sec = start_time.get_seconds()
        duration_sec = (end_time - start_time).get_seconds()

        try:
            # Build the FFmpeg command
            (
                ffmpeg
                .input(video_path, ss=start_sec)  # Seek to the start time
                .output(output_filename, t=duration_sec, c="copy")  # Set duration and copy codecs
                .run(overwrite_output=True, quiet=True) # Run quietly, overwriting old files
            )
            print(f"  -> Successfully saved {output_filename}")
            
        except ffmpeg.Error as e:
            print(f"[ERROR] Failed to save {output_filename}:")
            print(e.stderr.decode() if e.stderr else "Unknown FFmpeg error")
            
    print("[INFO] Done saving all scenes.")


def load_raft(option = SMALL_RAFT, device = "cpu"):
    if(option == SMALL_RAFT):
        weights = Raft_Small_Weights.DEFAULT
        model = raft_small(weights=weights, progress=False).to(device)
        return weights, model
    
    elif(option == BIG_RAFT):
        weights = Raft_Large_Weights.DEFAULT
        model = raft_large(weights=weights, progress=False).to(device)
        return weights, model


def create_leaf_clips_structure(video_path, scene_list, all_scene_trees, fps, output_base="out_clips"):
    """
    Creates video clips for all leaf nodes organized in a tree structure.
    
    Args:
        video_path: Path to original video
        scene_list: List of scene boundaries from PySceneDetect
        all_scene_trees: List of tree root nodes for each scene
        output_base: Base directory for outputs
    """
    print(f"\n[INFO] Creating leaf node clips structure in '{output_base}/'...")
    
    # Create base directory
    os.makedirs(output_base, exist_ok=True)
    
    total_clips_created = 0
    
    for i, (tree_root, (start_time, end_time)) in enumerate(zip(all_scene_trees, scene_list)):
        if tree_root is None:
            continue
            
        scene_dir = os.path.join(output_base, f"scene_{i+1:03d}")
        os.makedirs(scene_dir, exist_ok=True)
        
        # Get all leaf nodes from this tree
        leaf_nodes = []
        _collect_leaf_nodes(tree_root, leaf_nodes)
        
        print(f"  -> Scene {i+1}: Creating {len(leaf_nodes)} leaf node clips")
        
        # Create clips for each leaf node
        for leaf in leaf_nodes:
            # Convert frame numbers to time
            leaf_start_sec = leaf.start_frame / fps
            leaf_end_sec = leaf.end_frame / fps
            leaf_duration = leaf_end_sec - leaf_start_sec
            
            # Create filename with complexity info
            complexity = "complex" if leaf.is_complex else "simple"
            variance_str = f"{leaf.variance_score:.1f}" if leaf.variance_score is not None else "0.0"
            
            output_filename = os.path.join(
                scene_dir, 
                f"{leaf.node_id}_frames_{leaf.start_frame}-{leaf.end_frame}_{complexity}_V{variance_str}.mp4"
            )
            
            try:
                (
                    ffmpeg
                    .input(video_path, ss=leaf_start_sec)
                    .output(output_filename, t=leaf_duration, c="copy")
                    .run(overwrite_output=True, quiet=True)
                )
                total_clips_created += 1
                
            except ffmpeg.Error as e:
                print(f"    -> ERROR: Failed to create {output_filename}")
                print(f"       {e.stderr.decode() if e.stderr else 'Unknown FFmpeg error'}")
    
    print(f"[INFO] Created {total_clips_created} total leaf node clips")


def _collect_leaf_nodes(node, leaf_list):
    """Helper function to collect all leaf nodes from a tree."""
    if node.is_leaf():
        leaf_list.append(node)
    else:
        if node.left_child:
            _collect_leaf_nodes(node.left_child, leaf_list)
        if node.right_child:
            _collect_leaf_nodes(node.right_child, leaf_list)


def create_embedding_visualization(all_scene_trees, all_variance_results, fps, output_dir="out_clips"):
    """
    Creates visualizations of HTD analysis results.

    Args:
        all_scene_trees: List of tree root nodes for each scene
        all_variance_results: List of variance result dictionaries for each scene
        fps: Frames per second of the video
        output_dir: Directory to save visualization
    """
    print(f"\n[INFO] Creating HTD analysis visualizations...")

    # Collect metadata from all leaf nodes
    all_labels = []
    all_variances = []
    all_scene_ids = []
    all_durations = []
    all_frame_counts = []
    all_node_ids = []

    for scene_idx, tree_root in enumerate(all_scene_trees):
        if tree_root is None:
            continue

        # Collect leaf nodes
        leaf_nodes = []
        _collect_leaf_nodes(tree_root, leaf_nodes)

        for leaf in leaf_nodes:
            # Metadata
            all_labels.append("complex" if leaf.is_complex else "simple")
            all_variances.append(leaf.variance_score if leaf.variance_score is not None else 0.0)
            all_scene_ids.append(scene_idx + 1)
            all_durations.append((leaf.end_frame - leaf.start_frame) / fps)  # Use actual fps for duration
            all_frame_counts.append(leaf.end_frame - leaf.start_frame)
            all_node_ids.append(leaf.node_id)

    if len(all_labels) == 0:
        print("  -> No leaf nodes found for visualization")
        return

    print(f"  -> Collected {len(all_labels)} leaf nodes from {len(set(all_scene_ids))} scenes")

    # Create a comprehensive dashboard
    fig = plt.figure(figsize=(16, 12))

    # 1. Variance distribution histogram
    plt.subplot(2, 3, 1)
    complex_variances = [v for v, l in zip(all_variances, all_labels) if l == 'complex']
    simple_variances = [v for v, l in zip(all_variances, all_labels) if l == 'simple']

    plt.hist(simple_variances, bins=15, alpha=0.7, label='Simple nodes', color='blue', edgecolor='black')
    plt.hist(complex_variances, bins=15, alpha=0.7, label='Complex nodes', color='red', edgecolor='black')

    plt.xlabel('Variance Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Variance Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Scene-by-scene breakdown
    plt.subplot(2, 3, 2)
    scene_counts = {}
    scene_complex_counts = {}

    for scene_id, label in zip(all_scene_ids, all_labels):
        scene_counts[scene_id] = scene_counts.get(scene_id, 0) + 1
        if label == 'complex':
            scene_complex_counts[scene_id] = scene_complex_counts.get(scene_id, 0) + 1

    scenes = sorted(scene_counts.keys())
    total_per_scene = [scene_counts[s] for s in scenes]
    complex_per_scene = [scene_complex_counts.get(s, 0) for s in scenes]
    simple_per_scene = [total_per_scene[i] - complex_per_scene[i] for i in range(len(scenes))]

    plt.bar(scenes, simple_per_scene, color='blue', alpha=0.7, label='Simple')
    plt.bar(scenes, complex_per_scene, bottom=simple_per_scene, color='red', alpha=0.7, label='Complex')

    plt.xlabel('Scene Number')
    plt.ylabel('Number of Leaf Nodes')
    plt.title('Leaf Nodes per Scene')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3. Duration vs Variance scatter plot
    plt.subplot(2, 3, 3)
    colors = ['red' if label == 'complex' else 'blue' for label in all_labels]
    sizes = [50 if label == 'complex' else 20 for label in all_labels]

    plt.scatter(all_durations, all_variances, c=colors, s=sizes, alpha=0.7, edgecolors='black', linewidth=0.5)
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Variance Score')
    plt.title('Duration vs Variance Score')

    # Add complexity threshold line
    if len(all_variances) > 0:
        plt.axhline(y=100.0, color='orange', linestyle='--', alpha=0.8, label='Complexity Threshold')
        plt.legend(['Complex', 'Simple', 'Threshold'])

    plt.grid(True, alpha=0.3)

    # 4. Complexity percentage by scene
    plt.subplot(2, 3, 4)
    complexity_percentages = []
    for scene in scenes:
        total = scene_counts[scene]
        complex_count = scene_complex_counts.get(scene, 0)
        percentage = (complex_count / total) * 100 if total > 0 else 0
        complexity_percentages.append(percentage)

    bars = plt.bar(scenes, complexity_percentages, color='orange', alpha=0.7, edgecolor='black')
    plt.xlabel('Scene Number')
    plt.ylabel('Complex Nodes (%)')
    plt.title('Complexity Percentage by Scene')
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, pct in zip(bars, complexity_percentages):
        if pct > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{pct:.1f}%',
                    ha='center', va='bottom', fontsize=9)

    # 5. Node duration distribution
    plt.subplot(2, 3, 5)
    complex_durations = [d for d, l in zip(all_durations, all_labels) if l == 'complex']
    simple_durations = [d for d, l in zip(all_durations, all_labels) if l == 'simple']

    plt.hist(simple_durations, bins=15, alpha=0.7, label='Simple nodes', color='blue', edgecolor='black')
    plt.hist(complex_durations, bins=15, alpha=0.7, label='Complex nodes', color='red', edgecolor='black')

    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.title('Node Duration Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 6. Summary statistics text
    plt.subplot(2, 3, 6)
    plt.axis('off')

    total_nodes = len(all_labels)
    complex_nodes = sum(1 for l in all_labels if l == 'complex')
    simple_nodes = total_nodes - complex_nodes
    avg_duration = np.mean(all_durations)
    avg_variance = np.mean(all_variances)
    max_variance = np.max(all_variances) if all_variances else 0

    summary_text = f"""HTD Analysis Summary

Total Leaf Nodes: {total_nodes}
Complex Nodes: {complex_nodes} ({complex_nodes/total_nodes*100:.1f}%)
Simple Nodes: {simple_nodes} ({simple_nodes/total_nodes*100:.1f}%)

Average Duration: {avg_duration:.2f}s
Average Variance: {avg_variance:.2f}
Max Variance: {max_variance:.2f}

Scenes Processed: {len(scenes)}
VLM Efficiency: {(1-complex_nodes/total_nodes)*100:.1f}%
(Only {complex_nodes/total_nodes*100:.1f}% need expensive annotation)"""

    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()

    # Save the comprehensive dashboard
    dashboard_path = os.path.join(output_dir, "htd_analysis_dashboard.png")
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    print(f"  -> Saved HTD analysis dashboard to: {dashboard_path}")

    # Create individual variance distribution plot
    plt.figure(figsize=(10, 6))

    plt.hist(simple_variances, bins=20, alpha=0.7, label='Simple nodes', color='blue', edgecolor='black')
    plt.hist(complex_variances, bins=20, alpha=0.7, label='Complex nodes', color='red', edgecolor='black')

    plt.xlabel('Variance Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Variance Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save variance plot
    var_viz_path = os.path.join(output_dir, "variance_distribution.png")
    plt.savefig(var_viz_path, dpi=300, bbox_inches='tight')
    print(f"  -> Saved variance distribution to: {var_viz_path}")

    # Create a timeline visualization showing complexity over time
    plt.figure(figsize=(15, 8))

    # Sort nodes by start frame for timeline
    timeline_data = []
    for i, scene_idx in enumerate(all_scene_ids):
        # Get the actual leaf node to access frame numbers
        scene_tree = all_scene_trees[scene_idx - 1]
        if scene_tree:
            leaf_nodes = []
            _collect_leaf_nodes(scene_tree, leaf_nodes)
            for leaf in leaf_nodes:
                if leaf.node_id == all_node_ids[i]:
                    timeline_data.append({
                        'start_frame': leaf.start_frame,
                        'end_frame': leaf.end_frame,
                        'scene': scene_idx,
                        'complexity': all_labels[i],
                        'variance': all_variances[i],
                        'duration': all_durations[i]
                    })
                    break

    timeline_data.sort(key=lambda x: x['start_frame'])

    # Create timeline bars
    for i, data in enumerate(timeline_data):
        color = 'red' if data['complexity'] == 'complex' else 'blue'
        alpha = 0.8 if data['complexity'] == 'complex' else 0.6

        plt.barh(i, data['end_frame'] - data['start_frame'],
                left=data['start_frame'], color=color, alpha=alpha,
                height=0.8, edgecolor='black', linewidth=0.5)

        # Add scene labels
        if i == 0 or timeline_data[i-1]['scene'] != data['scene']:
            plt.text(data['start_frame'], i+0.5, f"Scene {data['scene']}",
                    fontsize=9, ha='left', va='center', weight='bold')

    plt.xlabel('Frame Number')
    plt.ylabel('Leaf Node Index')
    plt.title('HTD Timeline: Complexity Distribution Across Video\n(Red=Complex, Blue=Simple)')
    plt.grid(True, alpha=0.3, axis='x')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.8, label=f'Complex nodes ({complex_nodes})'),
        Patch(facecolor='blue', alpha=0.6, label=f'Simple nodes ({simple_nodes})')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    # Save timeline
    timeline_path = os.path.join(output_dir, "htd_timeline.png")
    plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
    print(f"  -> Saved HTD timeline to: {timeline_path}")

    plt.close('all')  # Close all figures to free memory

def process_leaf_node_flows(leaf_node, video_path, raft_model, raft_transforms, resnet_model, resnet_transforms, device, fps):
    """
    Process optical flow and embeddings for a single leaf node.
    This is where the actual OP-HTD processing happens.
    """
    global processing_interrupted

    if processing_interrupted:
        print(f"    -> Processing interrupted, skipping leaf {leaf_node.node_id}")
        return None, None

    # Extract frames for this leaf node
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    # Convert frame numbers to actual frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, leaf_node.start_frame)

    leaf_frames = []
    for frame_num in range(leaf_node.start_frame, leaf_node.end_frame):
        if processing_interrupted:
            break
        ret, frame = cap.read()
        if not ret:
            break
        leaf_frames.append({'frame_number': frame_num, 'frame_data': frame})

    cap.release()

    if len(leaf_frames) < 2 or processing_interrupted:
        return None, None

    # Clear CUDA cache before processing
    if device == "cuda":
        torch.cuda.empty_cache()

    scene_flow_tensor = None
    if FLOW_METHOD == "RAFT":
        scene_flow_tensor = calculate_flow_batched(
            leaf_frames, raft_model, raft_transforms, device,
            scene_index=0, batch_size=4  # Tune this (4, 2, or 1)
        )
    elif FLOW_METHOD == "FARNEBACK":
        scene_flow_tensor = calculate_flow_farneback(
            leaf_frames, scene_index=0
        )

    if scene_flow_tensor is None or processing_interrupted:
        return None, None

    # Get embeddings from flow
    leaf_embeddings = get_embeddings_from_flow(
        scene_flow_tensor, resnet_model, resnet_transforms, device, scene_index=0
    )

    # Clean up intermediate results
    del scene_flow_tensor
    if device == "cuda":
        torch.cuda.empty_cache()

    return leaf_embeddings, None  # Don't return flow tensor to save memory


def main(video_path, d_min=2.0, threshold_tau=100.0):
    # --- Setup signal handlers for graceful shutdown ---
    setup_signal_handlers()
    global processing_interrupted

    # --- Common Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Print memory info
    print(f"[INFO] Initial memory check:")
    check_memory_pressure()

    # Get video properties
    cap_temp = cv2.VideoCapture(video_path)
    fps = cap_temp.get(cv2.CAP_PROP_FPS)
    cap_temp.release()
    print(f"[INFO] Video FPS: {fps}")

    # Create directories for outputs
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    clips_dir = "out_clips"
    print(f"[INFO] Clips will be saved in: {clips_dir}")

    # --- Step 1: Detect All Scenes (SBD) ---
    if processing_interrupted:
        print("[INFO] Processing interrupted during initialization")
        return None, None, None

    scene_list = find_scenes(video_path)
    print(scene_list)
    if not scene_list:
        print("[ERROR] No scenes detected. Exiting.")
        return None, None, None

    # Save scene clips to disk (this is a fast copy operation)
    save_scenes_to_files(video_path, scene_list, clips_dir)


    # --- Step 2: Pre-cache all scene frames into CPU RAM ---
    print("\n[INFO] Pre-caching all scene frames into CPU RAM...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return
    
    all_scene_frames_data = []
    for i, (start_time, end_time) in enumerate(scene_list):
        start_frame = start_time.get_frames()
        end_frame = end_time.get_frames()

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        scene_frames = []
        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            scene_frames.append({'frame_number': frame_num, 'frame_data': frame})
        all_scene_frames_data.append(scene_frames)
    cap.release()
    print("[INFO] Frame caching complete.")

    # --- STAGE 1: Optical Flow (RAFT-only) ---
    print("\n--- STAGE 1: Calculating Optical Flow (RAFT Model Loaded) ---")

    # Conditionally load the RAFT model ONLY if we selected it
    raft_model, raft_transforms = None, None
    if FLOW_METHOD == "RAFT":
        print("[INFO] Loading RAFT model...")
        raft_weights, raft_model = load_raft(SMALL_RAFT, device)
        raft_transforms = raft_weights.transforms()
        raft_model.eval()
    else:
        print(f"[INFO] Using {FLOW_METHOD} (CPU) for optical flow. RAFT model will not be loaded.")
    
    
    all_flow_tensors_cpu = []
    
    for i, scene_frames in enumerate(all_scene_frames_data):
        print(f"\n[INFO] Processing Scene {i+1} (RAFT step)")
        
        scene_flow_tensor = None
        if FLOW_METHOD == "RAFT":
            scene_flow_tensor = calculate_flow_batched(
                scene_frames, raft_model, raft_transforms, device,
                scene_index=i, batch_size=1  # Tune this (4, 2, or 1)
            )
        elif FLOW_METHOD == "FARNEBACK":
            scene_flow_tensor = calculate_flow_farneback(
                scene_frames, scene_index=i
            )

        if scene_flow_tensor is not None:
            all_flow_tensors_cpu.append(scene_flow_tensor)
        else:
            print(f"  -> Scene {i+1} had no frame pairs, skipping.")
            all_flow_tensors_cpu.append(None)

    print("\n--- STAGE 1 COMPLETE ---")

    # --- STAGE 2: Unload RAFT from GPU ---
    print("[INFO] Unloading RAFT model from VRAM...")
    if FLOW_METHOD == "RAFT":
        del raft_weights, raft_model, raft_transforms
    torch.cuda.empty_cache()

       # --- STAGE 3: Embedding & Tree Building (ResNet-only) ---
    print("\n--- STAGE 2: Calculating Embeddings and Building Trees (ResNet Model Loaded) ---")
    resnet_model, resnet_transforms = load_embedding_model(device)
    
    # Store all results
    all_scene_trees = []
    all_variance_results = []
    all_complex_nodes = []
    
    for i, scene_flow_tensor in enumerate(all_flow_tensors_cpu):
        if processing_interrupted:
            print(f"\n[INFO] Processing interrupted. Stopping at scene {i+1}")
            break

        print(f"\n{'='*40}")
        print(f"PROCESSING SCENE {i+1}/{len(scene_list)}")
        print(f"{'='*40}")
        
        if scene_flow_tensor is not None:
            # Get embeddings
            scene_embeddings = get_embeddings_from_flow(scene_flow_tensor, resnet_model, resnet_transforms, device)
            print(f"  -> Got embedding set of shape: {scene_embeddings.shape}")

            # Get scene timing information
            start_time, end_time = scene_list[i]
            scene_start_frame = start_time.get_frames()
            scene_end_frame = end_time.get_frames()
            
            # Process scene with hierarchical tree
            root_node, variance_results, complex_leaves = process_scene_with_tree(
                scene_embeddings=scene_embeddings,
                scene_start_frame=scene_start_frame,
                scene_end_frame=scene_end_frame,
                fps=fps,
                scene_id=i+1,
                d_min_frames=d_min * fps,
                threshold_tau=threshold_tau
            )
            
            # Store results
            all_scene_trees.append(root_node)
            all_variance_results.append(variance_results)
            all_complex_nodes.extend(complex_leaves)  # Flatten across all scenes
            
        else:
            print(f"  -> Skipping scene {i+1} (no flow data).")
            all_scene_trees.append(None)
            all_variance_results.append({})

    print("\n--- STAGE 2 COMPLETE ---")
    
    # --- STAGE 4: Unload ResNet & Final Report ---
    del resnet_model, resnet_transforms
    torch.cuda.empty_cache

    # --- STAGE 5: Sparse VLM Annotation ---
    # This stage runs ONLY on the "complex" nodes after VRAM is cleared.
    if all_complex_nodes and not processing_interrupted:
        
        # !!! IMPORTANT !!! 
        # UPDATE THIS PATH to your downloaded 1.5B model directory
        VLM_MODEL_PATH = "model/llava-fastvithd_1.5b_stage3" 
        
       
        # Load the VLM model now that other models are unloaded
        vlm_model, vlm_tokenizer, vlm_image_processor = load_vlm_model(VLM_MODEL_PATH, device)
        
        annotate_complex_nodes_with_vlm(
            all_complex_nodes, video_path, 
            vlm_model, vlm_tokenizer, vlm_image_processor, 
            device
        )
        
        # Unload the VLM model to free memory before visualization
        print("[INFO] Unloading VLM model...")
        del vlm_model, vlm_tokenizer, vlm_image_processor
        torch.cuda.empty_cache()
        

    elif not all_complex_nodes:
        print("\n[INFO] No complex nodes found. Skipping VLM annotation stage.")
    else:
        print("\n[INFO] Processing was interrupted. Skipping VLM annotation stage.")

    # --- FINAL REPORTING ---
    print("\n" + "="*60)
    print("OP-HTD PIPELINE COMPLETE - SUMMARY")
    print("="*60)
    
    total_leaf_nodes = 0
    total_complex_nodes = len(all_complex_nodes)
    total_simple_nodes = 0

    # Convert scene list to frame-based boundaries
    scene_frame_boundaries = []
    for i, (start_time, end_time) in enumerate(scene_list):
        start_frame = start_time.get_frames()
        end_frame = end_time.get_frames()
        scene_frame_boundaries.append((start_frame, end_frame))
    
    for i, (tree, variance_results) in enumerate(zip(all_scene_trees, all_variance_results)):
        if tree is not None:
            leaf_count = len(variance_results)
            complex_count = sum(1 for v in variance_results.values() if v > threshold_tau)
            simple_count = leaf_count - complex_count
            
            total_leaf_nodes += leaf_count
            total_simple_nodes += simple_count
            
            # Get scene frame info for display
            scene_start_frame, scene_end_frame = scene_frame_boundaries[i]
            scene_frame_count = scene_end_frame - scene_start_frame
            scene_duration = scene_frame_count / fps
            
            print(f"Scene {i+1} (frames {scene_start_frame}-{scene_end_frame}, {scene_duration:.1f}s): {leaf_count} leaves ({complex_count} complex, {simple_count} simple)")
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total scenes: {len(scene_list)}")
    print(f"  Total leaf nodes: {total_leaf_nodes}")
    print(f"  Complex nodes (need VLM): {total_complex_nodes} ({total_complex_nodes/total_leaf_nodes*100:.1f}%)")
    print(f"  Simple nodes: {total_simple_nodes} ({total_simple_nodes/total_leaf_nodes*100:.1f}%)")
    print(f"  VLM efficiency: Only {total_complex_nodes/total_leaf_nodes*100:.1f}% of nodes need expensive annotation!")
    
    print(f"\nPARAMETERS USED:")
    print(f"  d_min: {d_min}s")
    print(f"  threshold_tau: {threshold_tau}")
    
    # List complex nodes that will need VLM annotation
    if all_complex_nodes:
        print(f"\nCOMPLEX NODES FOR VLM ANNOTATION ({len(all_complex_nodes)} total):")
        for node in all_complex_nodes[:10]:  # Show first 10
            print(f"  - {node.node_id}: frames {node.start_frame}-{node.end_frame} ({node.get_duration_seconds(fps):.2f}s) V={node.variance_score:.2f}")
        if len(all_complex_nodes) > 10:
            print(f"  ... and {len(all_complex_nodes) - 10} more")
    
    # --- VISUALIZATION STAGE ---
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Create leaf node clips organized by tree structure
    create_leaf_clips_structure(video_path, scene_list, all_scene_trees, fps, clips_dir)
    
    # Create HTD analysis visualizations
    create_embedding_visualization(all_scene_trees, all_variance_results, fps, clips_dir)
    
    print(f"\n[INFO] All outputs saved to: {clips_dir}/")
    print(f"  -> Scene clips: {clips_dir}/scene_XXX/")
    print(f"  -> Visualizations: {clips_dir}/*.png")
    
    return all_scene_trees, all_variance_results, all_complex_nodes


if __name__ == "__main__":
    # --- CHANGE THESE PARAMETERS ---
    VIDEO_PATH = "clips/small.mp4" 
    D_MIN_SECONDS = 3.0    # Minimum leaf duration in seconds (will be converted to frames)
    THRESHOLD_TAU = 55.0  # Variance threshold for complex/simple classification
    
    main(VIDEO_PATH, D_MIN_SECONDS, threshold_tau=THRESHOLD_TAU)