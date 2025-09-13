import cv2
import torch
import os
import torch.nn as nn
import ffmpeg
import shutil
from torchvision.models import resnet50, ResNet50_Weights
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from torchvision.utils import flow_to_image
from _htd_tree import process_scene_with_tree

SMALL_RAFT = 1
BIG_RAFT = 2

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


def calculate_flow_batched(scene_frames, model, transforms, device, scene_index, batch_size=32):
    if len(scene_frames) < 2:
        return None

    all_flow_outputs = []
    num_pairs = len(scene_frames) - 1
    
    # Start with requested batch size, reduce if OOM occurs
    current_batch_size = batch_size
    
    for i in range(0, num_pairs, current_batch_size):
        torch.cuda.empty_cache()
        
        try:
            current_batch_pairs = scene_frames[i : min(i + current_batch_size + 1, len(scene_frames))]
            
            # Your existing preprocessing code here...
            batch_img1_cpu_list = []
            batch_img2_cpu_list = []
            
            for j in range(len(current_batch_pairs) - 1):
                frame1_bgr = current_batch_pairs[j]['frame_data']
                frame2_bgr = current_batch_pairs[j+1]['frame_data']

                frame1_rgb = cv2.cvtColor(frame1_bgr, cv2.COLOR_BGR2RGB)
                frame2_rgb = cv2.cvtColor(frame2_bgr, cv2.COLOR_BGR2RGB)

                img1_tensor = torch.from_numpy(frame1_rgb).permute(2, 0, 1).float()
                img2_tensor = torch.from_numpy(frame2_rgb).permute(2, 0, 1).float()
                
                batch_img1_cpu_list.append(img1_tensor)
                batch_img2_cpu_list.append(img2_tensor)

            if not batch_img1_cpu_list:
                continue

            batch_img1_cpu = torch.stack(batch_img1_cpu_list)
            batch_img2_cpu = torch.stack(batch_img2_cpu_list)
            del batch_img1_cpu_list, batch_img2_cpu_list

            img1_transformed, img2_transformed = transforms(batch_img1_cpu, batch_img2_cpu)
            del batch_img1_cpu, batch_img2_cpu

            img1_gpu = img1_transformed.to(device)
            img2_gpu = img2_transformed.to(device)
            del img1_transformed, img2_transformed

            with torch.no_grad():
                list_of_flows = model(img1_gpu, img2_gpu)
                predicted_flow_batch = list_of_flows[-1].cpu()
                
            del img1_gpu, img2_gpu, list_of_flows
            torch.cuda.empty_cache()
            
            all_flow_outputs.append(predicted_flow_batch)
            
        except torch.cuda.OutOfMemoryError:
            # Reduce batch size and retry
            print(f"  -> OOM with batch size {current_batch_size}, reducing to {current_batch_size//2}")
            current_batch_size = max(1, current_batch_size // 2)
            torch.cuda.empty_cache()
            continue  # Retry this batch with smaller size

    if not all_flow_outputs:
        return None

    scene_flow_tensor = torch.cat(all_flow_outputs, dim=0)
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


def get_embeddings_from_flow(scene_flow_tensor, model, transforms, device):
    """
    Converts a stack of flow tensors into a stack of embedding vectors.
    
    Input: scene_flow_tensor (Shape: [N, 2, H, W])
    Output: scene_embeddings (Shape: [N, 2048])
    """
    
    # We must process the flow images one by one or in small batches
    # because flow_to_image and the transforms are not designed for large batches.
    # We'll do it one by one for simplicity.
    
    embedding_list = []
    
    # The input tensor has shape [N, 2, H, W], where N is (num_frames - 1)
    # We loop over the N dimension.
    for flow_tensor in scene_flow_tensor:
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

def main(video_path, d_min=2.0, threshold_tau=100.0):
    # --- Common Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Get video properties
    cap_temp = cv2.VideoCapture(video_path)
    fps = cap_temp.get(cv2.CAP_PROP_FPS)
    cap_temp.release()
    print(f"[INFO] Video FPS: {fps}")

    # Create directories for outputs
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    clips_dir = "out_clips"
    print(f"[INFO] Clips will be saved in: {clips_dir}")

    # --- Step 1: Detect All Scenes (CPU Task) ---
    scene_list = find_scenes(video_path)
    print(scene_list)
    if not scene_list:
        print("[ERROR] No scenes detected. Exiting.")
        return
    
    # Save clips to disk (this is a fast copy operation)
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
    raft_weights, raft_model = load_raft(SMALL_RAFT, device)
    raft_transforms = raft_weights.transforms()
    raft_model.eval()

    all_flow_tensors_cpu = []
    
    for i, scene_frames in enumerate(all_scene_frames_data):
        print(f"\n[INFO] Processing Scene {i+1} (RAFT step)")
        scene_flow_tensor = calculate_flow_batched(scene_frames, raft_model, raft_transforms, device, i, batch_size=1)
        
        if scene_flow_tensor is not None:
            all_flow_tensors_cpu.append(scene_flow_tensor)
        else:
            print(f"  -> Scene {i+1} had no frame pairs, skipping.")
            all_flow_tensors_cpu.append(None)

    print("\n--- STAGE 1 COMPLETE ---")

    # --- STAGE 2: Unload RAFT from GPU ---
    print("[INFO] Unloading RAFT model from VRAM...")
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
        print(f"\n[INFO] Processing Scene {i+1} (ResNet + Tree Building)")
        
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
    torch.cuda.empty_cache()

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
            print(f"  - {node.node_id}: {node.start_time:.2f}s-{node.end_time:.2f}s (V={node.variance_score:.2f})")
        if len(all_complex_nodes) > 10:
            print(f"  ... and {len(all_complex_nodes) - 10} more")
    
    return all_scene_trees, all_variance_results, all_complex_nodes


if __name__ == "__main__":
    # --- CHANGE THESE PARAMETERS ---
    VIDEO_PATH = "clips/med.mp4" 
    D_MIN_SECONDS = 3.0    # Minimum leaf duration in seconds (will be converted to frames)
    THRESHOLD_TAU = 100.0  # Variance threshold for complex/simple classification
    
    main(VIDEO_PATH, D_MIN_SECONDS, threshold_tau=THRESHOLD_TAU)