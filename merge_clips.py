import os
import glob
import ffmpeg
import sys

def merge_clips_in_folder(folder_path, file_extension="mp4", output_filename="merged_output.mp4"):
    """
    Finds all video files with a given extension in a folder, sorts them 
    alphabetically, and merges them into a single video file without re-encoding.
    """
    
    list_file = "mergelist.txt"
    output_path = os.path.join(folder_path, output_filename)

    # --- 1. Find and sort all video files ---
    print(f"[INFO] Finding all .{file_extension} files in {folder_path}...")
    search_path = os.path.join(folder_path, f"*.{file_extension}")
    clip_files = sorted(glob.glob(search_path))

    if not clip_files:
        print(f"Error: No .{file_extension} files found in the specified folder.")
        return

    print(f"Found {len(clip_files)} clips to merge.")

    # --- 2. Create the temporary merge list file for FFmpeg ---
    # FFmpeg's concat demuxer requires a specific text file format:
    # file '/path/to/file1.mp4'
    # file '/path/to/file2.mp4'
    try:
        with open(list_file, 'w') as f:
            for clip in clip_files:
                # Add 'file' and wrap the absolute path in single quotes
                f.write(f"file '{os.path.abspath(clip)}'\n")
        
        print("[INFO] Temporary merge list created.")

        # --- 3. Run the FFmpeg command using the Python wrapper ---
        print("[INFO] Merging videos with FFmpeg...")
        (
            ffmpeg
            .input(list_file, format='concat', safe=0)
            .output(output_path, c='copy')  # 'c=copy' copies streams, no re-encoding
            .run(overwrite_output=True, quiet=True)
        )
        
        print(f"\n[SUCCESS] All clips merged into: {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # --- 4. Clean up the temporary list file ---
        if os.path.exists(list_file):
            os.remove(list_file)
            print("[INFO] Cleaned up temporary files.")

# --- How to use this script ---
if __name__ == "__main__":
    # 1. SET THE PATH to the folder containing your clips
    # This will merge clips inside the 'BigBuckBunny_optical_flow' folder
    FOLDER_TO_MERGE = "clips_new"
    
    # 2. SET THE OUTPUT FILENAME
    OUTPUT_FILENAME = "small.mp4"
    
    # Run the function
    merge_clips_in_folder(FOLDER_TO_MERGE, file_extension="mp4", output_filename=OUTPUT_FILENAME)