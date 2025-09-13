OP-HTD Video Analysis Pipeline

This project implements the OP-HTD pipeline for hierarchical video analysis. It detects scenes, builds a temporal tree, performs optical flow triage on leaf nodes, and (optionally) runs a VLM for sparse annotation on "complex" clips.

This code depends on the local repository ml-fastvlm by Apple and requires specific package versions to function.

1. Directory Setup

You must clone both this repository and the ml-fastvlm dependency. They should be in the same parent folder (e.g., Documents).
Bash

cd ~/Documents
git clone https://github.com/apple/ml-fastvlm.git
git clone https://github.com/tharun-selvam/htd_video.git 

Your directory structure should look like this:

Documents/
├── ml-fastvlm/
└── htd_video/    <-- (This repository)

2. Download VLM Checkpoints

The VLM model must be downloaded into the ml-fastvlm repository.
Bash

# Navigate into the ml-fastvlm repo
cd ~/Documents/ml-fastvlm

# Run their script to download the model checkpoints
# (This downloads all models, including the 1.5B variant)
download the 1.5B stage 3 variant

3. Environment Setup

All remaining commands should be run from inside this project's directory (htd_video).
Bash

# Navigate to this project directory
cd ~/Documents/htd_video

# 1. Create a local virtual environment
python3 -m venv venv_htd

# 2. Activate the environment
source venv_htd/bin/activate

4. Install Dependencies (Critical Step)

This project requires very specific package versions to resolve dependency conflicts between the llava library and opencv. Run the following commands to install everything in the correct order.
Bash

# 1. Install core libraries for the pipeline
pip install torch torchvision ffmpeg-python scenedetect[opencv] matplotlib psutil transformers timm einops

# 2. Install the specific conflicting versions required by llava
pip install numpy==1.26.4 scikit-learn==1.2.2

# 3. Install a compatible version of OpenCV
pip install opencv-python==4.8.1.78

# 4. Install the local llava library (from the other repo)
pip install -e ../ml-fastvlm/

5. Running the Pipeline

Before running, you must edit process_video.py and confirm two paths:

    VIDEO_PATH: Set this variable (at the bottom of the script) to point to your input video file.

    VLM_MODEL_PATH: Inside the main() function, verify this variable points to the correct 1.5B model checkpoint you downloaded in Step 2 (e.g., ../ml-fastvlm/checkpoints/llava-fastvithd_1.5b_stage3).

Once configured, run the script:
Bash

# Make sure your (venv_htd) environment is active!
python3 process_video.py