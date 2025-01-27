conda create -n faceapt python=3.10 -y
conda activate faceapt
ml cuDNN/8.7.0.84-CUDA-11.8.0
pip install -r requirements.txt

huggingface-cli download FaceAdapter/FaceAdapter --local-dir ./checkpoints
# huggingface-cli download FaceAdapter/FaceAdapter --filename controlnet/config.json --local-dir ./checkpoints