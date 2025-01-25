conda create -n faceapt python=3.10 -y
conda activate faceapt
ml cuDNN/8.7.0.84-CUDA-11.8.0
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install accelerate diffusers==0.26.0 insightface onnxruntime-gpu==1.18.0

huggingface-cli download FaceAdapter/FaceAdapter --local-dir ./checkpoints
# huggingface-cli download FaceAdapter/FaceAdapter --filename controlnet/config.json --local-dir ./checkpoints