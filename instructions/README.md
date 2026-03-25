
### Slime Env Setup

```bash
# cuda 12.9 (nvcc -V, nvidia-smi)
conda create --name openclaw-rl python=3.12 -y
conda activate openclaw-rl
 
pip install \
  torch==2.9.1+cu129 \
  torchvision==0.24.1+cu129 \
  torchaudio==2.9.1+cu129 \
  --index-url https://download.pytorch.org/whl/cu129
 
pip install -r /absolute/path/to/OpenClaw-RL/requirements.txt

# DeepEP
pip install -e /absolute/path/DeepEP --no-build-isolation

pip install -e /absolute/path/to/OpenClaw-RL/slime/slime/backends/megatron_utils/kernels/int4_qat --no-build-isolation
 
# apex
git clone https://github.com/NVIDIA/apex.git
cd apex
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .
cd ..

# flash_attn
export MAX_JOBS=8
pip install --no-build-isolation -v flash-attn==2.7.4.post1
 
# flashinfer
pip install "flashinfer-jit-cache==0.6.3" --index-url https://flashinfer.ai/whl/cu129

# megatron-bridge
pip install "megatron-bridge @ git+https://github.com/fzyzcjy/Megatron-Bridge.git@35b4ebfc486fb15dcc0273ceea804c3606be948a" --no-build-isolation

# TransformerEngine
export NVTE_FRAMEWORK=pytorch
pip install --no-build-isolation "transformer_engine[pytorch,core_cu12]==2.10.0"

# apt
apt-get update
apt-get install -y python3-apt

# upgrade to support Qwen3.5
pip install transformers==5.3.0
```


















