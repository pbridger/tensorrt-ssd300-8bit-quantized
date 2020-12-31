FROM nvcr.io/nvidia/pytorch:20.10-py3

RUN python -c "import torch; torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math='fp32')" 2>/dev/null | :
RUN python -c "import torch; torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math='fp16')" 2>/dev/null | :

# Nvidia Apex for mixed-precision inference
RUN git clone https://github.com/NVIDIA/apex.git /build/apex
WORKDIR /build/apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

# use pbridger fork instead of NV repo - only change is NMS plugin updated to handle FP16
WORKDIR /build
RUN git clone --single-branch --branch release/7.2 https://github.com/pbridger/TensorRT.git
WORKDIR /build/TensorRT
RUN mkdir build && git submodule update --init --recursive
WORKDIR /build/TensorRT/build
# set GPU_ARCHS to match your GPU architecture, though this repo only makes sense for >= Volta (70)
RUN cmake .. -DTRT_LIB_DIR=`pwd`/lib -DTRT_OUT_DIR=`pwd`/out -DGPU_ARCHS="75" -DBUILD_SAMPLES=OFF
RUN make -j$(nproc) && make install && cp lib/* /usr/lib/x86_64-linux-gnu/

WORKDIR /build/TensorRT/tools/onnx-graphsurgeon
RUN make install

RUN pip install pycuda

WORKDIR /app
