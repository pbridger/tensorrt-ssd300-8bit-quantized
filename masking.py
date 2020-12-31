import sys, io
import torch
import tensorrt as trt

class Masking(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X):
        X = X[X.sum(dim=-1) > 0]
        return X


if __name__ == '__main__':
    m = Masking()
    onnx_filename = 'models/masking.onnx'

    print('exporting Masking to', onnx_filename)
    torch.onnx.export(
        Masking(),
        torch.randn((10, 10)),
        onnx_filename,
        opset_version=11
    )

    print('compiling', onnx_filename, 'with TensorRT')
    logger = trt.Logger()
    network_flags = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(logger) as builder, builder.create_network(network_flags) as network, trt.OnnxParser(network, logger) as parser:
        if not parser.parse(open(onnx_filename, 'rb').read()):
            sys.exit(1)
        engine = builder.build_cuda_engine(network)

