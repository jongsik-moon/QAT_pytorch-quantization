from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
quant_nn.TensorQuantizer.use_fb_fake_quant = True
quant_modules.initialize()

import torch
from resnet import resnet18

def export_to_onnx(model, onnx_filename, device):
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_filename,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True
    )
    print(f"ONNX model exported to {onnx_filename}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet18(num_classes=10)
    model.to(device)
    checkpoint = torch.load('resnet18_ptq.pth')
    model.load_state_dict(checkpoint)
    model.eval()

    export_to_onnx(model, "resnet18_ptq.onnx", device)