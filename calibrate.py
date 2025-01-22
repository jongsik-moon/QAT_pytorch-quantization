import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from pytorch_quantization import quant_modules
quant_modules.initialize()

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.calib import MaxCalibrator


from pytorch_quantization.tensor_quant import QuantDescriptor
quant_desc_input = QuantDescriptor(calib_method='histogram')
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

from resnet import resnet18


def collect_stats(model, data_loader, device="cuda", num_batches=None):
    model.eval()

    # Enable calibrators, disable actual quant
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    batch_count = 0
    with torch.no_grad():
        for i, (images, _) in enumerate(data_loader):
            images = images.to(device)

            _ = model(images)

            batch_count += 1
            if num_batches is not None and batch_count >= num_batches:
                break

    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, method='max', percentile=99.99):
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(method=method, percentile=percentile)
    return model


def ptq_calibration(
    device="cuda",
    calibration_samples=512,
    method='entropy',   # 'max', 'mse', or 'entropy'
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2023, 0.1994, 0.2010)
        ),
    ])
    full_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )

    indices = torch.randperm(len(full_dataset))[:calibration_samples]
    calib_subset = Subset(full_dataset, indices)
    calib_loader = DataLoader(
        calib_subset,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )


    model = resnet18(num_classes=10)
    checkpoint = torch.load('resnet18.pth')
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()


    collect_stats(model, calib_loader, device=device, num_batches=None)


    compute_amax(model, method=method)


    ckpt_path = f"resnet18_ptq.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[INFO] PTQ-calibrated model saved to: {ckpt_path}")

    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _ = ptq_calibration(
        device=device,
        calibration_samples=1024,
        method='percentile'  # or 'max', 'mse', etc.
    )
