import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Initializes CUDA driver
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def load_engine(engine_file_path: str) -> trt.ICudaEngine:
    """
    Loads a serialized TensorRT engine from file and returns a TensorRT ICudaEngine.
    """
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to load engine.")
    return engine

def allocate_buffers(engine: trt.ICudaEngine):
    """
    Allocate host/device buffers for input/output bindings.
    Returns (inputs, outputs, bindings, stream).
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        binding_idx = engine.get_binding_index(binding)
        shape = engine.get_binding_shape(binding_idx)
        size = abs(trt.volume(shape))
        dtype = trt.nptype(engine.get_binding_dtype(binding_idx))

        host_mem = cuda.pagelocked_empty(size, dtype=dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        if engine.binding_is_input(binding_idx):
            inputs.append({"host": host_mem, "device": device_mem, "shape": shape, "binding_idx": binding_idx})
        else:
            outputs.append({"host": host_mem, "device": device_mem, "shape": shape, "binding_idx": binding_idx})

        bindings.append(int(device_mem))

    return inputs, outputs, bindings, stream

def load_cifar10_testset(batch_size=1):
    """
    Returns a DataLoader for the CIFAR-10 test set.
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2)
    return test_loader

def infer_batch(
    context: trt.IExecutionContext,
    bindings,
    inputs,
    outputs,
    stream,
    batch_data: np.ndarray,
):
    """
    Copy a batch of data to the device, run inference, and return the output array.
    batch_data shape: (N, C, H, W), float32
    """
    inp = inputs[0]
    out = outputs[0]

    np.copyto(inp["host"], batch_data.ravel())

    cuda.memcpy_htod_async(inp["device"], inp["host"], stream)

    context.execute_async_v2(bindings, stream.handle)

    cuda.memcpy_dtoh_async(out["host"], out["device"], stream)

    stream.synchronize()

    batch_size = batch_data.shape[0]
    output_data = np.array(out["host"]).reshape(batch_size, 10)

    return output_data

def evaluate_engine(engine_file: str):
    """
    Load the engine from file, create context, evaluate on CIFAR-10 test set.
    """
    engine = load_engine(engine_file)

    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    test_loader = load_cifar10_testset(batch_size=1)

    total, correct = 0, 0

    for images, labels in test_loader:
        batch_size = images.size(0)
        batch_data = images.numpy().astype(np.float32)

        output_data = infer_batch(context, bindings, inputs, outputs, stream, batch_data)

        preds = np.argmax(output_data, axis=1)
        correct += np.sum(preds == labels.numpy())
        total += batch_size

    acc = 100.0 * correct / total
    return acc

def main():
    engine_file = "resnet18.engine"  

    # Evaluate
    accuracy = evaluate_engine(engine_file)
    print(f"TensorRT Engine Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()