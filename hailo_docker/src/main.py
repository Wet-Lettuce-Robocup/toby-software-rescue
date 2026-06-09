# main.py, credit to: https://medium.com/@mgreiner79

import os

from hailo_sdk_client import ClientRunner
import numpy as np
import onnx
from PIL import Image

END_NODES = [
    '/model.22/cv2.0/cv2.0.2/Conv',
    '/model.22/cv3.0/cv3.0.2/Conv',
    '/model.22/cv2.1/cv2.1.2/Conv',
    '/model.22/cv3.1/cv3.1.2/Conv',
    '/model.22/cv2.2/cv2.2.2/Conv',
    '/model.22/cv3.2/cv3.2.2/Conv',
]


def parse_onnx(
    onnx_path: str,
    calib_folder: str,
    net_name: str,
    hw_arch: str,
    num_classes: int,
    target_size: tuple[int, int] = (640, 640),
):
    # -------------------------------
    # Step 1. Load the ONNX model and test outputs
    # -------------------------------

    # Load the ONNX model
    model = onnx.load(onnx_path)
    print([o.name for o in model.graph.output])  # Print output node names for reference

    # -----------------------------------------------------
    # Step 2. Translate the ONNX model to Hailo format
    # -----------------------------------------------------

    # Create a ClientRunner instance.
    # The hw_arch parameter should match your target Hailo device (e.g., "hailo8")
    runner = ClientRunner(hw_arch=hw_arch, har=None)

    # Translate the ONNX model. Optionally, you can supply start and end node names if needed.
    hn, params = runner.translate_onnx_model(
        model_path=onnx_path,
        net_name=net_name,
        # end_nodes_names=END_NODES,
    )
    print('Model translation to Hailo format completed.')

    har_file = f'./models/{net_name}_raw.har'
    runner.save_har(har_file)
    print(f'Raw HAR file saved to: {har_file}')
    # -----------------------------------------------------
    # Step 3. Quantize the model using a calibration dataset
    # -----------------------------------------------------
    # For quantization, you need a calibration dataset.
    # Adjust the shape (batch, height, width, channels) as required by your model.
    # For many YOLO models the expected input is 640x640 with 3 channels.
    calib_dataset = load_calibration_dataset(calib_folder, target_size)
    print('Calibration dataset created.')

    # Run optimization (quantization). This process uses the calibration dataset to
    # convert floating-point parameters into their quantized (integer) counterparts.
    runner.optimize(calib_dataset)
    print('Model quantization complete.')

    # -----------------------------------------------------
    # Step 4. Compile the quantized model into a HAR file for deployment
    # -----------------------------------------------------
    hef = runner.compile()
    output_hef_path = f'./models/{net_name}.hef'
    with open(output_hef_path, 'wb') as f:
        f.write(hef)


def load_calibration_dataset(
    calib_folder: str, target_size: tuple[int, int] = (640, 640)
) -> np.ndarray:
    """
    Load and preprocess images from the specified folder.

    Args
    ----
        calib_folder : str
            Path to the folder with calibration images.
        target_size : tuple[int, int]
            Desired image size as (width, height).

    Returns
    -------
        numpy.ndarray
            Array of shape (num_images, height, width, 3) with dtype float32.

    """
    image_files = [
        os.path.join(calib_folder, f)
        for f in os.listdir(calib_folder)
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]
    if not image_files:
        raise ValueError('No calibration images found in the folder.')

    images = []
    for img_file in sorted(image_files):
        img = Image.open(img_file).convert('RGB')
        # Resize using bilinear interpolation
        img = img.resize(target_size, Image.BILINEAR)  # does this work
        img_np = np.array(img).astype(np.float32)
        images.append(img_np)

    calib_dataset = np.stack(images, axis=0)
    print(f'Loaded {calib_dataset.shape[0]} calibration images of size {target_size}.')
    return calib_dataset


def main():
    onnx_path = './models/best.onnx'
    calib_folder = './calibration_images'
    net_name = 'robotyolov8s'
    hw_arch = 'hailo10h'
    num_classes = 1

    parse_onnx(onnx_path, calib_folder, net_name, hw_arch, num_classes=num_classes)


if __name__ == '__main__':
    main()
