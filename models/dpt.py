import numpy as np
import cv2
from utils_function import load_image
import argparse
import os
import platform
from loguru import logger


if platform.system() != "Darwin":
    import warnings
    import pycuda.driver as cuda
    import tensorrt as trt

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    TRT_LOGGER = trt.Logger()
    TRT_LOGGER.min_severity = trt.Logger.Severity.ERROR
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")


class HostDeviceMem:
    """
    Host and Device Memory Package
    """

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class Dpt:
    """Depth Anything v2 inference class"""

    def __init__(self, args) -> None:
        self.reshape_size = [518, 518]
        self.model_path = args.engine
        self.__trt_init__(
            self.model_path,
            dynamic_shape=False,
            batch_size=1,
        )

    def __trt_init__(self, trt_file=None, dynamic_shape=False, gpu_idx=0, batch_size=1):
        """
        Init tensorrt.
        :param trt_file:    tensorrt file.
        :return:
        """
        cuda.init()
        self._batch_size = batch_size
        self._device_ctx = cuda.Device(gpu_idx).make_context()
        self._engine = self._load_engine(trt_file)
        self._context = self._engine.create_execution_context()
        if not dynamic_shape:
            (
                self._input,
                self._output,
                self._bindings,
                self._stream,
            ) = self._allocate_buffers(self._context)

        logger.info("Dpt model <loaded>...")

    def _load_engine(self, trt_file):
        """
        Load tensorrt engine.
        :param trt_file:    tensorrt file.
        :return:
            ICudaEngine
        """
        with open(trt_file, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def _allocate_buffers(self, context):
        """
        Allocate device memory space for data.
        :param context:
        :return:
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        for binding in self._engine:
            size = (
                trt.volume(self._engine.get_binding_shape(binding))
                * self._engine.max_batch_size
            )
            dtype = trt.nptype(self._engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self._engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def trt_infer(self, data):
        """
        Real inference process.
        :param model:   Model objects
        :param data:    Preprocessed data
        :return:
            output
        """
        # Copy data to input memory buffer
        [np.copyto(_inp.host, data.ravel()) for _inp in self._input]
        # Push to device
        self._device_ctx.push()
        # Transfer input data to the GPU.
        # cuda.memcpy_htod_async(self._input.device, self._input.host, self._stream)
        [
            cuda.memcpy_htod_async(inp.device, inp.host, self._stream)
            for inp in self._input
        ]
        # Run inference.
        self._context.execute_async_v2(
            bindings=self._bindings, stream_handle=self._stream.handle
        )
        # Transfer predictions back from the GPU.
        # cuda.memcpy_dtoh_async(self._output.host, self._output.device, self._stream)
        [
            cuda.memcpy_dtoh_async(out.host, out.device, self._stream)
            for out in self._output
        ]
        # Synchronize the stream
        self._stream.synchronize()
        # Pop the device
        self._device_ctx.pop()

        return [out.host.reshape(self._batch_size, -1) for out in self._output[::-1]]

    def preprocess(self, im):
        """Preprocess core
        :param im:   numpy.ndarray
        :return:
            im, origin_shape_info
        """
        im, (orig_h, orig_w) = load_image(im)
        return im, (orig_w, orig_h)

    def inference(self, input_frame_array) -> np.ndarray:
        """Inference core
        :param input_frame_array:   numpy.ndarray
        :return:
            net_out
        """
        trt_inputs = [input_frame_array]
        trt_inputs = np.vstack(trt_inputs)
        result = self.trt_infer(trt_inputs)
        net_out = result[0].reshape(self.reshape_size[0], self.reshape_size[1])
        return net_out

    def postprocess(self, shape_info: tuple, depth: np.ndarray) -> np.ndarray:
        """Postprocess core
        :param shape_info:   tuple
        :param depth:        numpy.ndarray
        :return:
            net_out
        """
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = cv2.resize(depth, (shape_info[0], shape_info[1]))
        return depth

    def run(
        self,
        input_frame_array: np.ndarray,
    ) -> np.ndarray:
        input_frame_array, shape_info = model.preprocess(input_frame_array)
        depth_res = model.inference(input_frame_array)
        depth_img = model.postprocess(shape_info, depth_res)

        return depth_img

    def __del__(self):
        self._device_ctx.pop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run depth estimation with a TensorRT engine."
    )
    parser.add_argument(
        "--img", type=str, required=True, help="Path to the input image"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./vis_depth",
        help="Output directory for the depth map",
    )
    parser.add_argument(
        "--engine", type=str, required=True, help="Path to the TensorRT engine"
    )
    parser.add_argument(
        "--grayscale", action="store_true", help="Save the depth map in grayscale"
    )

    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    image_path = args.img
    image = cv2.imread(image_path)

    # Load the model
    model = Dpt(args)
    image, shape_info = model.preprocess(image)
    depth_res = model.inference(image)
    depth_img = model.postprocess(shape_info, depth_res)

    # Save the results
    img_name = os.path.basename(args.img)
    if args.grayscale:
        cv2.imwrite(
            f'{args.outdir}/{img_name[:img_name.rfind(".")]}_depth.png', depth_img
        )
    else:
        colored_depth = cv2.applyColorMap(depth_img, cv2.COLORMAP_INFERNO)
        cv2.imwrite(
            f'{args.outdir}/{img_name[:img_name.rfind(".")]}_depth.png', colored_depth
        )
