import sys
import tensorrt as trt
import warnings
import os
import argparse

warnings.simplefilter("ignore", category=DeprecationWarning)


class EngineBuilder:
    def __init__(
        self,
        onnx_file_path,
        save_path,
        mode,
        log_level="ERROR",
        max_workspace_size=1,
        strict_type_constraints=False,
        int8_calibrator=None,
        **kwargs,
    ):
        """build TensorRT model from onnx model.
        Args:
            onnx_file_path (string or io object): onnx model name
            save_path (string): tensortRT serialization save path
            mode (string): Whether or not FP16 or Int8 kernels are permitted during engine build.
            log_level (string, default is ERROR): tensorrt logger level, now
                INTERNAL_ERROR, ERROR, WARNING, INFO, VERBOSE are support.
            max_workspace_size (int, default is 1):
                The maximum GPU temporary memory which the ICudaEngine can use at
                execution time. default is 1GB.
            strict_type_constraints (bool, default is False):
                When strict type constraints is set, TensorRT will choose
                the type constraints that conforms to type constraints.
                If the flag is not enabled higher precision
                implementation may be chosen if it results in higher performance.
            int8_calibrator (volksdep.calibrators.base.BaseCalibrator, default is None):
            calibrator for int8 mode,
                if None, default calibrator will be used as calibration data."""
        self.onnx_file_path = onnx_file_path
        self.save_path = save_path
        self.mode = mode.lower()
        assert self.mode in [
            "fp32",
            "fp16",
            "int8",
        ], f"mode should be in ['fp32', 'fp16', 'int8'], but got {mode}"

        self.trt_logger = trt.Logger(getattr(trt.Logger, log_level))
        self.builder = trt.Builder(self.trt_logger)
        self.network = None
        self.max_workspace_size = max_workspace_size
        self.strict_type_constraints = strict_type_constraints
        self.int8_calibrator = int8_calibrator

    def create_network(self, **kwargs):
        EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(EXPLICIT_BATCH)
        parser = trt.OnnxParser(self.network, self.trt_logger)
        if isinstance(self.onnx_file_path, str):
            with open(self.onnx_file_path, "rb") as f:
                print("Beginning ONNX file parsing")
                flag = parser.parse(f.read())
        else:
            flag = parser.parse(self.onnx_file_path.read())
        if not flag:
            for error in range(parser.num_errors):
                print(parser.get_error(error))

        print("Completed parsing of ONNX file.")
        # re-order output tensor
        output_tensors = [
            self.network.get_output(i) for i in range(self.network.num_outputs)
        ]

        [self.network.unmark_output(tensor) for tensor in output_tensors]

        for tensor in output_tensors:
            identity_out_tensor = self.network.add_identity(tensor).get_output(0)
            identity_out_tensor.name = "identity_{}".format(tensor.name)
            self.network.mark_output(tensor=identity_out_tensor)

    def create_engine(self):
        config = self.builder.create_builder_config()
        config.max_workspace_size = self.max_workspace_size * (1 << 25)
        if self.mode == "fp16":
            assert self.builder.platform_has_fast_fp16, "not support fp16"
            config.set_flag(trt.BuilderFlag.FP16)
            # builder.fp16_mode = True
        if self.mode == "int8":
            assert self.builder.platform_has_fast_int8, "not support int8"
            config.set_flag(trt.BuilderFlag.INT8)
            config.int8_calibrator = self.int8_calibrator
            # builder.int8_mode = True
            # builder.int8_calibrator = int8_calibrator

        if self.strict_type_constraints:
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        config.set_preview_feature(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805, True)

        print(
            f"Building an engine from file {onnx_file_path}; this may take a while..."
        )
        profile = self.builder.create_optimization_profile()

        config.add_optimization_profile(profile)

        engine = self.builder.build_engine(self.network, config)
        print("Create engine successfully!")

        print(f"Saving TRT engine file to path {self.save_path}")
        with open(self.save_path, "wb") as f:
            f.write(engine.serialize())
        print(f"Engine file has already saved to {self.save_path}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", help="The input ONNX model file to load")
    parser.add_argument(
        "--mode",
        default="fp16",
        help="use fp32 or fp16 or int8, default: fp16",
    )
    parser.add_argument("--output", default=None, help="The output TensorRT file name")
    parser.add_argument(
        "--workspace", default=1, type=int, help="The workspace size in bytes"
    )
    args = parser.parse_args()

    if not args.onnx:
        print("These arguments are required: --onnx")
        sys.exit(1)

    onnx_file_path = args.onnx
    if args.output is not None:
        work_path = onnx_file_path.rsplit("/", maxsplit=1)
        if len(work_path) > 1:
            engineFile = os.path.join(work_path[0], args.output)
        else:
            engineFile = args.output

    else:
        engineFile = onnx_file_path.replace(".onnx", ".engine")

    # 执行转化
    builder = EngineBuilder(
        onnx_file_path,
        engineFile,
        args.mode,
        log_level="WARNING",
        max_workspace_size=args.workspace,
    )
    builder.create_network()
    builder.create_engine()
