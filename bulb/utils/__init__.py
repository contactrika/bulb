from importlib.util import find_spec
try:
    find_spec('torch')
    from bulb.utils.torch_init_and_log import (
        Logger2, get_logger2_args, init_gpus_and_randomness)
except ImportError:
    pass  # not using torch-based code
