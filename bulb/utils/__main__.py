"""
A short test/demo for using utilities.
"""

from .torch_init_and_log import (
    get_logger2_args, init_gpus_and_randomness, Logger2)


if __name__ == "__main__":
    args = get_logger2_args()
    args.device = init_gpus_and_randomness(args.seed, args.gpu)
    logger = Logger2('/tmp/tmp', use_tensorboardX=True)
    logger.log_tb_object(args, 'args')
