import sys
import gc
import logging
import traceback

import torch
from docopt import docopt

from trainer import PPOTrainer
from yaml_parser import YamlParser


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def _cleanup_cuda(trainer=None):
    # Attempt to gracefully close trainer
    try:
        if trainer is not None and hasattr(trainer, "close"):
            trainer.close()
    except Exception:  # pragma: no cover - best effort cleanup
        logging.debug("trainer.close() raised:", exc_info=True)

    try:
        del trainer
    except Exception:  # pragma: no cover - best effort cleanup
        logging.debug("del trainer raised:", exc_info=True)
    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:  # pragma: no cover - best effort cleanup
            logging.debug("cuda.synchronize failed", exc_info=True)

        for i in range(torch.cuda.device_count()):
            try:
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:  # pragma: no cover - best effort cleanup
                logging.debug("cuda cleanup for device %d failed", i, exc_info=True)

        try:
            torch.set_default_tensor_type("torch.FloatTensor")
        except Exception:  # pragma: no cover - best effort cleanup
            logging.debug("set_default_tensor_type reset failed", exc_info=True)


def main():
    _USAGE = """
    Usage:
        train.py [options]
        train.py --help

    Options:
        --config=<path>            Path to the yaml config file [default: ./configs/cartpole.yaml]
        --run-id=<path>            Specifies the tag for saving the tensorboard summary [default: run].
        --cpu                      Force training on CPU [default: False]
    """
    options = docopt(_USAGE)
    run_id = options["--run-id"]
    force_cpu = options["--cpu"]

    trainer = None
    try:
        config = YamlParser(options["--config"]).get_config()
        if not force_cpu and torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
        else:
            device = torch.device("cpu")
            torch.set_default_tensor_type("torch.FloatTensor")

        logging.info("Starting training | run_id=%s | device=%s", run_id, device)

        trainer = PPOTrainer(config, run_id=run_id, device=device)
        trainer.run_training()

        logging.info("Training finished successfully")

    except KeyboardInterrupt:
        logging.warning("Training interrupted by user (Ctrl+C)")
    except Exception:
        logging.exception("Fatal error during training (run_id=%s)", run_id)
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        _cleanup_cuda(trainer)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted by user, shutting down...")
        sys.exit(130)
    except Exception:
        sys.exit(1)

