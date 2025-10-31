
import logging
import os

from init import read_config, set_seed, ROOT_DIR
from src.training.finetuning import FineTuner

def setup_logging(output_dir):
    """Setup logging configuration."""
    log_file = os.path.join(output_dir, "finetuning_run.log")
    # create the output directory if it doesn't exist
    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)




def main():
   cfg = read_config(f"{ROOT_DIR}/config/finetune/conf.yaml")
   set_seed(cfg.seed)
   logger = setup_logging(cfg.output_dir)
   # create the fine-tuner
   finetuner = FineTuner(cfg=cfg, logger=logger)
   # start the training loop
   finetuner.training_loop()


if __name__ == "__main__":
    main()

