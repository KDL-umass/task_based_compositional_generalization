import logging

from init import read_config, set_seed, ROOT_DIR
from src.training.finetuning_livecodebench import LiveCodeBenchFineTuner


def main():
    cfg = read_config(f"{ROOT_DIR}/config/finetune/livecodebench.yaml")
    set_seed(cfg.seed)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger = logging.getLogger("finetune_livecodebench")
    finetuner = LiveCodeBenchFineTuner(cfg=cfg, logger=logger)
    finetuner.training_loop()


if __name__ == "__main__":
    main()
