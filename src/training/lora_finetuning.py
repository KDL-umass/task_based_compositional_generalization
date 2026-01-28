import os
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from init import ROOT_DIR
from src.models.pretrained import (
    load_gpt_oss_20b,
    load_granite_2b,
    load_gemma_1b,
    load_llama3_8b,
    load_deepseek_coder_1_3b,
)
from src.training.utils import (
    configure_optimizers,
    log_eval,
    log_train,
    update_cosine_warmup_lr,
)

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
except ImportError as exc:
    raise ImportError(
        "peft is required for LoRA fine-tuning. Install with `pip install peft`."
    ) from exc

try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None


class LiveCodeBenchDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], tokenizer, cfg):
        self.samples = samples
        self.tokenizer = tokenizer
        self.cfg = cfg
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.samples)

    def _format_prompt(self, sample: Dict[str, Any]) -> str:
        code = sample.get("code", "").rstrip()
        inp = sample.get("input", "")
        instruction = (
            "You are given a Python function and an assertion containing an input to "
            "the function. Complete the assertion with a literal (no unsimplified "
            "expressions, no function calls) containing the output when executing the "
            "provided code on the given input, even if the function is incorrect or "
            "incomplete. Do NOT output any extra information. Provide the full "
            "assertion with the correct output in [ANSWER] and [/ANSWER] tags"
        )
        fewshot = (
            "[PYTHON]\n"
            "def repeatNumber(number : int) -> int:\n"
            "    return number\n"
            "assert repeatNumber(number = 17) == ??\n"
            "[/PYTHON]\n"
            "[ANSWER]\n"
            "assert repeatNumber(number = 17) == 17\n"
            "[/ANSWER]\n\n"
            "[PYTHON]\n"
            "def addCharacterA(string : str) -> str:\n"
            "    return string + \"a\"\n"
            "assert addCharacterA(string = \"x9j\") == ??\n"
            "[/PYTHON]\n"
            "[ANSWER]\n"
            "assert addCharacterA(string = \"x9j\") == \"x9ja\"\n"
            "[/ANSWER]\n\n"
        )
        suffix = "following the examples.\n\n" if self.cfg.include_fewshot else ".\n\n"
        prompt = (
            "[PYTHON]\n"
            f"{code}\n"
            f"assert {inp} == ??\n"
            "[/PYTHON]\n"
            "[ANSWER]\n"
        )
        if self.cfg.include_fewshot:
            return instruction + ", " + suffix + fewshot + prompt
        return instruction + suffix + prompt

    @staticmethod
    def _format_output_literal(output: Any) -> str:
        return repr(output)

    def _build_text_and_labels(self, sample: Dict[str, Any]) -> Tuple[List[int], List[int]]:
        prompt = self._format_prompt(sample)
        output_literal = self._format_output_literal(sample.get("output", ""))
        target = f"assert {sample.get('input', '')} == {output_literal}\n[/ANSWER]"

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        target_ids = self.tokenizer(target, add_special_tokens=False).input_ids

        max_seq_len = int(self.cfg.max_seq_len)
        if len(prompt_ids) + len(target_ids) > max_seq_len:
            if len(target_ids) >= max_seq_len:
                target_ids = target_ids[-max_seq_len:]
                prompt_ids = []
            else:
                keep_prompt = max_seq_len - len(target_ids)
                prompt_ids = prompt_ids[-keep_prompt:]

        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        return input_ids, labels

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        input_ids, labels = self._build_text_and_labels(self.samples[idx])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _collate_fn(batch, pad_id: int):
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for item in batch:
        ids = item["input_ids"]
        lab = item["labels"]
        pad_len = max_len - ids.size(0)
        if pad_len:
            ids = torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
            lab = torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)])
        input_ids.append(ids)
        labels.append(lab)
        attention_mask.append((ids != pad_id).long())
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(attention_mask),
    }


def _build_lora_config(cfg) -> LoraConfig:
    lora_cfg = cfg.lora
    target_modules = list(lora_cfg.target_modules) if lora_cfg.target_modules else None
    return LoraConfig(
        r=int(lora_cfg.r),
        lora_alpha=int(lora_cfg.alpha),
        lora_dropout=float(lora_cfg.dropout),
        bias=str(lora_cfg.bias),
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )


class LiveCodeBenchLoRAFineTuner:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.model = None
        self.tokenizer = None
        self._resume_dir = None
        self._resume_step = 0

    def _find_latest_checkpoint(self, output_dir: str) -> Optional[Tuple[str, int]]:
        if not os.path.isdir(output_dir):
            return None
        candidates = []
        for name in os.listdir(output_dir):
            if not name.startswith("step_"):
                continue
            parts = name.split("_", 1)
            if len(parts) != 2:
                continue
            try:
                step = int(parts[1])
            except ValueError:
                continue
            candidates.append((step, os.path.join(output_dir, name)))
        if not candidates:
            return None
        step, path = max(candidates, key=lambda x: x[0])
        return path, step

    def load_model_and_tokenizer(self):
        if self.cfg.model_name == "llama3":
            self.model, self.tokenizer, _ = load_llama3_8b()
        elif self.cfg.model_name == "gpt2":
            self.model, self.tokenizer, _ = load_gpt_oss_20b()
        elif self.cfg.model_name == "granite":
            self.model, self.tokenizer, _ = load_granite_2b()
        elif self.cfg.model_name == "gemma1":
            self.model, self.tokenizer, _ = load_gemma_1b()
        elif self.cfg.model_name == "deepseek_coder":
            self.model, self.tokenizer, _ = load_deepseek_coder_1_3b()
        else:
            raise ValueError(f"Model {self.cfg.model_name} not supported")

        if hasattr(self.model, "config") and hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        if getattr(self.cfg, "gradient_checkpointing", False):
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()

        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

        output_dir = self.get_output_dir()
        resume_enabled = getattr(self.cfg, "resume_if_available", True)
        resume_info = self._find_latest_checkpoint(output_dir) if resume_enabled else None

        if resume_info:
            self._resume_dir, self._resume_step = resume_info
            self.model = PeftModel.from_pretrained(
                self.model, self._resume_dir, is_trainable=True
            )
            if self.logger:
                self.logger.info(
                    "Resuming from adapter checkpoint: %s (step %d)",
                    self._resume_dir,
                    self._resume_step,
                )
        else:
            lora_config = _build_lora_config(self.cfg)
            self.model = get_peft_model(self.model, lora_config)

        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()

    def _resolve_data_path(self, split: str) -> str:
        if split == "train" and self.cfg.train_data_path:
            return self.cfg.train_data_path
        if split == "test" and self.cfg.test_data_path:
            return self.cfg.test_data_path
        base_dir = os.path.join(
            ROOT_DIR,
            "data",
            "livecodebench",
        )
        if getattr(self.cfg, "data_subdir", ""):
            base_dir = os.path.join(base_dir, str(self.cfg.data_subdir))
        return os.path.join(
            base_dir,
            f"holdout_{self.cfg.holdout_size}",
            f"seed_{self.cfg.data_seed}",
            f"{split}.pkl",
        )

    def load_dataloaders(self):
        train_path = self._resolve_data_path("train")
        test_path = self._resolve_data_path("test")
        with open(train_path, "rb") as f:
            train_samples = pickle.load(f)
        with open(test_path, "rb") as f:
            test_samples = pickle.load(f)

        self._train_dataset = LiveCodeBenchDataset(train_samples, self.tokenizer, self.cfg)
        self._test_dataset = LiveCodeBenchDataset(test_samples, self.tokenizer, self.cfg)
        self._pad_id = self.tokenizer.pad_token_id

        test_loader = DataLoader(
            self._test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=lambda b: _collate_fn(b, self._pad_id),
        )
        return test_loader

    def _build_train_loader(self, epoch: int) -> DataLoader:
        generator = torch.Generator()
        seed = int(getattr(self.cfg, "data_seed", 0)) + int(epoch)
        generator.manual_seed(seed)
        return DataLoader(
            self._train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=lambda b: _collate_fn(b, self._pad_id),
            generator=generator,
        )

    def get_output_dir(self):
        data_subdir = ""
        if getattr(self.cfg, "data_subdir", ""):
            data_subdir = str(self.cfg.data_subdir)
        elif getattr(self.cfg, "train_data_path", ""):
            marker = os.path.join("data", "livecodebench") + os.sep
            if marker in self.cfg.train_data_path:
                suffix = self.cfg.train_data_path.split(marker, 1)[-1]
                if os.sep + "holdout_" in suffix:
                    data_subdir = suffix.split(os.sep + "holdout_", 1)[0].strip(os.sep)

        output_dir = os.path.join(
            ROOT_DIR,
            "models",
            "ckpts",
            "livecodebench_lora",
            data_subdir if data_subdir else "default",
            f"holdout_{self.cfg.holdout_size}",
            f"seed_{self.cfg.data_seed}",
            f"model_{self.cfg.model_name}",
            f"lora_r{self.cfg.lora.r}_a{self.cfg.lora.alpha}",
        )
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _save_adapter(self, output_dir: str, step: Optional[int] = None, optimizer=None):
        tag = f"step_{step}" if step is not None else "final"
        save_dir = os.path.join(output_dir, tag)
        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_dir)
        if optimizer is not None:
            torch.save(
                {"optimizer": optimizer.state_dict(), "iter": step},
                os.path.join(save_dir, "trainer_state.pt"),
            )
        if OmegaConf is not None:
            OmegaConf.save(self.cfg, os.path.join(save_dir, "finetune_config.yaml"))

    def training_loop(self):
        self.load_model_and_tokenizer()
        test_loader = self.load_dataloaders()
        self.optimizer = configure_optimizers(self.model, self.cfg.optimizer)
        if self._resume_dir:
            state_path = os.path.join(self._resume_dir, "trainer_state.pt")
            if os.path.exists(state_path):
                state = torch.load(state_path, map_location="cpu")
                if "optimizer" in state:
                    self.optimizer.load_state_dict(state["optimizer"])
                if "iter" in state and state["iter"] is not None:
                    self._resume_step = int(state["iter"])
                if self.logger:
                    self.logger.info("Loaded optimizer state from %s", state_path)
        self.train(test_loader)

    def train(self, test_loader):
        self.logger.info("Starting LoRA fine-tuning for %d epochs", self.cfg.epochs)
        self.model.train()
        dt = torch.bfloat16 if self.cfg.bf16 else torch.float32
        device = self.cfg.device
        lr, it = 0.0, int(self._resume_step or 0)
        steps_per_epoch = (len(self._train_dataset) + self.cfg.batch_size - 1) // self.cfg.batch_size
        total_steps = steps_per_epoch * self.cfg.epochs
        train_loss = []
        output_dir = self.get_output_dir()
        save_interval = getattr(self.cfg.log, "save_interval", None)
        start_epoch = int(it // steps_per_epoch) if steps_per_epoch else 0
        start_batch = int(it % steps_per_epoch) if steps_per_epoch else 0
        if self.logger and it > 0:
            self.logger.info(
                "Resuming from iter %d (epoch %d, batch %d of %d)",
                it,
                start_epoch,
                start_batch,
                steps_per_epoch,
            )

        for epoch in range(start_epoch, self.cfg.epochs):
            train_loader = self._build_train_loader(epoch)
            self.logger.info(
                "Starting epoch %d (steps_per_epoch=%d, start_batch=%d)",
                epoch,
                steps_per_epoch,
                start_batch if epoch == start_epoch else 0,
            )
            for batch_idx, batch in enumerate(train_loader):
                if epoch == start_epoch and batch_idx < start_batch:
                    continue
                if batch_idx == 0 and self.logger:
                    self.logger.info("Fetched first batch for epoch %d", epoch)
                step_start = time.time()
                if self.logger and it == 0:
                    self.logger.info("Starting first training step")
                it, lr = update_cosine_warmup_lr(it, self.cfg.optimizer, self.optimizer, total_steps)
                self.optimizer.zero_grad(set_to_none=True)
                batch = {k: v.to(device) for k, v in batch.items()}
                if self.logger and it == 0:
                    self.logger.info("Running forward pass")
                with torch.amp.autocast(device_type=device, dtype=dt):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs.loss
                if self.logger and it == 0:
                    self.logger.info("Forward pass complete (loss=%.4f)", loss.item())
                train_loss.append(loss.item())
                if self.logger and it == 0:
                    self.logger.info("Running backward pass")
                loss.backward()
                if self.logger and it == 0:
                    self.logger.info("Backward pass complete")
                if self.cfg.optimizer.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.optimizer.grad_clip
                    )
                if self.logger and it == 0:
                    self.logger.info("Running optimizer step")
                self.optimizer.step()
                if self.logger and it == 0:
                    self.logger.info("Optimizer step complete")
                if self.logger and it % self.cfg.log.log_interval == 0:
                    self.logger.info("Step %d finished in %.2fs", it, time.time() - step_start)

                if it % self.cfg.log.log_interval == 0:
                    log_train(it, lr, train_loss)

                if it % self.cfg.log.eval_interval == 0:
                    eval_info = self.evaluate(test_loader, device, dt)
                    log_eval(it, lr, eval_info, logger=self.logger)

                if save_interval and it % save_interval == 0:
                    self._save_adapter(output_dir, step=it, optimizer=self.optimizer)

        eval_info = self.evaluate(test_loader, device, dt)
        log_eval(it, lr, eval_info, logger=self.logger)
        self._save_adapter(output_dir, step=None, optimizer=self.optimizer)

    @torch.no_grad()
    def evaluate(self, test_loader, device, dt):
        self.model.eval()
        total_loss = 0.0
        total = 0
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast(device_type=device, dtype=dt):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
            bs = batch["input_ids"].size(0)
            total_loss += loss.item() * bs
            total += bs

        avg_loss = total_loss / total if total else float("inf")
        self.model.train()
        return {"loss": {"test": avg_loss}}
