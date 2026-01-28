import os
import pickle
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from init import ROOT_DIR
from src.models.pretrained import (
    load_gpt_oss_20b,
    load_granite_2b,
    load_gemma_1b,
    load_llama3_8b,
)
from src.training.utils import (
    configure_optimizers,
    log_eval,
    log_train,
    update_cosine_warmup_lr,
)


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


class LiveCodeBenchFineTuner:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.model = None
        self.tokenizer = None

    def load_model_and_tokenizer(self):
        if self.cfg.model_name == "llama3":
            self.model, self.tokenizer, _ = load_llama3_8b()
        elif self.cfg.model_name == "gpt2":
            self.model, self.tokenizer, _ = load_gpt_oss_20b()
        elif self.cfg.model_name == "granite":
            self.model, self.tokenizer, _ = load_granite_2b()
        elif self.cfg.model_name == "gemma1":
            self.model, self.tokenizer, _ = load_gemma_1b()
        else:
            raise ValueError(f"Model {self.cfg.model_name} not supported")

    def _resolve_data_path(self, split: str) -> str:
        if split == "train" and self.cfg.train_data_path:
            return self.cfg.train_data_path
        if split == "test" and self.cfg.test_data_path:
            return self.cfg.test_data_path
        return os.path.join(
            ROOT_DIR,
            "data",
            "livecodebench",
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

        train_dataset = LiveCodeBenchDataset(train_samples, self.tokenizer, self.cfg)
        test_dataset = LiveCodeBenchDataset(test_samples, self.tokenizer, self.cfg)
        pad_id = self.tokenizer.pad_token_id

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            collate_fn=lambda b: _collate_fn(b, pad_id),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            collate_fn=lambda b: _collate_fn(b, pad_id),
        )
        return train_loader, test_loader

    def get_output_dir(self):
        output_dir = os.path.join(
            ROOT_DIR,
            "models",
            "ckpts",
            "livecodebench",
            f"holdout_{self.cfg.holdout_size}",
            f"seed_{self.cfg.data_seed}",
            f"model_{self.cfg.model_name}",
        )
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def training_loop(self):
        self.load_model_and_tokenizer()
        train_loader, test_loader = self.load_dataloaders()
        self.optimizer = configure_optimizers(self.model, self.cfg.optimizer)
        self.train(train_loader, test_loader)

    def train(self, train_loader, test_loader):
        self.model.train()
        dt = torch.bfloat16 if self.cfg.bf16 else torch.float32
        device = self.cfg.device
        lr, it = 0.0, 0
        total_steps = len(train_loader) * self.cfg.epochs
        train_loss = []

        for epoch in range(self.cfg.epochs):
            for batch in train_loader:
                it, lr = update_cosine_warmup_lr(it, self.cfg.optimizer, self.optimizer, total_steps)
                self.optimizer.zero_grad(set_to_none=True)
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.amp.autocast(device_type=device, dtype=dt):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = outputs.loss
                train_loss.append(loss.item())
                loss.backward()
                if self.cfg.optimizer.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.optimizer.grad_clip
                    )
                self.optimizer.step()

                if it % self.cfg.log.log_interval == 0:
                    log_train(it, lr, train_loss)

                if it % self.cfg.log.eval_interval == 0:
                    eval_info = self.evaluate(test_loader, device, dt)
                    log_eval(it, lr, eval_info, logger=self.logger)

        eval_info = self.evaluate(test_loader, device, dt)
        log_eval(it, lr, eval_info, logger=self.logger)

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
