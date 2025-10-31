from src.data.corpus_generator.token_manager import DictionaryLoader
from src.data.loaders import get_data_loader, MappedSyntheticDataset
from src.models.pretrained import load_llama3_8b, load_gpt_oss_20b, load_granite_2b
import torch
import torch.nn.functional as F
import math
import inspect
import os

class FineTuner:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

    def load_model_and_tokenizer(self):
        if self.cfg.model_name == "llama3":
            self.model, self.tokenizer, self.config = load_llama3_8b()
        elif self.cfg.model_name == "gpt2":
            self.model, self.tokenizer, self.config = load_gpt_oss_20b()
        elif self.cfg.model_name == "granite":
            self.model, self.tokenizer, self.config = load_granite_2b()
        else:
            raise ValueError(f"Model {self.cfg.model_name} not supported")
        
    def load_dictionary_and_resize_tokenizer(self):
        self.dictionary = DictionaryLoader(self.cfg.data_path)
        additional_tokens = []
        self.token_map = {}
        # check if token has mapping in model tokenizer
        for token, idx in self.dictionary.token_idx_dict.items():
            model_token_to_idx = self.tokenizer.convert_tokens_to_ids(token)
            if model_token_to_idx is not None:
                self.token_map[idx] = model_token_to_idx
            else:
                additional_tokens.append(token)
        if additional_tokens:
            self.tokenizer.add_tokens(additional_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
        for token, idx in self.dictionary.token_idx_dict.items():
            if token not in self.token_map:
                self.token_map[idx] = self.tokenizer.convert_tokens_to_ids(token)
        

    def load_dataloaders(self):
        loaders = []
        train_dataset = MappedSyntheticDataset(self.cfg.data_path, split="train", mode=self.cfg.mode, token_map=self.token_map)
        train_loader = get_data_loader(train_dataset, self.cfg.batch_size, self.cfg.num_workers)
        val_dataset = MappedSyntheticDataset(self.cfg.data_path, split="train_heldout", mode=self.cfg.mode, token_map=self.token_map)
        val_loader = get_data_loader(val_dataset, self.cfg.batch_size, self.cfg.num_workers)
        test_dataset = MappedSyntheticDataset(self.cfg.data_path, split="test", mode=self.cfg.mode, token_map=self.token_map)
        test_loader = get_data_loader(test_dataset, self.cfg.batch_size, self.cfg.num_workers)
        loaders.append(train_loader)
        loaders.append(val_loader)
        loaders.append(test_loader)
        return loaders

    def configure_optimizers(self):
        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.cfg.optimizer.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and torch.cuda.is_available()
        extra_args = dict(fused=True) if use_fused else dict()
        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=float(self.cfg.optimizer.learning_rate),
            betas=(self.cfg.optimizer.beta1, self.cfg.optimizer.beta2),
            **extra_args,
        )
        print(f"using fused AdamW: {use_fused}")
        
    
    def move_to_device(self, dat, targets, device):
        if device == "cuda":
            dat = dat.pin_memory().cuda(non_blocking=True)
            targets = targets.pin_memory().cuda(non_blocking=True)
        return dat, targets

        
    def update_cosine_warmup_lr(self, it, total_steps):
        it += 1
        lr = float(self.cfg.optimizer.learning_rate)
        if self.cfg.optimizer.decay_lr:
            # if it is before the warmup period
            if it < self.cfg.optimizer.warmup_iters:
                lr = lr * (it) / self.cfg.optimizer.warmup_iters
            elif it > total_steps:
                lr = self.cfg.optimizer.min_lr
            else:
                # get the number of steps after the warmup period
                num = it - self.cfg.optimizer.warmup_iters
                decay_ratio = num / (total_steps - self.cfg.optimizer.warmup_iters)
                coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
                lr = self.cfg.optimizer.min_lr + coeff * (lr - self.cfg.optimizer.min_lr)

        # update the learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        return it, lr

    def training_loop(self):
        # first load the model and tokenizer
        self.load_model_and_tokenizer()
        # then load the dictionary and resize the tokenizer
        self.load_dictionary_and_resize_tokenizer()
        # then load the dataloaders after the tokenizer is resized
        loaders = self.load_dataloaders()
        train_loader = loaders[0]
        sep_pos = self.dictionary.get_sep_pos(train_loader.dataset.data[0])
        # then configure the optimizers
        self.configure_optimizers()
        self.train(train_loader, loaders, sep_pos)

    def train(self, train_loader, loaders, sep_pos):
        # set the model to training mode
        self.model.train()
        # set the device and data type
        dt = torch.bfloat16 if self.cfg.bf16 else torch.float32
        device_info = (self.cfg.device, dt)
        # initialize the learning rate and iteration
        lr, it = 0.0, 0
        # calculate the total number of steps
        total_steps = len(train_loader) * self.cfg.epochs
        train_loss = []
        # start the training loop
        for epoch in range(self.cfg.epochs):
            for dat, targets in train_loader:
                
                # update the learning rate
                it, lr = self.update_cosine_warmup_lr(it, total_steps)
                # zero the gradients
                self.optimizer.zero_grad(set_to_none=True)
                # move the data to the device
                dat, targets = self.move_to_device(dat, targets, self.cfg.device)
                # compute the loss
                with torch.amp.autocast(device_type=self.cfg.device, dtype=dt):
                    logits = self.model(dat)
                    loss = F.cross_entropy(logits.logits.reshape(-1, logits.logits.size(-1)), targets.reshape(-1))
                
                train_loss.append(loss.item())
                # backward the loss
                loss.backward()
                # clip the gradients
                if self.cfg.optimizer.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.optimizer.grad_clip
                    )
                # step the optimizer
                self.optimizer.step()

                # if it is time to log the training loss
                if it % self.cfg.log.log_interval == 0:
                    self.log_train(loss.item())

                # if it is time to evaluate
                if it % self.cfg.log.eval_interval == 0:
                    eval_info = self.evaluate(loaders[1:], device_info, sep_pos)
                    self.log_eval(it, lr, eval_info)
                

        # log the final evaluation metrics
        eval_info = self.evaluate(loaders[1:], device_info, sep_pos)
        self.log_eval(it, lr, eval_info)
        self.save_model(it, self.cfg.output_dir)

    @torch.no_grad()
    def evaluate(self, evalLoaders, device_info, sep_pos):
        all_loss, all_acc = [], []
        device, dt = device_info
        self.model.eval()
        for idx, split in enumerate(("train_heldout", "test")):
            loader = evalLoaders[idx]
            sequences, total_loss, total_acc = 0.0, 0.0, 0.0
            for dat, targets in loader:
                dat, targets = self.move_to_device(dat, targets, device)
                bs = dat.size(0)
                with torch.amp.autocast(device_type=device, dtype=dt):
                    # get the logits B*T*V
                    logits = self.model(dat)
                    logits = logits.logits[:, sep_pos:]
                    # get the targets
                    targets = targets[:, sep_pos:]
                    # reshape the logits and targets 
                    logits = logits.reshape(-1, logits.size(-1))
                    targets = targets.reshape(-1)
                    # compute the loss
                    loss = F.cross_entropy(logits, targets)
                    total_loss += loss.item() * bs
                    # compute the accuracy
                    acc = logits.argmax(-1) == targets
                    # calculate sharp accuracy
                    sharp_acc = acc.all(dim=-1).float().mean().item()
                    
                    # calculate total ood accuracy
                    total_acc += sharp_acc * bs
                sequences += bs
            if sequences == 0:
                    all_loss.append(float("inf"))
                    all_acc.append(float("inf"))
            else:
                all_loss.append(total_loss / sequences)
                all_acc.append(total_acc / sequences)
        info = {
            "heldout_loss": all_loss[0],
            "heldout_acc": all_acc[0],
            "test_loss": all_loss[1],
            "test_acc": all_acc[1],
        }
        self.model.train()
        return info

    def log_eval(self, it, lr, eval_info):
        print(f"Iteration {it}, Learning Rate {lr:.6f}")
        print(f"Heldout Loss: {eval_info['heldout_loss']:.6f}, Heldout Acc: {eval_info['heldout_acc']:.6f}")
        print(f"Test Loss: {eval_info['test_loss']:.6f}, Test Acc: {eval_info['test_acc']:.6f}")

    def log_train(self, train_loss):
        print(f"Train Loss: {train_loss:.6f}")

    def save_model(self, it, fdir):
        checkpoint = {
        "net": self.model.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "iter": it,
        "config": self.cfg,
        }
        fname = os.path.join(fdir, "ckpt_" + str(it + 1) + ".pt")
        torch.save(checkpoint, fname)
    