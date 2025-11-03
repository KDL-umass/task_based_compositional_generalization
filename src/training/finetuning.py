from src.data.corpus_generator.token_manager import DictionaryLoader
from src.data.loaders import get_data_loader, MappedSyntheticDataset
from src.models.pretrained import load_llama3_8b, load_gpt_oss_20b, load_granite_2b
from src.training.trainer import configure_optimizers, move_to_device, update_cosine_warmup_lr, log_train, log_eval, save_model
import torch
import torch.nn.functional as F


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
        for split in ["train", "train_heldout", "test"]:
            dataset = MappedSyntheticDataset(self.cfg.data_path, split=split, mode=self.cfg.mode, token_map=self.token_map)
            if split == "train":
                batch_size = self.cfg.batch_size
            else:
                batch_size = self.cfg.batch_size * 100
            loader = get_data_loader(dataset, batch_size, self.cfg.num_workers)
            loaders.append(loader)
        return loaders

        
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
        self.optimizer = configure_optimizers(self.model, self.cfg.optimizer)
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
                it, lr = update_cosine_warmup_lr(it, self.cfg.optimizer, self.optimizer, total_steps)
                # zero the gradients
                self.optimizer.zero_grad(set_to_none=True)
                # move the data to the device
                dat, targets = move_to_device(dat, targets, self.cfg.device)
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
                    log_train(it, lr, train_loss)

                # if it is time to evaluate
                if it % self.cfg.log.eval_interval == 0:
                    eval_info = self.evaluate(loaders[1:], device_info, sep_pos)
                    log_eval(it, lr, eval_info, logger=self.logger, sharp_acc=True)
                

        # log the final evaluation metrics
        eval_info = self.evaluate(loaders[1:], device_info, sep_pos)
        log_eval(it, lr, eval_info, logger=self.logger, sharp_acc=True)
        save_model(self.cfg, self.model, self.optimizer, it, self.cfg.output_dir)

    @torch.no_grad()
    def evaluate(self, evalLoaders, device_info, sep_pos):
        all_loss, all_acc, all_sharp_acc = [], [], []
        device, dt = device_info
        self.model.eval()
        for idx, split in enumerate(("train_heldout", "test")):
            loader = evalLoaders[idx]
            sequences, total_loss, total_acc, sharp_acc = 0.0, 0.0, 0.0, 0.0
            for dat, targets in loader:
                dat, targets = move_to_device(dat, targets, device)
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
                    total_acc += acc.float().mean().item() * bs
                    sharp_acc += acc.all(dim=-1).float().mean().item() * bs
                    
                sequences += bs
            if sequences == 0:
                    all_loss.append(float("inf"))
                    all_acc.append(float("inf"))
                    all_sharp_acc.append(float("inf"))
            else:
                all_loss.append(total_loss / sequences)
                all_acc.append(total_acc / sequences)
                all_sharp_acc.append(sharp_acc / sequences)
        info = {
            "train_loss": all_loss[0],
            "train_acc": all_acc[0],
            "test_loss": all_loss[1],
            "test_acc": all_acc[1],
            "train_sharp_acc": all_sharp_acc[0],
            "test_sharp_acc": all_sharp_acc[1],
        }
        self.model.train()
        return info