import torch
import torch.nn.functional as F
import os
from src.data.corpus_generator.token_manager import TokenManager
from src.data.loaders import get_data_loader, MappedSyntheticDataset
from src.models.pretrained import load_llama3_8b, load_gpt_oss_20b, load_granite_2b, load_gemma_1b
from src.training.utils import configure_optimizers, move_to_device, update_cosine_warmup_lr, log_train, log_eval, save_model
from src.utils.storage_utils import get_directory_path

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
        elif self.cfg.model_name == "gemma1":
            self.model, self.tokenizer, self.config = load_gemma_1b()
        else:
            raise ValueError(f"Model {self.cfg.model_name} not supported")
        
    def load_dictionary_and_resize_tokenizer(self):
        self.dictionary = TokenManager(load_path=self.cfg.data_path)
        additional_tokens = []
        self.token_map = {}
        # check if token has mapping in model tokenizer
        for token, idx in self.dictionary.token_idx.items():
            model_token_to_idx = self.tokenizer.convert_tokens_to_ids(token)
            if model_token_to_idx is not None and token not in ["map1", "map2", "map3", "map4", "map5", "map6"] and model_token_to_idx != 3:
                self.token_map[idx] = model_token_to_idx
            else:
                additional_tokens.append(token)
        if additional_tokens:
            self.tokenizer.add_tokens(additional_tokens)
            self.model.resize_token_embeddings(len(self.tokenizer))
        for token, idx in self.dictionary.token_idx.items():
            if token not in self.token_map:
                self.token_map[idx] = self.tokenizer.convert_tokens_to_ids(token)
        print(f"token_map: {self.token_map}")
    
    def load_dataloaders(self):
        loaders = []
        for split in ["train", "train_heldout", "test"]:
            dataset = MappedSyntheticDataset(self.cfg.data_path, split=split, mode=self.cfg.prompt_mode, token_map=self.token_map)
            if split == "train":
                batch_size = self.cfg.batch_size
            else:
                batch_size = self.cfg.batch_size * 100
            loader = get_data_loader(dataset, batch_size, self.cfg.num_workers)
            loaders.append(loader)
        return loaders

    def get_output_dir(self):
        output_dir = get_directory_path(self.cfg, key='train', prefix_dir='models/ckpts')
        print(f"output_dir: {output_dir}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        return output_dir
        
    def training_loop(self):
        # first load the model and tokenizer
        self.load_model_and_tokenizer()
        # then load the dictionary and resize the tokenizer
        self.load_dictionary_and_resize_tokenizer()
        # then load the dataloaders after the tokenizer is resized
        loaders = self.load_dataloaders()
        train_loader = loaders[0]
        if isinstance(train_loader.dataset.data[0], torch.Tensor):
            sample = train_loader.dataset.data[0].cpu().numpy()
        else:
            sample = train_loader.dataset.data[0]
        self.seq_info = self.dictionary.get_seq_info(sample)
        # then configure the optimizers
        self.optimizer = configure_optimizers(self.model, self.cfg.optimizer)
        self.train(train_loader, loaders)

    def train(self, train_loader, loaders):
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
            for dat, targets, elems in train_loader:
                
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
                    eval_info = self.evaluate(loaders[1:], device_info)
                    log_eval(it, lr, eval_info, logger=self.logger)
                

        # log the final evaluation metrics
        eval_info = self.evaluate(loaders[1:], device_info)
        log_eval(it, lr, eval_info, logger=self.logger)
        save_model(self.cfg, self.model, self.optimizer, it, self.get_output_dir(), self.token_map)

    @torch.no_grad()
    def _predict(self, inputs, new_length):
        """Predict the next tokens."""
        self.model.eval()
        for _ in range(new_length):
            logits = self.model(inputs)
            logits = logits.logits
            next_token = torch.argmax(logits[:, -1, :], -1, keepdims=True)
            inputs = torch.cat((inputs, next_token), dim=1)
        return inputs

    @torch.no_grad()
    def evaluate(self, evalLoaders, device_info):
        all_loss, all_acc, all_sharp_acc = [], [], []
        device, dt = device_info
        self.model.eval()
        for idx, split in enumerate(("train_heldout", "test")):
            loader = evalLoaders[idx]
            sequences, total_loss, total_acc, sharp_acc = 0.0, 0.0, 0.0, 0.0
            for dat, targets, elems in loader:
                dat, targets = move_to_device(dat, targets, device)
                bs = dat.size(0)
                with torch.amp.autocast(device_type=device, dtype=dt):
                    # get the logits B*T*V
                    logits = self.model(dat)
                    logits = logits.logits
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                    total_loss += loss.item() * bs
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
            "loss": {
                "train_heldout": all_loss[0],
                "test": all_loss[1],
            },
            "acc": {
                "train_heldout": all_acc[0],
                "test": all_acc[1],
            },
            "sharp_acc": {
                "train_heldout": all_sharp_acc[0],
                "test": all_sharp_acc[1],
            },
        }
        
        self.model.train()
        return info