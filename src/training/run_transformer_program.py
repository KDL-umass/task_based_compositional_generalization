import torch
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
from pathlib import Path
from synthetic.init import set_seed
from net.tf_programs import return_transformer_program_from_cfg
from net.previous_tp_code.programs import softmax, gumbel_hard, gumbel_soft, argmax
from net.utils import code_utils, metric_utils
from net.runner import save_model_transformer_program
import os


def decode(token, token_indices):
        # txt_list = [self.token[i] for i in token_indices]
        txt_list = []
        for i in token_indices:
            if i in token:
                txt_list.append(token[i])
            # add a space for readability
            txt_list.append(' ')
        # remove the last space
        if txt_list[-1] == ' ':
            txt_list = txt_list[:-1]
    
        return ''.join(txt_list)


def run_transformer_programs(cfg, logger):
    X_train, Y_train, X_test, Y_test, X_val, Y_val, w_idx, t_idx, idx_w, idx_t = get_dataset(cfg)
    cfg.net.context_size = X_train.shape[1]
    cfg.net.vocab_size = len(idx_w)
    
    # Load network
    device_str = cfg.system.device
    if device_str == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU for nanoGPT")
        device_str = 'cpu'
    elif device_str == 'mps' and not torch.backends.mps.is_available():
        logger.warning("MPS not available, falling back to CPU for nanoGPT")
        device_str = 'cpu'
    
    device = torch.device(device_str)
    run_program(cfg,
    idx_w=idx_w,
    w_idx=w_idx,
    idx_t=idx_t,
    t_idx=t_idx,
    X_train=X_train,
    Y_train=Y_train,
    X_test=X_test,
    Y_test=Y_test,
    X_val=X_val,
    Y_val=Y_val,
    device=device)

def get_train_data(train_data, token_idx, autoregressive=False):
    if not autoregressive:
       
        sep_idx = np.where(train_data == token_idx['<SEP>'])[1][-1]
        X_train = train_data[:, :sep_idx]
        # replace only last <SEP> with <END>
        # add <END> to the end of X_train
        X_train = np.concatenate([X_train, np.full((X_train.shape[0], 1), token_idx['<END>'])], axis=1)
        Y_train = train_data[:, sep_idx+1:]
        
        Y_train = np.where(Y_train == token_idx['<END>'], token_idx['<NULL>'], Y_train)
        # fill beginning of Y_train with <NULL> so that it is the same length as X_train
        Y_train = np.concatenate([np.full((Y_train.shape[0], X_train.shape[1] - Y_train.shape[1]), token_idx['<NULL>']), Y_train], axis=1)
    else:
        X_train = train_data[:, :-1]
        Y_train = train_data[:, 1:]
    return X_train, Y_train
    

def get_dataset(cfg):
    fpath = cfg.data.path
    mode = cfg.tag
    train_dataset = os.path.join(fpath, 'train_{}_corpus.npy'.format(mode))
    test_dataset = os.path.join(fpath, 'test_{}_corpus.npy'.format(mode))
    token = np.load(os.path.join(fpath, 'token.pkl'),
                    allow_pickle=True)
    token_idx = np.load(os.path.join(fpath, 'token_idx.pkl'),
                        allow_pickle=True)
    train_data = np.load(train_dataset)
    test_data = np.load(test_dataset)
    print(train_data.shape)
    print(test_data.shape)
    train_data_sample = decode(token, train_data[0])
    test_data_sample = decode(token, test_data[0])
    print(train_data_sample)
    print(test_data_sample)
    X_train, Y_train = get_train_data(train_data, token_idx, autoregressive=cfg.net.autoregressive)
    val_idx = np.random.choice(len(X_train), size=1000, replace=False)
    X_val = X_train[val_idx]
    Y_val = Y_train[val_idx]
    X_train = np.delete(X_train, val_idx, axis=0)
    Y_train = np.delete(Y_train, val_idx, axis=0)
    X_test, Y_test = get_train_data(test_data, token_idx, autoregressive=cfg.net.autoregressive)

    x_train_sample = decode(token, X_train[0])
    y_train_sample = decode(token, Y_train[0])
    print(x_train_sample)
    print(y_train_sample)
    
    #idx_w is mapping from token to index
    idx_w = np.array(list(token_idx.keys()))
    w_idx = token_idx
    #w_idx is mapping from index to index
    idx_t = idx_w
    t_idx = w_idx
    return X_train, Y_train, X_test, Y_test, X_val, Y_val, w_idx, t_idx, idx_w, idx_t

def run_program(
    cfg,
    idx_w=None,
    w_idx=None,
    idx_t=None,
    t_idx=None,
    X_train=None,
    Y_train=None,
    X_test=None,
    Y_test=None,
    X_val=None,
    Y_val=None,
    device=None,
):
    d = max(len(idx_w), X_train.shape[-1])
    cfg.net.init_emb = None
    cfg.net.unembed_mask = None
    if cfg.net.unembed_mask:
        cfg.net.unembed_mask = np.array([t in ("<unk>", "<pad>") for t in idx_t])

    set_seed(cfg.seed)
    cfg.net.vocab_size = len(idx_w)
    cfg.net.vocab_size_out = len(idx_t)
    cfg.net.seq_len = X_train.shape[-1]
    model = return_transformer_program_from_cfg(cfg)
    model.to(device)

    opt = Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.optim.learning_rate)
    n_epochs = cfg.train.n_epochs
    if cfg.train.tau_schedule not in ("linspace", "geomspace"):
        raise NotImplementedError(cfg.train.tau_schedule)
    tau_schedule = (
        np.linspace if cfg.train.tau_schedule == "linspace" else np.geomspace
    )
    set_seed(cfg.seed)
    
    out = run_training(
        model,
        opt,
        X_train,
        Y_train,
        eval_splits=[("val", X_val, Y_val), ("test", X_test, Y_test)],
        batch_size=cfg.train.batch_size,
        n_epochs=n_epochs,
        n_samples=cfg.train.gumbel_samples,
        autoregressive=cfg.net.autoregressive,
        temps=tau_schedule(cfg.train.tau_init, cfg.train.tau_end, n_epochs),
        x_pad_idx=-1,
        y_pad_idx=-1,
        loss_agg=cfg.train.loss_agg,
        max_grad_norm=cfg.optim.grad_clip,
        o_idx=t_idx.get("O", None),
        idx_t=idx_t,
    )
    
    out["sample_fn"] = cfg.net.sample_fn
    model.set_temp(cfg.train.tau_end, argmax)
    dfs = [out]
    
    for split, X, Y in [
        ("train", X_train, Y_train),
        ("val", X_val, Y_val),
        ("test", X_test, Y_test),
    ]:
        loss, acc, metrics, preds, true = run_test(
            model,
            X,
            Y,
            return_preds=True,
            x_pad_idx=-1,
            y_pad_idx=-1,
            autoregressive=cfg.net.autoregressive,
            loss_agg=cfg.train.loss_agg,
            o_idx=t_idx.get("O", None),
            idx_t=idx_t,
        )
        print(f"{split}: loss={loss}, acc={acc}, metrics={metrics}")
        df = pd.DataFrame(
            {
                "epoch": [n_epochs],
                "split": split,
                "loss": loss,
                "acc": acc,
                "sample_fn": "argmax",
            }
        )
        for k, v in metrics.items():
            df[k] = v
        dfs.append(df)
    df = pd.concat(dfs).reset_index(drop=True)

    # create output directory if it doesn't exist
    fdir = "{}/{}/{}/".format(cfg.system.output_dir, cfg.tag, cfg.data.train_split)
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    
    if cfg.train.save_final_model:
        save_model_transformer_program(cfg, model, opt, fdir)

    if cfg.train.save_code:
        print(f"saving code to {fdir}")
        if not os.path.exists(fdir):
            os.makedirs(fdir)
        # print(idx_w)
        # print(X_val[0])
        x = [idx_w[i] for i in X_val[0]]
        x = [i for i in x if i != " "]
        print(fdir)
        code_utils.model_to_code(
            model=model,
            idx_w=idx_w,
            idx_t=idx_t,
            embed_csv=not cfg.net.one_hot_embed,
            unembed_csv=True,
            one_hot=cfg.net.one_hot_embed,
            autoregressive=cfg.net.autoregressive,
            var_types=True,
            output_dir=fdir,
            name=cfg.data.dataset_name_for_code_utils,
            example=x,
        )
        

    return df

def run_test(
    model,
    X,
    Y,
    batch_size=256,
    return_preds=False,
    x_pad_idx=None,
    y_pad_idx=None,
    autoregressive=False,
    func=torch.argmax,
    loss_agg="per_token",
    o_idx=None,
    idx_t=None,
):
    
    dataloader = DataLoader(
        list(zip(X, Y)), batch_size=batch_size, shuffle=False
    )
    out = []
    preds = []
    true = []
    model.eval()
    for x, y in dataloader:
        x = x.to(model.device)
        m = (x != x_pad_idx).float()
        mask = (m.unsqueeze(-1) @ m.unsqueeze(-2)).bool()
        if autoregressive:
            mask = torch.tril(mask)
        with torch.no_grad():
            log_probs = model(x, mask=mask).log_softmax(-1)
        tgts = y.to(model.device)
        if loss_agg == "per_seq":
            losses = -log_probs.gather(2, tgts.unsqueeze(-1))
            losses = losses.masked_fill(
                (tgts == y_pad_idx).unsqueeze(-1), 0.0
            ).sum(-1)
        else:
            all_losses = -log_probs.gather(2, tgts.unsqueeze(-1)).squeeze(-1)
            masked_losses = all_losses.masked_fill((tgts == y_pad_idx), 0.0)
            lengths = (tgts != y_pad_idx).sum(-1)
            losses = masked_losses.sum(-1) / lengths
        out.append(losses.detach().cpu().numpy())
        pred = func(log_probs, -1)
        preds.append(pred.detach().cpu().numpy())
        true.append(tgts.detach().cpu().numpy())
    preds = np.concatenate(preds, 0)
    true = np.concatenate(true, 0)
    m = true != y_pad_idx
    # have sharp accuracy where preds and true are exactly the same 
    
    acc = (preds == true)[m].mean()
    metrics = {}
    if o_idx is not None:
        y_true = [idx_t[y[y != y_pad_idx]].tolist() for y in true]
        y_pred = [
            idx_t[y_hat[y != y_pad_idx]].tolist()
            for y, y_hat in zip(true, preds)
        ]
        metrics = metric_utils.conll_score(y_true=y_true, y_pred=y_pred)
    loss = np.concatenate(out, 0).mean()
    if return_preds:
        return loss, acc, metrics, preds, true
    return loss, acc, metrics


def run_training(
    model,
    opt,
    X_train,
    Y_train,
    X_test=None,
    Y_test=None,
    eval_splits=None,
    batch_size=256,
    n_epochs=5,
    temps=None,
    n_samples=1,
    x_pad_idx=0,
    y_pad_idx=0,
    autoregressive=False,
    loss_agg="per_token",
    max_grad_norm=None,
    reg_alpha=None,
    patience=None,
    o_idx=None,
    idx_t=None,
    smooth_temps=True,
):
    
    train_dataloader = DataLoader(
        list(zip(X_train, Y_train)), batch_size=batch_size, shuffle=True
    )
    out = []
    metrics = []
    t = tqdm(range(n_epochs), total=n_epochs)
    if temps is None:
        temps = [None] * n_epochs
    if eval_splits is None and X_test is not None:
        eval_splits = [("val", X_test, Y_test)]
    for epoch in t:
        temp = temps[epoch]
        model.train()
        if temp is not None and smooth_temps and epoch + 1 < len(temps):
            ttemps = np.geomspace(temp, temps[epoch + 1], len(train_dataloader))
        elif temp is not None:
            ttemps = [temp] * len(train_dataloader)
        else:
            ttemps = [None] * len(train_dataloader)
        epoch_losses = []
        for ttemp, (x, y) in zip(ttemps, train_dataloader):
            if ttemp is not None:
                model.set_temp(ttemp)
            x = x.to(model.device)
            m = (x != x_pad_idx).float()
            mask = (m.unsqueeze(-1) @ m.unsqueeze(-2)).bool()
            if autoregressive:
                mask = torch.tril(mask)
            lst = []
            losses_lst = []
            tgts = y.to(model.device)
            for _ in range(n_samples):
                logits = model(x, mask=mask)
                if loss_agg == "per_seq":
                    log_probs = logits.log_softmax(-1)
                    losses = -log_probs.gather(2, tgts.unsqueeze(-1))
                    losses = losses.masked_fill(
                        (tgts == y_pad_idx).unsqueeze(-1), 0.0
                    ).sum(-1)
                else:
                    log_probs = logits.log_softmax(-1)
                    all_losses = -log_probs.gather(
                        2, tgts.unsqueeze(-1)
                    ).squeeze(-1)
                    masked_losses = all_losses.masked_fill(
                        (tgts == y_pad_idx), 0.0
                    )
                    lengths = (tgts != y_pad_idx).sum(-1)
                    losses = masked_losses.sum(-1) / lengths
                loss = losses.mean()
                if reg_alpha:
                    loss += reg_alpha * model.embed.reg()
                lst.append(loss)
                losses_lst.append(losses.detach().cpu())
            loss = torch.stack(lst, 0).mean(0)
            loss.backward()
            if torch.isnan(loss):
                m = torch.isnan(losses_lst[-1])
                # print(losses_lst[-1])
                # print(m.nonzero())
                # print(log_probs[m])
                # print(x[m], tgts[m])
                raise ValueError("loss is nan")
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_grad_norm,
                )
            epoch_losses.append(torch.stack(losses_lst, 0).mean(0).numpy())
            opt.step()
            opt.zero_grad()
            model.zero_grad()
        epoch_loss = np.concatenate(epoch_losses, 0).mean()
        d = {"loss": epoch_loss.mean()}
        t.set_postfix(d)
        out.append(epoch_loss.mean())
        model.eval()
        with torch.no_grad():
            d = {
                "epoch": epoch,
                "epoch_loss": epoch_loss.mean(),
                "split": "train",
            }
            d["loss"], d["acc"], m = run_test(
                model,
                X_train,
                Y_train,
                x_pad_idx=x_pad_idx,
                y_pad_idx=y_pad_idx,
                autoregressive=autoregressive,
                loss_agg=loss_agg,
                o_idx=o_idx,
                idx_t=idx_t,
            )
            d.update(m)
            metrics.append(d)
            for split, X, Y in eval_splits:
                d = {
                    "epoch": epoch,
                    "epoch_loss": epoch_loss.mean(),
                    "split": split,
                }
                d["loss"], d["acc"], m = run_test(
                    model,
                    X,
                    Y,
                    batch_size=batch_size,
                    x_pad_idx=x_pad_idx,
                    y_pad_idx=y_pad_idx,
                    autoregressive=autoregressive,
                    loss_agg=loss_agg,
                    o_idx=o_idx,
                    idx_t=idx_t,
                )
                d.update(m)
                metrics.append(d)
        if patience is not None and epoch - np.argmin(out) > patience:
            print(f"no improvement for {patience} epochs, stopping")
            break

    return pd.DataFrame(metrics)



