MiB = 1024 ** 2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from abc import ABC, abstractmethod
from typing import Any
from tqdm.auto import tqdm
import math
import time
from dimol.diffusion.paths import ConditionalProbabilityPath
from collections import defaultdict
import os



def reduce_loss_dict(loss_dict: dict, ddp: bool) -> dict:
    """
    Average a dict of scalar tensors across all DDP ranks.
    Stacks values into a single tensor for one all-reduce instead of N.
    """
    if not ddp or not loss_dict:
        return {k: v.detach().clone() for k, v in loss_dict.items()}

    keys = list(loss_dict.keys())
    stacked = torch.stack([loss_dict[k].detach() for k in keys])
    dist.all_reduce(stacked, op=dist.ReduceOp.AVG)
    return {k: stacked[i] for i, k in enumerate(keys)}

def model_size_b(model: nn.Module) -> int:
    """
    Returns model size in bytes. Based on https://discuss.pytorch.org/t/finding-model-size/130275/2
    Args:
    - model: self-explanatory
    Returns:
    - size: model size in bytes
    """
    size = 0
    for param in model.parameters():
        size += param.nelement() * param.element_size()
    for buf in model.buffers():
        size += buf.nelement() * buf.element_size()
    return size


class Trainer(ABC):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

    @abstractmethod
    def get_loss(self, **kwargs) -> torch.Tensor:
        pass

    @abstractmethod
    def get_lr(self, **kwargs) -> float:
        pass

    @abstractmethod
    @torch.no_grad()
    def evaluate(self, val_dataloader, step, training_config) -> dict:
        """Run validation and return averaged loss components.
        Returns a dict of scalar tensors (or empty dict on rank != master)."""
        pass

    def train(self, dataloader, training_config, val_dataloader=None):
    
        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=training_config.lr)#self.model.get_optimizer()
        configure_optimizers = self.model.module.configure_optimizers if hasattr(self.model, "module") else self.model.configure_optimizers
        optimizer = configure_optimizers(training_config.optimiser.weight_decay, training_config.lr.learning_rate, training_config.hardware.device_type, master_process=training_config.ddp.master_process)

        use_fp16_scaler = (
            training_config.hardware.use_mp
            and torch.cuda.is_available()
            and training_config.hardware.mixed_dtype == torch.float16
        )
        if use_fp16_scaler:
            from torch.amp import GradScaler
            scaler = GradScaler(device=training_config.hardware.device)

        grad_accum_steps = training_config.data.grad_accum_steps or 1

        optimizer.zero_grad(set_to_none=True)
        micro_step = 0
        # accum_loss = 0.0
        accum_losses: dict = {}   # accumulates DETACHED scalars across grad_accum_steps
        t0 = time.time()
        self.model.train()
        total_step = 0

        for epoch in range(training_config.schedule.num_epochs):
            if training_config.data.sampler is not None:
                training_config.data.sampler.set_epoch(epoch)

            ran_validation = False

            for step, batch in enumerate(dataloader):
                
                if val_dataloader is not None and total_step>0 and total_step % training_config.schedule.validation_step == 0 and not ran_validation:
                    ran_validation = True
                    if training_config.data.val_sampler is not None:
                        training_config.data.val_sampler.set_epoch(epoch)
                    val_metrics = self.evaluate(val_dataloader, total_step, training_config)

                    if training_config.ddp.master_process and val_metrics:
                        parts = [f"VAL step {total_step:5d}"]
                        if "loss" in val_metrics:
                            parts.append(f"loss: {val_metrics['loss'].item():.6f}")
                        for k, v in val_metrics.items():
                            if k == "loss":
                                continue
                            parts.append(f"{k}: {v.item():.6f}")
                        print(" | ".join(parts))

                        if training_config.exp is not None:
                            prefix = 'val'
                            payload = {f"{prefix}_{key}": value.item() for key, value in val_metrics.items()}
                            training_config.exp.log_metrics(payload, step=total_step)

                    if training_config.ddp.ddp:
                        dist.barrier()

                for key in batch.keys():
                    batch[key] = batch[key].to(training_config.hardware.device, non_blocking=True)

                if training_config.ddp.ddp:
                    self.model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

                losses = self.get_loss(batch, training_config)
                loss = losses['loss']
                loss = loss / grad_accum_steps              # scale down for averaging
                # accum_loss += loss.detach()
                # accumulate every component (detached so we don't hold autograd graphs)
                for k, v in losses.items():
                    if not torch.is_tensor(v):
                        v = torch.as_tensor(float(v), device=loss.device)
                    accum_losses[k] = accum_losses.get(k, 0.0) + v.detach() / grad_accum_steps


                if use_fp16_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                micro_step += 1
                if micro_step % grad_accum_steps != 0:
                    continue                                # accumulate more
                
                training_config.schedule.decoder_pretrain_steps -= 1
                ran_validation = False
                # ---- optimizer step boundary ----
                norm = None
                if training_config.optimiser.max_grad_norm is not None:
                    if use_fp16_scaler:
                        scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), training_config.optimiser.max_grad_norm
                    )

                lr = self.get_lr(total_step, training_config)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                if use_fp16_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                micro_step = 0

                if training_config.hardware.device_type == "cuda":
                    torch.cuda.synchronize() # wait for the GPU to finish work
                
                # if training_config.ddp:
                #     dist.all_reduce(accum_loss, op=dist.ReduceOp.AVG)
                reduced = reduce_loss_dict(accum_losses, ddp=training_config.ddp.ddp)

                t1 = time.time()
                dt = t1 - t0 # time difference in seconds
                tokens_processed = training_config.data.batch_size * training_config.data.seq_len  * grad_accum_steps * training_config.ddp.ddp_world_size
                tokens_per_sec = tokens_processed / dt
                norm_str = f"{norm:.4f}" if norm is not None else "n/a"
                # if training_config.master_process:
                #     print(f"step {step:5d} | loss: {accum_loss.item():.6f} | lr: {lr:.6f} | norm: {norm_str} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
                
                if training_config.ddp.master_process:
                    norm_str = f"{norm:.4f}" if norm is not None else "n/a"
                    # Build "k: value" pairs in insertion order from `reduced`, headlining `loss`.
                    parts = [f"step {total_step:5d}"]
                    if "loss" in reduced:
                        parts.append(f"loss: {reduced['loss'].item():.6f}")
                    for k, v in reduced.items():
                        if k == "loss":
                            continue
                        parts.append(f"{k}: {v.item():.6f}")
                    parts += [
                        f"lr: {lr:.6f}",
                        f"norm: {norm_str}",
                        f"dt: {dt*1000:.2f}ms",
                        f"tok/sec: {tokens_per_sec:.2f}",
                    ]
                    print(" | ".join(parts))
                    if training_config.exp is not None:
                        prefix = 'train'
                        payload = {f"{prefix}_{key}": value.item() for key, value in reduced.items()}
                        payload.update({'lr': lr, 'norm': norm_str, 'dt': dt*1000, 'tok/sec': tokens_per_sec})
                        training_config.exp.log_metrics(payload, step=total_step)



                # accum_loss = 0.0
                accum_losses.clear()
                t0 = time.time()
                total_step += 1

            if epoch > 0  and training_config.ddp.master_process and (epoch % training_config.schedule.epoch_save_checkpoint == 0 or epoch+1==training_config.schedule.num_epochs):
                path = os.path.join('./checkpoints', training_config.run_name, str(epoch))
                os.makedirs(path, exist_ok=True)
                
                if hasattr(self.model, "module"):
                    self.model.module.save_model(path)
                else:
                    self.model.save_model(path)

class ConditionalGaussianDenoiserTrainerLite(Trainer):
    def __init__(self, path: ConditionalProbabilityPath, model: nn.Module, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path

    def get_loss(self, batch: dict, training_config) -> torch.Tensor:
        token_ids = batch['token_ids'] # B L
        attn_mask = batch.get('attention_mask').bool() # B L 

        # loss_mask = attn_mask.float()     

        embed = self.model.module.token_embedding if hasattr(self.model, "module") else self.model.token_embedding
        get_logits = self.model.module.out_proj if hasattr(self.model, "module") else self.model.out_proj


        token_embeddings = embed(token_ids)

        sampling_noise_std = training_config.sampling.sampling_noise_std or 0.25
        x0 = token_embeddings + torch.randn_like(token_embeddings)*sampling_noise_std
        
        batch_size, seq_len, emb_dim = x0.shape

        eps = 5e-5
        t = torch.rand(batch_size, 1, 1, device=token_ids.device) * (1 - 2*eps) + eps

        noise = torch.randn_like(x0)
        # noise_mask = attn_mask.unsqueeze(-1).float()
        # noise = noise * noise_mask

        alpha, beta = self.path.alpha(t), self.path.beta(t)
        x = alpha * x0 + beta * noise
            
        # import code; code.interact(local=dict(globals(), **locals()))
        time = t
        if time.dim() == 3:
            time = time.squeeze(-1)   # (B, 1, 1) -> (B,1)

        if training_config.regime == 'epsilon':
            target = noise
        elif training_config.regime == 'x':
            target = x0
        else:
            raise ValueError(f'Incorrect training regime: {training_config.regime}. Expected to be "epsilon" or "x"')

        attn_mask = None
        if training_config.hardware.use_mp:
            with torch.autocast(device_type=training_config.hardware.device_type, dtype=training_config.hardware.mixed_dtype):
                eps_theta = self.model(input_embeddings=x, time=time, attention_mask=attn_mask)
        else:
            eps_theta = self.model(input_embeddings=x, time=time, attention_mask=attn_mask)

        #--------------------------------
        #eps_theta = eps_theta * noise_mask
        #--------------------------------

        # # ----- Build padding mask -----
        # loss_mask = attn_mask.float()                              # (B, L)
        # # n_valid = loss_mask.sum().clamp(min=1)                     # scalar
        # n_valid = loss_mask.sum(-1).clamp(min=1)                    # (B)

        # # ----- Epsilon MSE (full noise range, masked over padding) -----
        # mse_per_pos = ((eps_theta - noise) ** 2).mean(dim=-1)          # (B, L), sum over C
        # # mse_loss = (mse_per_pos * loss_mask).sum() / n_valid
        # mse_per_sample = (mse_per_pos*loss_mask).sum(-1) / n_valid #(B,)
        # mse_loss = mse_per_sample.mean()
        mse_loss = ((eps_theta - target) ** 2).mean()


        # ----- Denoise to predicted x0 -----
        # alpha = self.path.alpha(t)                                 # (B, 1, 1)
        alpha = alpha.clamp(min=training_config.optimiser.eps)      # (B, 1, 1)

        if training_config.regime == 'epsilon':
            x0_hat = (x - beta * eps_theta) / alpha
        elif training_config.regime == 'x':
            x0_hat = eps_theta # (B, L, C)


        # ----- Reconstruction MSE (only at low-noise / high-alpha steps) -----
        # combine the per-sample alpha gate with the per-token padding mask
        mse_t0_sample_mask = (alpha > 0.80).squeeze(-1).squeeze(-1)     # (B,) bool ~ t > 0.33
        # mse_t0_loss_mask = loss_mask * mse_t0_sample_mask[:, None].float()  # (B, L)
        mse_t0_loss_mask = mse_t0_sample_mask[:, None].float()  # (B, L)

        if mse_t0_loss_mask.sum() > 0:
            sq = ((x0_hat - token_embeddings) ** 2).mean(-1)           # (B, L) avg over C
            mse_loss_t0 = (sq * mse_t0_loss_mask).sum() / mse_t0_loss_mask.sum().clamp(min=1)
        else:
            mse_loss_t0 = torch.tensor(0.0, device=x.device)

        # ----- Cross-entropy (gate by alpha) -----
        logits = get_logits(x0) # (B, L, V)
        # logits = get_logits(x0) # (B, L, V)

        # ce_sample_mask = (alpha > training_config.alpha_threshold).squeeze(-1).float() #(B,L)
        ce_sample_mask = (alpha > training_config.optimiser.alpha_threshold).squeeze(-1).squeeze(-1) #(B)

        if ce_sample_mask.any():
            ce_loss = F.cross_entropy(
                logits[ce_sample_mask].reshape(-1, logits.size(-1)),
                token_ids[ce_sample_mask].reshape(-1),
                ignore_index=training_config.data.pad_idx,
                label_smoothing=training_config.optimiser.label_smoothing,
                reduction='mean',
                weight=training_config.class_weight.class_weights
            )
        else:
            ce_loss = logits.sum() * 0.0

        alpha_flat = alpha.squeeze(-1).squeeze(-1)  # (B,)
        probs_mask = alpha_flat>0.5
        if probs_mask.any():
            probs = logits[probs_mask].softmax(-1) # (B, L, V)
            # paren balance: cumulative net parens must stay >= 0 and end at 0
            delta   = (probs * training_config.paren_delta).sum(-1)                     # (B, L)
            balance = delta.cumsum(dim=1)
            paren_loss = F.relu(-balance).mean() + balance[:, -1].abs().mean()

            # ring parity: each digit's expected count should be even
            ring_exp = torch.einsum('blv,vd->bd', probs, training_config.ring_count)    # (B, 10)
            ring_loss = (ring_exp - ring_exp.round()).pow(2).mean()     # ~0 when even

            grammar_loss = paren_loss + ring_loss
        else:
            grammar_loss = logits.sum() * 0.0
        
        
        metrics = {}
        

        # ----- Accuracy -----
        with torch.no_grad():
            pred_ids = logits.argmax(-1)  # (B, L)
            correct = (pred_ids == token_ids) & (token_ids != training_config.data.pad_idx) # (B, L)
            valid = (token_ids != training_config.data.pad_idx)
            metrics['token_acc'] = correct.sum() / valid.sum().clamp(min=1)

            for lo, hi, name in [(0.0, 0.3, 'high_noise'),
                                (0.3, 0.7, 'mid_noise'),
                                (0.7, 1.0, 'low_noise')]:
                bucket = (alpha_flat >= lo) & (alpha_flat < hi)
                if bucket.any():
                    # sel_ce = ce_per_pos[bucket]
                    # sel_mask = loss_mask[bucket]
                    # metrics[f'ce_{name}'] = (sel_ce * sel_mask).sum() / sel_mask.sum().clamp(min=1)
                    # metrics[f'mse_{name}'] = mse_per_sample[bucket].mean()

                    sel_correct = correct[bucket]
                    sel_valid = valid[bucket]
                    metrics[f'token_acc_{name}'] = sel_correct.sum() / sel_valid.sum().clamp(min=1)
                else:
                    metrics[f'ce_{name}'] = torch.tensor(0.0, device=x.device)
                    metrics[f'mse_{name}'] = torch.tensor(0.0, device=x.device)
                    metrics[f'token_acc_{name}'] = torch.tensor(0.0, device=x.device)

            emb = self.model.token_embedding.weight if not hasattr(self.model, 'module') else self.model.module.token_embedding.weight
            emb_norms = emb.norm(dim=-1)
            metrics['emb_norm_mean'] = emb_norms.mean()
            metrics['emb_norm_std'] = emb_norms.std()
            metrics['eps_theta_norm'] = eps_theta.detach().norm(dim=-1).mean()

            # Treat tokens with norm < 1e-3 as dead (never updated / WD-collapsed)
            # and exclude them from the ratio; report the count separately.
            alive = emb_norms > 1e-3
            metrics['emb_n_dead'] = (~alive).sum().float()
            if alive.any():
                alive_norms = emb_norms[alive]
                metrics['emb_norm_ratio'] = alive_norms.max() / alive_norms.min().clamp(min=1e-6)
            else:
                metrics['emb_norm_ratio'] = torch.tensor(0.0, device=emb.device)
                    
            # metrics['eps_theta_pad_norm'] = (eps_theta.detach().norm(dim=-1) * (1 - loss_mask)).sum() / (1 - loss_mask).sum().clamp(min=1)
            # metrics['eps_theta_real_norm'] = (eps_theta.detach().norm(dim=-1) * loss_mask).sum() / loss_mask.sum().clamp(min=1)

        # ----- Total -----
        lambda_ce = training_config.optimiser.lambda_ce or 1.0
        lambda_mse = training_config.optimiser.lambda_mse or 1.0
        lambda_grammar = training_config.optimiser.lambda_grammar or 0.0

        if training_config.schedule.decoder_pretrain_steps>0:
            lambda_mse = 0.0

        loss = lambda_mse*mse_loss + lambda_ce * ce_loss + lambda_mse*mse_loss_t0 + lambda_grammar * grammar_loss    

        metrics.update({'loss':loss, 'mse_loss': mse_loss.detach(), 'ce_loss': ce_loss.detach(), 'mse_loss_t0': mse_loss_t0.detach(), 'grammar_loss': grammar_loss.detach()})
        return metrics #{'loss':loss, 'mse_loss': mse_loss, 'ce_loss': ce_loss, 'mse_loss_t0': mse_loss_t0}

    @torch.no_grad()
    def evaluate(self, val_dataloader, step, training_config) -> dict:
        """Run validation and return averaged loss components.
        Returns a dict of scalar tensors (or empty dict on rank != master)."""
        pass
        self.model.eval()

        sums: dict = {}                 # running sums of detached scalars (this rank)
        n_batches = 0

        for val_batch in val_dataloader:
            for k in val_batch:
                val_batch[k] = val_batch[k].to(training_config.hardware.device, non_blocking=True)

            losses = self.get_loss(val_batch, training_config)
            for k, v in losses.items():
                if not torch.is_tensor(v):
                    v = torch.as_tensor(float(v), device=training_config.hardware.device)
                sums[k] = sums.get(k, torch.zeros((), device=training_config.hardware.device)) + v.detach()
            n_batches += 1

        # local mean over batches on this rank
        local_mean = {k: v / max(n_batches, 1) for k, v in sums.items()}

        if training_config.sampling.sample_examples and step % training_config.sampling.sample_step == 0:
            from dimol.diffusion.diff_eqs import LearnedScoreSDE
            from dimol.models.models import  DenoiserModel
            from dimol.tokenizer.smiles_tokenizer import SmilesTokenizer
            from dimol.diffusion.simulators import EulerMaruyamaSimulator

            tokenizer = SmilesTokenizer.load(training_config.path_to_tokenizer)
            score_model = DenoiserModel(self.model, self.path, regime=training_config.regime)
            sde = LearnedScoreSDE(self.path, score_model, training_config.sampling.sampling_variance)
            simulator = EulerMaruyamaSimulator(sde)

            x0 = self.path.p_simple.sample(training_config.sampling.num_samples, seed=training_config.ddp.ddp_rank)

            # eps = 1e-3
            # ts = torch.linspace(eps, 1 - eps, training_config.num_samling_timesteps).view(1, training_config.num_samling_timesteps, 1, 1).expand(training_config.num_samples, -1, -1, -1).to(training_config.device) # (num_samples, nts, 1)
            ts = torch.linspace(1e-4, 0.999, training_config.sampling.num_sampling_timesteps).view(1, training_config.sampling.num_sampling_timesteps, 1, 1).expand(training_config.sampling.num_samples, -1, -1, -1).to(training_config.hardware.device) # (num_samples, nts, 1)
            xts = simulator.simulate(x0, ts) 

            get_logits = self.model.module.out_proj if hasattr(self.model, "module") else self.model.out_proj
            probs = get_logits(xts).softmax(-1).detach().cpu()
            ids = probs.argmax(-1).tolist()

            smiles_list = tokenizer.decode_batch(ids, special_decode=True)

            try:
                from rdkit import Chem
                from rdkit import RDLogger
                RDLogger.DisableLog("rdApp.*")
            except ImportError:
                return None

            n_decoded_valid = 0
            for s in smiles_list:
                if Chem.MolFromSmiles(s) is not None:
                    n_decoded_valid += 1
            
            validity = torch.tensor(
                n_decoded_valid / len(smiles_list),
                device=training_config.hardware.device  # ← pin to correct device
            )
            local_mean.update({'validity': validity})
            print(smiles_list[:10])

        # one all-reduce across ranks at the end
        reduced = reduce_loss_dict(local_mean, ddp=training_config.ddp.ddp)
        self.model.train()
        return reduced

    def get_lr(self, it, training_config):
        # 1) linear warmup for warmup_iters steps
        if it < training_config.lr.warmup_steps:
            return training_config.lr.max_lr * (it+1) / training_config.lr.warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > training_config.schedule.max_steps:
            return training_config.lr.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - training_config.lr.warmup_steps) / (training_config.schedule.max_steps - training_config.lr.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return training_config.lr.min_lr + coeff * (training_config.lr.max_lr - training_config.lr.min_lr)




class ARTrainer:
    def __init__(self, model: nn.Module):
        self.model = model

    # ---------- core ----------
    def get_loss(self, batch: dict, cfg) -> dict:
        token_ids = batch["token_ids"]               # (B, L)
        inp = token_ids[:, :-1]                       # (B, L-1)
        tgt = token_ids[:, 1:]                        # (B, L-1)

        if cfg.use_mp:
            with torch.autocast(device_type=cfg.device_type, dtype=cfg.mixed_dtype):
                logits = self.model(inp)  
        else:
            logits = self.model(inp)  


        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1),
            ignore_index=cfg.pad_idx,
            label_smoothing=cfg.label_smoothing,
        )
        with torch.no_grad():
            pred  = logits.argmax(-1)
            valid = tgt != cfg.pad_idx
            acc   = ((pred == tgt) & valid).sum().float() / valid.sum().clamp(min=1)
            ppl   = loss.detach().exp()
        return {"loss": loss, "token_acc": acc.detach(), "ppl": ppl}

    def get_lr(self, step: int, cfg) -> float:
        if step < cfg.warmup_steps:
            return cfg.max_lr * (step + 1) / cfg.warmup_steps
        if step > cfg.max_steps:
            return cfg.min_lr
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_lr + coeff * (cfg.max_lr - cfg.min_lr)

    # ---------- val ----------
    @torch.no_grad()
    def evaluate(self, val_loader, cfg) -> dict:
        self.model.eval()
        sums, n = {}, 0
        for batch in val_loader:
            for k in batch: 
                batch[k] = batch[k].to(cfg.device, non_blocking=True)
            losses = self.get_loss(batch, cfg)
            for k, v in losses.items():
                sums[k] = sums.get(k, torch.zeros((), device=cfg.device)) + v.detach()
            n += 1
        means   = {k: v / max(n, 1) for k, v in sums.items()}
        reduced = reduce_loss_dict(means, ddp=cfg.ddp)
        self.model.train()
        return reduced

    # ---------- train ----------
    def train(self, train_loader, cfg, val_loader=None):
        unwrap = lambda m: m.module if hasattr(m, "module") else m
        optimizer = unwrap(self.model).configure_optimizers(
            cfg.weight_decay, cfg.max_lr, cfg.device_type, cfg.master_process
        )

        use_fp16_scaler = (
            cfg.use_mp
            and torch.cuda.is_available()
            and cfg.mixed_dtype == torch.float16
        )

        if use_fp16_scaler:
            from torch.amp import GradScaler
            scaler = GradScaler(device=cfg.device)


        accum, micro, total_step = {}, 0, 0
        t0 = time.time()
        self.model.train()
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(cfg.num_epochs):
            if cfg.sampler is not None: 
                cfg.sampler.set_epoch(epoch)
            ran_val = False

            for step, batch in enumerate(train_loader):

                # ---- validation
                if val_loader is not None and total_step > 0 and total_step % cfg.validation_step == 0 and not ran_val:
                    ran_val = True
                    if cfg.val_sampler is not None: 
                        cfg.val_sampler.set_epoch(epoch)
                    val = self.evaluate(val_loader, cfg)

                    if cfg.master_process:
                        msg = " | ".join(f"{k}: {v.item():.4f}" for k, v in val.items())
                        print(f"VAL step {total_step:5d} | {msg}")

                    if cfg.exp is not None:
                        prefix = 'val'
                        payload = {f"{prefix}_{key}": value.item() for key, value in val.items()}
                        cfg.exp.log_metrics(payload, step=total_step)

                # ---- batch -> device
                for k in batch: 
                    batch[k] = batch[k].to(cfg.device, non_blocking=True)

                # ---- DDP sync flag (fixed, mirrors the diffusion-trainer fix)
                if cfg.ddp:
                    is_sync = ((micro + 1) % cfg.grad_accum_steps) == 0
                    self.model.require_backward_grad_sync = is_sync

                losses = self.get_loss(batch, cfg)
                loss = losses["loss"] / cfg.grad_accum_steps
                for k, v in losses.items():
                    if not torch.is_tensor(v):
                        v = torch.as_tensor(float(v), device=loss.device)
                    accum[k] = accum.get(k, 0.0) + v.detach() / cfg.grad_accum_steps

                if use_fp16_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                micro += 1
                if micro % cfg.grad_accum_steps != 0:
                    continue

                # ---- optimizer step
                norm = None
                if cfg.max_grad_norm is not None:
                    if use_fp16_scaler:
                        scaler.unscale_(optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), cfg.max_grad_norm
                    )    
                
                lr = self.get_lr(total_step, cfg)

                for g in optimizer.param_groups: 
                    g["lr"] = lr

                if use_fp16_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                if cfg.device_type == "cuda": 
                    torch.cuda.synchronize()

                # ---- log
                reduced = reduce_loss_dict(accum, ddp=cfg.ddp)
                dt = time.time() - t0
                tokens_processed = cfg.batch_size * cfg.seq_len  * cfg.grad_accum_steps * cfg.ddp_world_size
                tokens_per_sec = tokens_processed / dt

                if cfg.master_process:
                    norm_str = f"{norm:.4f}" if norm is not None else "n/a"
                    parts = [f"step {total_step:5d}"] \
                          + [f"{k}: {v.item():.4f}" for k, v in reduced.items()] \
                          + [f"lr: {lr:.6f}",
                             f"norm: {norm:.4f}" if norm is not None else "norm: n/a",
                             f"dt: {dt*1000:.1f}ms",
                             f"tok/sec: {tokens_per_sec:.2f}"]
                    print(" | ".join(parts))
                    if cfg.exp is not None:
                        prefix = 'train'
                        payload = {f"{prefix}_{key}": value.item() for key, value in reduced.items()}
                        payload.update({'lr': lr, 'norm': norm_str, 'dt': dt*1000, 'tok/sec': tokens_per_sec})
                        cfg.exp.log_metrics(payload, step=total_step)

                accum.clear()
                t0 = time.time()
                total_step += 1
                ran_val = False
                micro = 0

            # ---- save (master only)
            if cfg.master_process and (epoch > 0 and epoch % cfg.epoch_save_checkpoint == 0 or epoch + 1 == cfg.num_epochs):
                path = os.path.join("./checkpoints_ar", cfg.run_name, str(epoch))
                os.makedirs(path, exist_ok=True)
                unwrap(self.model).save_model(path)
                print(f"[AR] saved to {path}")













class ConditionalScoreMatchingTrainer(Trainer):
    def __init__(self, path: ConditionalProbabilityPath, model: nn.Module, logger = None, **kwargs):
        super().__init__(model, logger, **kwargs)
        self.path = path


    def get_train_loss(self, z: torch.Tensor, pad_mask: torch.Tensor, alpha=0.05) -> torch.Tensor:
        batch_size = z.shape[0]
        eps = 1e-3
        t = torch.rand(batch_size, 1, 1, device=z.device) * (1 - 2*eps) + eps
        # t = torch.rand(batch_size, 1, 1, device=z.device)

        # 🔑 convert tokens → embeddings
        embed = self.model.module.embed_tokens if hasattr(self.model, "module") else self.model.embed_tokens
        z = embed(z) 
        # z: (B, L, D)
        
        x = self.path.sample_conditional_path(z, t)

        attn_mask = pad_mask[:, None, None, :]   # (B,1,1,L)
        loss_mask = ~pad_mask                    # (B,L)
        
        s_theta = self.model(x, t, attention_mask=attn_mask)
        s_ref = self.path.conditional_score(x, z, t)

        beta_t = torch.clamp(self.path.beta(t), min=1e-4).squeeze(-1) # (B,1)
        beta_t2 = beta_t * beta_t
    
        per_token_loss = (s_theta - s_ref).pow(2).sum(dim=-1)  # (B,L)
        per_token_loss = per_token_loss * loss_mask
        loss = (beta_t2 * per_token_loss).sum() / loss_mask.sum().clamp(min=1)

        # book-consistent DSM loss
        # loss = torch.mean(
        #     beta_t**2 * torch.sum((s_theta - s_ref)**2, dim=-1)
        # )
        return loss




class ConditionalGaussianDenoiserTrainer(Trainer):
    def __init__(self, path: ConditionalProbabilityPath, model: nn.Module, pad_idx = None, lambda_ce = 1.0, label_smoothing=0, **kwargs):
        super().__init__(model, **kwargs)
        self.path = path
        self.pad_idx = pad_idx
        self.label_smoothing = label_smoothing


    # def get_train_loss(self, z: torch.Tensor, pad_mask: torch.Tensor, **kwargs) -> torch.Tensor:
    def get_train_loss(self, batch: dict, **kwargs) -> torch.Tensor:
        #bs seq_len

        z = batch['smiles_ids']
        pad_mask = batch['smiles_mask']
        text_cond = batch.get('text_emb')
        text_mask = batch.get('text_mask')
        epoch = batch['epoch']



        embed = self.model.module.embed_tokens if hasattr(self.model, "module") else self.model.embed_tokens
        z_embed = embed(z)
        #x0 = embed(z)
        
        #bs seq_len, emd_dim
        # with torch.no_grad():
        #     var = z_embed.var(dim=(0,1), keepdim=True) + 1e-6
        #     std = var.sqrt().view(1, 1, -1)
        #     scale = math.sqrt(var.numel() / var.sum().item())

        pad_mask_f = pad_mask.unsqueeze(-1).float()
        # noise0 = torch.randn_like(z_embed) * std
        # noise0 = noise0 * pad_mask_f

        # x0 = z_embed + 0.5 * noise0 * scale
        x0 = z_embed + torch.randn_like(z_embed)*0.50#*1.0#*0.50#*0.10
        
        batch_size, seq_len, emb_dim = x0.shape

        eps = 1e-4
        t = torch.rand(batch_size, 1, 1, device=z.device) * (1 - 2*eps) + eps

        noise = torch.randn_like(x0)

        #--------------------------------

        # zero noise on padding tokens
        noise = noise * pad_mask_f

        # freeze padding embeddings
        x = self.path.alpha(t) * x0 + self.path.beta(t) * noise
        #--------------------------------

        attn_mask = pad_mask[:, None, None, :]   # (B,1,1,L)
        if text_mask is not None:
            text_mask = text_mask[:, None, None, :]
            
        #attn_mask = pad_mask
        loss_mask = pad_mask.float()                    # (B,L)


        model_kwargs = {
            "text_embed": x,
            "time": t,
            "attention_mask": attn_mask,
        }

        if text_cond is not None:
            text_cond = text_cond.type_as(x)

            # CFG dropout
            p = 0.10
            drop_mask = torch.rand(batch_size, device=text_cond.device) < p  # (B,)

            if drop_mask.any():
                text_cond = text_cond.clone()
                text_mask = text_mask.clone()

                # replace with learned null token
                text_cond[drop_mask] = self.model.module.null_text if hasattr(self.model, "module") else self.model.null_text
                text_mask[drop_mask] = False
                text_mask[drop_mask, 0] = True

            if epoch>50:
                model_kwargs["text_cond"] = text_cond
                model_kwargs["key_padding_mask"] = text_mask
            else:
                text_cond = text_cond.clone()
                text_mask = text_mask.clone()

                # replace with learned null token
                drop_mask = torch.rand(batch_size, device=text_cond.device) < 1.1 # (B,)
                text_cond[drop_mask] = self.model.module.null_text if hasattr(self.model, "module") else self.model.null_text
                text_mask[drop_mask] = False
                text_mask[drop_mask, 0] = True

                model_kwargs["text_cond"] = text_cond
                model_kwargs["key_padding_mask"] = text_mask   
            

        #eps_theta = self.model(x, t, attention_mask=attn_mask)
        eps_theta = self.model(**model_kwargs)

        #--------------------------------
        eps_theta = eps_theta * pad_mask_f
        #--------------------------------

        eps_ref = noise

        mse_loss = ((eps_theta - eps_ref)**2).sum(dim=-1)#.sum(dim=-1)
        mse_loss = (mse_loss * loss_mask).sum() / loss_mask.sum()


        # denoise
        x0_hat = (x - self.path.beta(t) * eps_theta) / self.path.alpha(t)

        alpha = self.path.alpha(t)
        mse_mask = (alpha > 0.75).squeeze(-1).squeeze(-1)  # ~ t > 0.33
        if mse_mask.any():
            mse_loss_t0 = ((x0_hat[mse_mask] - z_embed[mse_mask])**2).sum(dim=-1)#.sum(dim=-1)
            mse_loss_t0 = (mse_loss_t0 * loss_mask[mse_mask]).sum() / loss_mask[mse_mask].sum()
        else:
            mse_loss_t0 = 0.0

        get_logits = self.model.module.get_logits if hasattr(self.model, "module") else self.model.get_logits

        # CE
        logits = get_logits(x0_hat)

        vocab = logits.shape[-1]
        
        ce_mask = (alpha > 0.55).squeeze(-1).squeeze(-1)
        if ce_mask.any():
            ce_loss = nn.functional.cross_entropy(
                logits[ce_mask].view(-1, vocab),
                z[ce_mask].view(-1),
                ignore_index=self.pad_idx,
                label_smoothing=self.label_smoothing
            )
        else:
            ce_loss = 0.0

        lambda_ce = kwargs.pop('lambda_ce')

        loss = mse_loss + lambda_ce * ce_loss + mse_loss_t0

        if ce_mask.any():
            ce_loss = ce_loss.mean().detach().cpu().item()
        
        if mse_mask.any():
            mse_loss_t0 = mse_loss_t0.mean().detach().cpu().item()

        return loss, mse_loss, ce_loss, mse_loss_t0