# utils/trainer.py
import os
import logging
import math
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from tqdm.auto import tqdm
import json
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from datetime import timedelta
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR

from .sbp            import radial_profile_loss_per_sample, radial_angular_profile_loss_per_sample, auxiliary_kl_loss, radial_chi2_profile_per_sample
from .condition      import get_condition_stats_from_filenames
from .dataset        import CustomMatrixDataset, compute_global_stats
from .model          import ConditionalUnet
from .plotting       import save_loss_plots
from configs         import TrainingConfig
from .training_io import save_loss_data_to_file, save_checkpoint, prepare_directories


class Trainer:
    def __init__(self, config: TrainingConfig):
        self.cfg = config # load the config from the parameter, set to self.cfg
        self._setup_distributed() # for multi gpu training
        self.device = self._setup_device() # sets the used device to self.device.
        self.logger = logging.getLogger(__name__) # set the logger
        prepare_directories(
            self.cfg.output_dir,
            self.cfg.ddim_dir,
            self.cfg.plots_dir,
        )
        self._load_data()
        self._build_model()
        self._setup_optimizer()
        self.global_step = 0
        self.ddpm = DDPMScheduler(num_train_timesteps=1000, beta_schedule="linear")

        # Optional: these two make math faster on recent NVIDIA (still FP32, not mixed precision)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def _setup_distributed(self):
        # If launched with torchrun, these env vars will exist.
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank       = int(os.environ.get("RANK", "0"))

        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank

        if self.world_size > 1:
            if torch.cuda.is_available():
                torch.cuda.set_device(self.local_rank)
            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl" if torch.cuda.is_available() else "gloo",
                    timeout=timedelta(seconds=1800),
                )
            # refresh from the actual process group (safety)
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            # single-process
            if dist.is_available() and dist.is_initialized():
                # Unlikely, but be clean.
                dist.destroy_process_group()

    def _setup_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if torch.cuda.is_available():
            # Optimize cuDNN algorithms for fixed input shapes
            torch.backends.cudnn.benchmark = True

        return device

    def _load_data(self):
        path = os.path.join(self.cfg.data_root, self.cfg.dataset_name, "matrices")

        # --- Compute dataset statistics for z-score normalization ---

        self.mean_c1, self.std_c1, self.mean_c2, self.std_c2 = \
            get_condition_stats_from_filenames(path)  # for computing x and nH stats
        self.logger.info(
            f"Condition stats -> mean_c1={self.mean_c1:.4f}, std_c1={self.std_c1:.4f}, "
            f"mean_c2={self.mean_c2:.4f}, std_c2={self.std_c2:.4f}"
        )

        stats = {
            "mean_c1": self.mean_c1,
            "std_c1": self.std_c1,
            "mean_c2": self.mean_c2,
            "std_c2": self.std_c2,
        }
        stats_path = os.path.join(self.cfg.output_dir, "condition_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=4)
        self.logger.info(f"Condition stats saved to {stats_path}")

        mean_global, std_global, max_global = compute_global_stats(path) # for computing DSH stats
        # --- Save global matrix normalization stats ---
        file_path = os.path.join(self.cfg.output_dir, "global_data_stats.csv")

        # create header only if file does not exist
        write_header = not os.path.exists(file_path)

        with open(file_path, "a") as f:
            if write_header:
                f.write("mean_global,std_global,max_global\n")
            f.write(f"{mean_global},{std_global},{max_global}\n")

        self.logger.info(f"Saved global normalization stats -> {file_path}")

        ds = CustomMatrixDataset( # initialize the data, use z score normalization
            root_dir=path,
            data_max=max_global,
            mean_val1=self.mean_c1, std_val1=self.std_c1,
            mean_val2=self.mean_c2, std_val2=self.std_c2,
            resize_to=self.cfg.image_size
        )

        # --- Train/val split ---
        val_ratio = getattr(self.cfg, "val_ratio", 0.1) # set to val ratio, if not set, set to 0.1
        n_total   = len(ds) # n total is data size
        n_val     = max(1, int(n_total * val_ratio)) # n val is validation data size
        n_train   = max(1, n_total - n_val) # n train is training data size
        g = torch.Generator().manual_seed(getattr(self.cfg, "split_seed", 42)) # random number generator
        train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=g) # dataset is 
        # split into train ds and val ds.

        if self.world_size > 1: # if we have multiple gpu s
            train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=False) # distribute training data
            shuffle_flag = False # sampler already handles
        else:
            train_sampler = None # if there is only 1 gpu
            shuffle_flag = True # for single gpu dataloader must shuffle

        # --- DataLoaders ---

        self.dataloader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle_flag,          # << use flag
            sampler=train_sampler,         # << attach sampler
            num_workers=8,
            pin_memory=True,
            persistent_workers=True 
        )
        self.val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False, # val stays deterministic
            sampler=None, # we will use only 1 gpu for validaton. it is already a small thing to do.
            num_workers=4,
            pin_memory=True,
            persistent_workers=True 
        )

    def _build_model(self):
        m = ConditionalUnet(
            sample_size=self.cfg.image_size,
            in_channels=1,
            out_channels=1,
            num_conditions=2,
            condition_embed_dim=16,
            predict_labels=True
        ).to(self.device)

        """         try: FOR MEMORY OPTIMIZATION. I DONT THINK IT HELPS MUCH.
                    m.model.enable_gradient_checkpointing()
                    self.logger.info("Enabled gradient checkpointing on UNet2DModel.")
                except Exception as e:
                    self.logger.warning(f"Could not enable gradient checkpointing: {e}") """

        if self.world_size > 1:
            m = DDP(m, device_ids=[self.local_rank], output_device=self.local_rank, find_unused_parameters=False)
        self.model = m

        self.logger.info("Model initialized.")

    def _setup_optimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr) # setting to ADAM
        # Optimizer. using lr from config.

        # Cosine annealing LR scheduler (per-epoch)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.n_epochs,
            eta_min=self.cfg.lr * 0.01
        )

        self.logger.info(
            f"Optimizer: Adam | LR: {self.cfg.lr} | Scheduler: CosineAnnealingLR (T_max={self.cfg.n_epochs})"
        )

    def _is_rank0(self):
        return getattr(self, "rank", 0) == 0
    
    @torch.no_grad()
    def save_ddim_grid(self, epoch: int, num_images: int = 16, seed: int = 123):
        """
        Runs a 2-step DDIM x0 on a small batch and saves a grid for visual tracking.
        Uses a batch from val_loader if available; otherwise from the train loader.
        """
        if not self._is_rank0():
            return

        was_training = self.model.training
        self.model.eval()

        # 1) Grab a batch (val preferred)
        try:
            batch_x, batch_cond = next(iter(self.val_loader))
        except Exception:
            batch_x, batch_cond = next(iter(self.dataloader))

        # 2) Subset & move
        x     = batch_x[:num_images].to(self.device, non_blocking=True)
        conds = batch_cond[:num_images].to(self.device, non_blocking=True)

        # 3) Deterministic sampling for comparability
        g = torch.Generator(device=self.device)
        g.manual_seed(seed + epoch)  # slight epoch jitter to prevent trivial memorization

        # 4) Sample random timesteps & noise, make noisy_x
        t = torch.randint(
            0, self.ddpm.config.num_train_timesteps,
            (x.size(0),), device=self.device, generator=g
        ).long()
        noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=g)
        noisy_x = self.ddpm.add_noise(x, noise, t)

        # 5) DDIM 2-step x0
        pred_x0 = self._ddim_x0(noisy_x, t, conds, steps = self.cfg.ddim_steps_val )

        # 6) Save grid
        nrow = int(math.sqrt(max(1, pred_x0.shape[0])))
        save_path = os.path.join(self.cfg.ddim_dir, f"ddim_x0_epoch_{epoch:05d}.png")
        torchvision.utils.save_image(
            pred_x0.clamp(-1, 1),
            save_path,
            normalize=True,
            value_range=(-1, 1),
            nrow=nrow
        )
        self.logger.info(f"Saved DDIM x0 grid to {save_path}")

        # 7) Save per-image min–max grid (min->0, max->1 -> PNG writes as 0..255)
        #    Handles flat images safely (max==min) and any NaNs/Infs.
        px = torch.nan_to_num(pred_x0.detach())  # replace NaN/Inf if any
        mins = px.amin(dim=(1, 2, 3), keepdim=True)
        maxs = px.amax(dim=(1, 2, 3), keepdim=True)
        denom = (maxs - mins).clamp(min=1e-8)
        px01 = (px - mins) / denom              # per-image scale to [0,1]

        save_path_mm = os.path.join(self.cfg.ddim_dir, f"ddim_x0_epoch_{epoch:05d}_minmax.png")
        torchvision.utils.save_image(
            px01.clamp(0, 1),
            save_path_mm,
            normalize=False,   # already in [0,1]
            nrow=nrow
        )
        self.logger.info(f"Saved DDIM x0 grid (per-image min-max) to {save_path_mm}")
        
        if was_training:
            self.model.train()

    def _ddim_x0(self, noisy_x, timesteps, conds, steps: int = 2, pred_noise_t=None):
        """
        Differentiable DDIM-style partial reverse:
        - Start at x_t (noisy_x) and do `steps` model evaluations at a grid of times
        t, floor(t*(steps-1)/steps), ..., floor(t*1/steps), 0.
        - Return the final x0 estimate (clamped to [-1, 1]).
        - If `pred_noise_t` is provided, we reuse it for the first step at `t`.

        Args:
            noisy_x:    (B, 1, H, W) at time `t`
            timesteps:  (B,) int64
            conds:      (B, num_conditions)
            steps:      int >= 1 (steps=1 = single x0-from-t; steps=2 ≈ your old two-step)
            pred_noise_t: optional (B, 1, H, W) noise prediction at `t`
        """
        device = noisy_x.device
        alphas = self.ddpm.alphas_cumprod.to(device)
        t0 = timesteps.long()

        # Make sure steps is sensible
        steps = max(1, int(steps))

        cur_x = noisy_x
        cur_t = t0
        cur_pred_noise = pred_noise_t  # may be None

        for m in range(steps):
            # predict noise at current time
            if cur_pred_noise is None:
                out = self.model(cur_x, cur_t, conds)
                cur_pred_noise = out['noise'] if isinstance(out, dict) else out

            # compute x0 at current time
            alpha_cur = alphas[cur_t].view(-1, 1, 1, 1)
            pred_x0 = (cur_x - torch.sqrt(1.0 - alpha_cur) * cur_pred_noise) / (torch.sqrt(alpha_cur) + 1e-8)

            # last iteration: return x0
            if m == steps - 1:
                return torch.clamp(pred_x0, -1.0, 1.0)

            # move to the next (earlier) timestep on a uniform grid in index space
            remaining = steps - 1 - m  # how many steps left *after* we jump
            next_t = (t0 * remaining) // steps  # elementwise floor for each sample

            # DDIM deterministic update to x_{next_t} using current pred_noise and pred_x0
            alpha_next = alphas[next_t].view(-1, 1, 1, 1)
            next_x = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(1.0 - alpha_next) * cur_pred_noise

            # advance
            cur_x = next_x
            cur_t = next_t
            cur_pred_noise = None  # force a fresh prediction at the new time
    
    @torch.no_grad()
    def _validate_total_loss(self) -> float:
        """Compute validation TOTAL loss with the same recipe as training."""
        self.model.eval()

        sum_loss = 0.0
        sum_noise = 0.0
        sum_label = 0.0
        sum_sbp = 0.0
        epoch_sample_count = 0

        alphas = self.ddpm.alphas_cumprod.to(self.device)
        for x, conds in self.val_loader:
            x     = x.to(self.device, non_blocking=True)
            conds = conds.to(self.device, non_blocking=True)

            # sample t like training
            t = torch.randint(0, self.ddpm.config.num_train_timesteps, (x.size(0),),
                            device=self.device).long()
            noise   = torch.randn_like(x)
            noisy_x = self.ddpm.add_noise(x, noise, t)

            out = self.model(noisy_x, t, conds)
            pred_noise  = out['noise']
            pred_labels = out.get('labels', None)

            # losses (same weighting as train)
            loss_noise = F.mse_loss(pred_noise, noise)
            loss_label = F.mse_loss(pred_labels, conds) if pred_labels is not None else torch.tensor(0., device=self.device)

            # 2-step DDIM x0 for SBP
            pred_x0_ddim = self._ddim_x0(noisy_x, t, conds, steps = self.cfg.ddim_steps_val ,pred_noise_t=pred_noise)

            sbp_per_sample = radial_chi2_profile_per_sample(
                pred_x0_ddim, x,
                nbins=24,
                inputs_are_neg1_to_1=True,
                anscombe_glob_max=self.cfg.anscombe_global_max,
            )
            # Min-SNR-γ weighting for SBP (cap high-SNR timesteps)
            alphas  = self.ddpm.alphas_cumprod.to(x.device, dtype=x.dtype)
            alpha_t = alphas[t]                                          # [B]
            snr     = alpha_t / (1.0 - alpha_t).clamp_min(1e-8)          # [B] >= 0

            gamma = torch.as_tensor(self.cfg.snr_gamma, device=x.device, dtype=x.dtype)
            w = torch.minimum(snr, gamma)                                # cap high SNR
                         # keep loss scale stable
            w = torch.nan_to_num(w, nan=0.0, posinf=1.0, neginf=1.0)     # safety
            w = w.detach()                                               # no grad through weights

            loss_sbp = (sbp_per_sample * w).mean()

            # apply lambdas and compose total
            lt = self.cfg.val_loss_type.upper()
            ln = loss_noise * self.cfg.lambda_lit
            ll = loss_label * self.cfg.lambda_label
            ls = loss_sbp

            if   lt == "NOISE":         total = ln
            elif lt == "LABEL":              total = ll
            elif lt == "NOISE+LABEL":   total = ln + ll
            elif lt == "SBP+NOISE+LABEL":      total = ln + ll + ls
            elif lt == "SBP+NOISE":          total = ln + ls
            elif lt == "SBP":          total = ls
            else: raise ValueError(f"Invalid loss_type: {lt}")

            b = x.size(0)
            sum_loss  += float(total.item()) * b
            sum_noise += float(ln.item())    * b
            sum_label += float(ll.item())    * b
            sum_sbp   += float(ls.item())    * b
            epoch_sample_count += b

        # average per sample
        denom = max(1, epoch_sample_count)
        return float(sum_loss / denom)

    def _accumulate_batch_losses(
        self,
        batch_losses,
        batch_size,
        sum_loss,
        sum_noise,
        sum_label,
        sum_sbp,
        sum_kl,
        epoch_sample_count,
    ):
        loss, noise_loss, label_loss, sbp_loss, kl_loss = batch_losses

        sum_loss += loss * batch_size
        sum_noise += noise_loss * batch_size
        sum_label += label_loss * batch_size
        sum_sbp += sbp_loss * batch_size
        sum_kl += kl_loss * batch_size
        epoch_sample_count += batch_size

        return sum_loss, sum_noise, sum_label, sum_sbp, sum_kl, epoch_sample_count

    def _compute_epoch_average_losses(self, totals):
        total_samples_denom = totals[5].clamp_min(1.0)

        avg_epoch_loss = (totals[0] / total_samples_denom).item()
        avg_epoch_noise_loss = (totals[1] / total_samples_denom).item()
        avg_epoch_label_loss = (totals[2] / total_samples_denom).item()
        avg_epoch_sbp_loss = (totals[3] / total_samples_denom).item()
        avg_epoch_kl_loss = (totals[4] / total_samples_denom).item()

        return (
            avg_epoch_loss,
            avg_epoch_noise_loss,
            avg_epoch_label_loss,
            avg_epoch_sbp_loss,
            avg_epoch_kl_loss,
        )

    def train(self):

        epoch_losses = []
        epoch_noise_losses = []
        epoch_label_losses = []
        epoch_sbp_losses = []
        epoch_kl_losses = []
        val_sbp_history = []
        
        for epoch in range(1, self.cfg.n_epochs + 1):
            self.model.train() # set the model to training , if it is in inference mode

            sum_loss = sum_kl = sum_noise = sum_label = sum_sbp = 0.0 # setting the losses for each epoch to 0
            epoch_sample_count = 0

            if self.world_size > 1 and hasattr(self.dataloader, "sampler") and self.dataloader.sampler is not None:
                self.dataloader.sampler.set_epoch(epoch)

            for batch_x, batch_cond in tqdm(self.dataloader, desc=f"Epoch {epoch}", disable=not self._is_rank0()):
                x = batch_x.to(self.device, non_blocking=True)
                conds = batch_cond.to(self.device, non_blocking=True)

                batch_losses = self._train_step(x, conds, epoch)
                batch_size = x.size(0)
                sum_loss, sum_noise, sum_label, sum_sbp, sum_kl, epoch_sample_count = self._accumulate_batch_losses(
                    batch_losses,
                    batch_size,
                    sum_loss,
                    sum_noise,
                    sum_label,
                    sum_sbp,
                    sum_kl,
                    epoch_sample_count,
                )

            if epoch % self.cfg.sample_every == 0:
                self.save_ddim_grid(epoch, num_images=self.cfg.num_samples)

            # --- All batch totals into one tensor.  ---
            totals = torch.tensor([sum_loss, sum_noise, sum_label, sum_sbp, sum_kl ,float(epoch_sample_count)],
                                device=self.device, dtype=torch.float32)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(totals, op=dist.ReduceOp.SUM) # Sum totals from all distributed processes (GPUs)

            avg_epoch_loss, avg_epoch_noise_loss, avg_epoch_label_loss, avg_epoch_sbp_loss, avg_epoch_kl_loss = \
                self._compute_epoch_average_losses(totals)

            current_lr = self.optimizer.param_groups[0]["lr"]

            if self._is_rank0(): # logging the losses
                self.logger.info(
                    f"Epoch {epoch} | Avg Loss: {avg_epoch_loss:.4f} | "
                    f"Avg Noise (MSE): {avg_epoch_noise_loss:.4f} | "
                    f"Avg Label (MSE): {avg_epoch_label_loss:.4f} | "
                    f"Avg SBP: {avg_epoch_sbp_loss:.4f} | "
                    f"Avg KL: {avg_epoch_kl_loss:.4f} |"
                    f"LR: {current_lr:.10f}"
                )


            # Appending the losses to lists.
            epoch_losses.append(avg_epoch_loss)
            epoch_noise_losses.append(avg_epoch_noise_loss)
            epoch_label_losses.append(avg_epoch_label_loss)
            epoch_sbp_losses.append(avg_epoch_sbp_loss)
            epoch_kl_losses.append(avg_epoch_kl_loss)

            if self._is_rank0():
                val_sbp = self._validate_total_loss()  # I will come back to this function TODO
            else:
                val_sbp = 0.0


            val_sbp_tensor = torch.tensor([val_sbp], device=self.device)
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(val_sbp_tensor, src=0)
            val_sbp = val_sbp_tensor.item()
            if self._is_rank0():
                val_sbp_history.append(val_sbp)
                
            # Step cosine LR scheduler each epoch
            self.lr_scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self._is_rank0():
                self.logger.info(f"Cosine LR step -> LR: {current_lr:.8f}")

            # ---- periodic plot/save ----
            if (self._is_rank0()) and (epoch % 200 == 0) and (epoch != 0):
                save_loss_data_to_file(
                    self.cfg.output_dir,
                    self.logger,
                    epoch_losses,
                    epoch_noise_losses,
                    epoch_label_losses,
                    epoch_sbp_losses,
                    epoch_kl_losses,
                    val_sbp_history,
                )
                save_loss_plots(
                    epoch_losses,
                    epoch_noise_losses,
                    epoch_label_losses,
                    epoch_sbp_losses,
                    epoch_kl_losses,
                    val_sbp_history,
                    self.cfg.plots_dir
                )
            
            if (self._is_rank0()) and (epoch % 1000 == 0) and (epoch != 0):
                save_checkpoint(
                    self.cfg.output_dir,
                    self.cfg.checkpoint_path,
                    self.model,
                    self.optimizer,
                    self.logger,
                    epoch,
                )

        # ---- final save/plots ----
        if self._is_rank0():
            save_loss_data_to_file(
                self.cfg.output_dir,
                self.logger,
                epoch_losses,
                epoch_noise_losses,
                epoch_label_losses,
                epoch_sbp_losses,
                epoch_kl_losses,
                val_sbp_history,
            )
            save_loss_plots(
                epoch_losses,
                epoch_noise_losses,
                epoch_label_losses,
                epoch_sbp_losses,
                epoch_kl_losses,
                val_sbp_history,
                self.cfg.plots_dir
            )

    def _train_step(self, x, conds, epoch):
        # ---- pick random timesteps ----
        T = self.ddpm.config.num_train_timesteps

        timesteps = torch.randint(0, T, (x.size(0),), device=self.device).long()

        # Noise and noisy input
        noise   = torch.randn_like(x)
        noisy_x = self.ddpm.add_noise(x, noise, timesteps)

        # ---- Single forward: predict noise and labels at t ----
        out_t = self.model(noisy_x, timesteps, conds)
        pred_noise = out_t['noise']              # (B, 1, H, W)
        pred_labels = out_t.get('labels', None)  # (B, num_conditions)

        # Losses
        loss_noise = F.mse_loss(pred_noise, noise)
        loss_label = F.mse_loss(pred_labels, conds) if pred_labels is not None else \
                     torch.tensor(0.0, device=self.device)


        # ---- Compute single-pass x0_hat at current t (NO [0,1] mapping here) ----
        alphas_cum = self.ddpm.alphas_cumprod.to(x.device, dtype=x.dtype)     # [T]
        alpha_t    = alphas_cum[timesteps].view(-1, 1, 1, 1)                  # [B,1,1,1]
        x0_hat     = (noisy_x - torch.sqrt(1.0 - alpha_t) * pred_noise) / (torch.sqrt(alpha_t) + 1e-8)

        # ---- Min-SNR-γ weighting (detached) ----
        alpha_vec = alpha_t.view(-1)                              # [B]
        snr = alpha_vec / (1.0 - alpha_vec).clamp_min(1e-8)       # [B]
        gamma = torch.as_tensor(self.cfg.snr_gamma, device=x.device, dtype=x.dtype)
        w = torch.minimum(snr, gamma)

        w = torch.nan_to_num(w, nan=0.0, posinf=1.0, neginf=1.0).detach()  


        # ---- Physics gate by timestep (keep only low-noise steps) ----
        t_frac = (timesteps.float() / (T - 1)).to(x.dtype)     # [B]
        physics_gate = (t_frac < 0.6).float().detach()         # [B]

        
        # ---- Physics losses on raw [-1,1] tensors (your fns do positivity/eps) ----
        sbp_per_sample = radial_chi2_profile_per_sample(
            x0_hat, x,
            nbins=24,
            inputs_are_neg1_to_1=True,
            anscombe_glob_max=self.cfg.anscombe_global_max,
        )
        kl_per_sample  = auxiliary_kl_loss(
            x0_hat, x, nrad=24, nang=6
        )                                    # no grad through weights

        # Apply SNR weight and gate; do NOT multiply lambdas here
        loss_sbp = (sbp_per_sample * w).mean()
        loss_kl  = (kl_per_sample  * w * physics_gate).mean()

        # Compose final loss (SBP removed)
        lt = self.cfg.loss_type.upper()
        loss_noise = loss_noise * self.cfg.lambda_lit
        loss_label = loss_label * self.cfg.lambda_label

        # -------------------------------
        # Apply lambdas
        # -------------------------------
        loss_sbp = loss_sbp * self.cfg.lambda_sbp
        loss_kl  = loss_kl  * self.cfg.lambda_kl

        if lt == "NOISE":
            loss = loss_noise
        elif lt == "LABEL":
            loss = loss_label
        elif lt == "NOISE+LABEL":
            loss = loss_noise + loss_label
        elif lt == "SBP+NOISE+LABEL":
            loss = loss_noise + loss_label + loss_sbp
        elif lt == "SBP+NOISE+LABEL+KL":
            loss = loss_noise + loss_label + loss_sbp + loss_kl
        elif lt == "SBP+NOISE":
            loss = loss_noise + loss_sbp
        elif lt == "KL":
            loss = loss_kl
        else:
            raise ValueError(f"loss_type must be one of ['NOISE', 'LABEL', 'NOISE+LABEL'], got {self.cfg.loss_type}")


        # ---- Backprop ----
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.global_step += 1

        # Scalars for logging
        return (
            float(loss.detach()),
            float(loss_noise.detach()),
            float(loss_label.detach()),
            float(loss_sbp.detach()),
            float(loss_kl.detach())
        )



