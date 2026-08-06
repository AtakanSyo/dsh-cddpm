# configs.py
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    # Data settings
    dataset_name: str = "reflex_data_8_corrected_camera_grid"
    image_size: int   = 48
    batch_size: int   = 256

    # validation params
    val_ratio: float         = 0.1 # validaton data ratio
    split_seed: int          = 42
    early_stop_patience: int = 20000 
    min_improve_delta: float = 1e-4
    anscombe_global_max: float = 78.1888738888085

    #lr decay parameters
    lr_decay_factor: float   = 0.97
    lr_patience: int         = 70
    
    # Training hyperparams
    n_epochs: int           = 12000
    lr: float               = 1e-4
    loss_type: str          = "SBP+NOISE+LABEL"
    val_loss_type: str      = "SBP"
    lambda_label: float     = 0.05 
    lambda_sbp: float       = 1e-4
    lambda_lit: float       = 1.0
    lambda_kl: float        = 1.0
    sbp_threshold: float    = 0.5

    # SNR 
    snr_gamma: float        = 5.0

    # Sampling
    sample_every: int       = 30
    num_samples: int        = 16
    example_cond1: float    = 6200.0
    example_cond2: float    = 22.0
    loss_sampling_every_x_epoch: int  = 2000
    ddim_steps_loss: int    = 2
    ddim_steps_val: int     = 2 

    # Paths
    data_root: str         = "../data"
    output_dir: str        = "./outputs"
    plots_dir: str         = "./plots"
    ddim_dir: str          = "./ddim_samples"
    loss_vis: str          = "./loss_vis"
    checkpoint_path: str   = "conditional_unet.pth"

