import os

import torch


def save_loss_data_to_file(
    output_dir,
    logger,
    losses,
    noise_losses,
    label_losses,
    sbp_losses,
    kl_losses,
    val_sbp_history,
):
    file_path = os.path.join(output_dir, "loss_log.csv")
    try:
        with open(file_path, "w") as f:
            f.write("epoch,total_loss,noise_mse,label_mse,train_sbp,kl_loss,val_sbp\n")

            for i in range(len(sbp_losses)):
                l = losses[i]
                n = noise_losses[i]
                lab = label_losses[i] if label_losses is not None and i < len(label_losses) else ""
                sbp = sbp_losses[i]
                kl = kl_losses[i] if kl_losses is not None and i < len(kl_losses) else ""
                v = val_sbp_history[i] if i < len(val_sbp_history) else ""

                f.write(f"{i + 1},{l},{n},{lab},{sbp},{kl},{v}\n")

        logger.info(f"Loss data successfully saved to {file_path}")

    except IOError as e:
        logger.error(f"Failed to save loss data to {file_path}: {e}")


def save_checkpoint(
    output_dir,
    checkpoint_path,
    model,
    optimizer,
    logger,
    epoch,
):
    model_to_save = model.module if hasattr(model, "module") else model
    state = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    base_name, ext = os.path.splitext(checkpoint_path)
    ckpt_filename = f"{base_name}_epoch_{epoch:05d}{ext}"
    ckpt_path = os.path.join(output_dir, ckpt_filename)

    torch.save(state, ckpt_path)
    logger.info(f"Checkpoint saved to {ckpt_path}")


def prepare_directories(output_dir, ddim_dir, plots_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ddim_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
