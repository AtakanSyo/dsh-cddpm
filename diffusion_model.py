from typing import Optional
import logging
import argparse
from configs import TrainingConfig
from utils.trainer import Trainer
from utils.training_io import save_checkpoint
from dataclasses import asdict
import yaml

def main():

    cfg = TrainingConfig() # initiate config
    logging.basicConfig(level=logging.INFO) # initiate logging
    trainer = Trainer(cfg) # initiate training with config
    logging.info("Training configuration:\n%s", yaml.dump(asdict(cfg), 
    sort_keys=False, default_flow_style=False)) # log the config
    trainer.train() # start training
    save_checkpoint(
        trainer.cfg.output_dir,
        trainer.cfg.checkpoint_path,
        trainer.model,
        trainer.optimizer,
        trainer.logger,
        cfg.n_epochs,
    ) # save checkpoint at every n'th epoch
    logging.info("TRAINING FINISHED.")
if __name__ == '__main__':
    main() # call main