# Configuration and hyperparameters

import os
from pathlib import Path
import torch
import numpy as np

class Config:
    def __init__(self,combo=None):
        self.csv_file = Path('/home/sayantan/Alzheimer/ADNIMERGE_18Sep2023_final3.csv')
        self.img_folder = Path('/home/sayantan/Alzheimer/images_Adni_final_v5')
        self.genetic_folder_path = Path("/home/sayantan/Alzheimer/ADNI_Genetic_Merge")
        self.atlas_folder_path = Path('/home/sayantan/Alzheimer/nicara_preprocessed_iuids')

        self.transform_shape = (64, 128, 128)
        
        # Training configuration
        self.training_config = {
            'epochs': 1,
            'lr': 0.001,
            'patience': 8
        }
        self.result_dir = Path('./combination_results')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.num_workers = 16
        self.seed = 42

        self.save_every_n_epochs = 1
        self.split_ratio = 0.8

        self.epoch_nodes = np.linspace(0, self.training_config['epochs'], 5).astype('int').tolist()

        self.__combo_fn(combo)

    
    def __combo_fn(self, combo):
        if combo is not None:
            self.combo = combo         

            self.checkpoints_dir = self.result_dir/self.combo/"checkpoints"
            self.log_dir = self.result_dir/self.combo/"logs"
            self.metrics_dir = self.result_dir/self.combo/"metrics"
            self.figures_dir = self.result_dir/self.combo/"figure"

        # figure_dir = config.figures_dirs


