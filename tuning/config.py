# tuning/config.py
import os
import torch

class Config:
    DATA_PATH = "./data_prepared"
    AFTER_TUNING_PATH = "./results/after_tuning"
    TUNING_LOGS_PATH = "./results/tuning_logs"
    
    TUNING_EPOCHS = 10
    FINAL_EPOCHS = 30
    EARLY_STOPPING_PATIENCE = 7
    TUNING_EARLY_STOPPING_PATIENCE = 3
    IN_CHANNELS = 3
    OUT_CHANNELS = 1
    
    LR_VALUES = [1e-3, 1e-4, 1e-5]
    BATCH_SIZES = [8, 16]
    LOSS_FUNCTIONS = ['bce', 'bce_dice']
    OPTIMIZERS = ['adam', 'sgd']
    DROPOUT_VALUES = [0.2, 0.3, 0.5]
    
    DEFAULT_LR = 1e-3
    DEFAULT_BATCH_SIZE = 8
    DEFAULT_LOSS = 'bce_dice'
    DEFAULT_OPTIMIZER = 'adam'
    DEFAULT_DROPOUT = 0.3
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATALOADER_WORKERS = 2 if torch.cuda.is_available() else 0
    PIN_MEMORY = True if torch.cuda.is_available() else False
    
    @classmethod
    def setup_dirs(cls):
        os.makedirs(cls.AFTER_TUNING_PATH, exist_ok=True)
        os.makedirs(cls.TUNING_LOGS_PATH, exist_ok=True)
