
from torch.utils.data import Dataset
import os
import pandas as pd
from PIL import Image
import numpy as np

class MNIST(Dataset):

    def __init__(self, config, mode=None):
        self.config = config
        self.mode = self.config['mode'] if mode is None else mode
        self.data_dir = self.config["dataset"][self.mode]["dir"]
        
        # In this example, metadata and data is combined
        self.metadata = self.get_metadata()

    def get_metadata(self):
        metadata_path = os.path.join(self.config["dataset"][self.mode]["metadata"])
        metadata = pd.read_csv(metadata_path)
        return metadata

    def __getitem__(self, idx):
        
        img, target = self.metadata.iloc[idx,1:].to_numpy(), int(self.metadata.iloc[idx,0])

        return img, target

    def __len__(self):
        return len(self.metadata)