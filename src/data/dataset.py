import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2

class FERDataset(Dataset):

    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        emotion = row["emotion"]

        pixels = np.array(row["pixels"].split(), dtype=np.float32)

        image = pixels.reshape(48,48)

        image = cv2.resize(image,(224,224))

        image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

        image = image/255.0

        image = torch.tensor(image).permute(2,0,1).float()

        label = torch.tensor(emotion)

        return image,label
