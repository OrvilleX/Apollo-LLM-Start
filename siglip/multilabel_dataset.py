from torch.utils.data import Dataset
import torch
from PIL import Image
import os
import numpy as np

class MultiLabelDataset(Dataset):
  def __init__(self, root, df, transform):
    self.root = root
    self.df = df
    self.transform = transform

  def __getitem__(self, idx):
    item = self.df.iloc[idx]
    # get image
    image_path = os.path.join(self.root, item["Image_Name"])

    if not os.path.exists(image_path):
        return None

    image = Image.open(image_path).convert("RGB")

    # prepare image for the model
    pixel_values = self.transform(image)

    # get labels
    labels = item[2:].values.astype(np.float32)

    # turn into PyTorch tensor
    labels = torch.from_numpy(labels)

    return pixel_values, labels

  def __len__(self):
    return len(self.df)