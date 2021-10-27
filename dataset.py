import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import os
import cv2
class CarvanaDataset(Dataset):
    def __init__(self,transform=None):
        self.image_dir = os.path.join("data", "train")
        self.mask_dir = os.path.join("data", "train_masks")
        self.transform = transform 
        self.images = os.listdir(self.image_dir)
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
         
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        image= image.astype(float)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
        