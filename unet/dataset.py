import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform 
        self.images = os.listdir(image_dir)
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))

        #Converting the image and the mask to a numpy array
        
        #converting the imahe into torch supported channels
        image= np.array(Image.open(img_path).convert("RGB"))
        #converting colour into black and white colour channels using "L"
        mask= np.array(Image.open(mask_path).convert("L"), dtype= np.float32)

        #preprocessing the mask


        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augumentation = self.transform(image= image, mask= mask)
            mask = augumentation["mask"]
            image = augumentation["image"]


            return image, mask