from pathlib import Path
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
import torchvision.transforms as T

class CoinDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = Path(img_dir)
        self.transform = transform

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = str(self.df.loc[idx, "Id"])
        
        matching_files = list(self.img_dir.glob(f"{image_id}*"))

        if not matching_files:
            raise FileNotFoundError(f"Could not find any image matching ID: {image_id} in {self.img_dir}")

        img_path = matching_files[0]
        
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            else:
                image = self.to_tensor(image)
        except (UnidentifiedImageError, IOError):
            blank_image = Image.new('RGB', (224, 224), color=0)
            if self.transform:
                image = self.transform(blank_image)
            else:
                image = self.to_tensor(blank_image)

        label = int(self.df.loc[idx, "label"])

        return image, label