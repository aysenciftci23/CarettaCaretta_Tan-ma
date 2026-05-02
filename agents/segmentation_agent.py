import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from config import IMG_SIZE, BATCH_SIZE, NUM_WORKERS

class CroppedTurtleDataset(Dataset):
    """
    Dataset wrapper that takes original paths, loads images,
    crops the head region using bbox, and applies transformations.
    """
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)
        
    def _get_mock_image(self):
        """Helper to generate a mock image if file is not found."""
        img = np.zeros((500, 500, 3), dtype=np.uint8)
        cv2.circle(img, (250, 250), 100, (0, 255, 0), -1) # Mock head
        return img

    def crop_head(self, image, bbox):
        """
        Uses the bounding box to crop the head from the image.
        bbox should be [x, y, w, h].
        """
        h_img, w_img = image.shape[:2]
        
        if bbox is None or len(bbox) != 4:
            # Fallback if bbox is empty: return center crop
            return image[h_img//4:3*h_img//4, w_img//4:3*w_img//4]

        x, y, w, h = [int(v) for v in bbox]
        
        # Add some padding
        pad = 20
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(w_img - x, w + 2*pad)
        h = min(h_img - y, h + 2*pad)

        cropped_img = image[y:y+h, x:x+w]
        
        # In case crop is somehow empty
        if cropped_img.size == 0:
            return image[h_img//4:3*h_img//4, w_img//4:3*w_img//4]
            
        return cropped_img

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        img_path = sample['image_path']
        bbox = sample['bbox']
        
        # Load image
        if os.path.exists(img_path):
            try:
                pil_img = Image.open(img_path).convert('RGB')
                image = np.array(pil_img)
            except Exception as e:
                image = self._get_mock_image()
        else:
            image = self._get_mock_image()

        # Crop head
        cropped_head = self.crop_head(image, bbox)

        # Convert to PIL Image for torchvision transforms
        cropped_pil = Image.fromarray(cropped_head)
        
        # Resize to IMG_SIZE
        cropped_pil = cropped_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)

        if self.transform:
            tensor_img = self.transform(cropped_pil)
        else:
            # Fallback
            from torchvision.transforms import ToTensor
            tensor_img = ToTensor()(cropped_pil)

        return {
            'image': tensor_img,
            'label': sample['label'],
            'turtle_id': sample['turtle_id']
        }

class SegmentationAgent:
    """
    Agent responsible for extracting the head region from turtle images.
    Single Responsibility: Crop ROIs based on masks.
    """
    def __init__(self):
        pass

    def preprocess_image(self, pil_img: Image.Image) -> np.ndarray:
        """
        Prepares a single uploaded image for inference.
        Resizes to IMG_SIZE x IMG_SIZE and converts to numpy array.
        """
        resized = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        return np.array(resized)

    def process(self, data_dict):
        """
        Takes the data dictionary from DataAgent and wraps the datasets
        with the cropping logic. Returns a new dictionary with updated DataLoaders.
        """
        print("[SegmentationAgent] Wrapping DataLoaders with head cropping logic...")
        
        heads_data = {}
        
        # Extract original datasets and their transforms
        transforms = data_dict['transforms']
        
        train_ds = data_dict['train'].dataset
        val_ds = data_dict['val'].dataset
        test_ds = data_dict['test'].dataset
        
        # Create wrapped datasets
        cropped_train = CroppedTurtleDataset(train_ds, transform=transforms['train'])
        cropped_val = CroppedTurtleDataset(val_ds, transform=transforms['val'])
        cropped_test = CroppedTurtleDataset(test_ds, transform=transforms['val'])
        
        # Create new dataloaders
        heads_data['train'] = DataLoader(cropped_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
        heads_data['val'] = DataLoader(cropped_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
        heads_data['test'] = DataLoader(cropped_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
        heads_data['num_classes'] = data_dict['num_classes']
        
        print("[SegmentationAgent] Processing pipeline configured.")
        return heads_data
