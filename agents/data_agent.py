import os
import json
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from config import BATCH_SIZE, NUM_WORKERS, IMG_SIZE, USE_SUBSET, SUBSET_SIZE

REAL_BASE_PATH = r"C:\CarettaProje\data\caretta_data\turtles-data\data"

class SeaTurtleDataset(Dataset):
    """
    PyTorch Dataset for SeaTurtleID2022.
    Expects a dataframe with columns: 'image_path', 'bbox', 'turtle_id'
    """
    def __init__(self, df, id_to_label=None, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
        # Create a mapping from turtle_id to integer label
        self.unique_ids = sorted(self.df['turtle_id'].unique())
        if id_to_label is None:
            self.id_to_label = {t_id: i for i, t_id in enumerate(self.unique_ids)}
        else:
            self.id_to_label = id_to_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        bbox = row['bbox']
        turtle_id = row['turtle_id']
        
        # If the turtle ID isn't in our training set (shouldn't happen in closed-set, but just in case)
        label = self.id_to_label.get(turtle_id, -1)
        
        sample = {
            'image_path': img_path,
            'bbox': bbox,
            'label': label,
            'turtle_id': turtle_id
        }
        
        return sample

class DataAgent:
    """
    Agent responsible for loading metadata, annotations, and providing DataLoaders.
    Single Responsibility: Data provisioning.
    """
    def __init__(self):
        self.data_dir = REAL_BASE_PATH
        
    def load_metadata(self):
        """Loads and parses the dataset metadata and annotations."""
        print(f"[DataAgent] Loading data from {self.data_dir}...")
        
        splits_path = os.path.join(self.data_dir, "metadata_splits.csv")
        annotations_path = os.path.join(self.data_dir, "annotations.json")
        
        if not os.path.exists(splits_path) or not os.path.exists(annotations_path):
            raise FileNotFoundError(f"Missing required files in {self.data_dir}")
            
        # Read splits
        df = pd.read_csv(splits_path)
        
        # Read annotations to get bboxes
        with open(annotations_path, 'r') as f:
            anno_data = json.load(f)
            
        # Map image_id to bbox
        bbox_map = {}
        for ann in anno_data.get('annotations', []):
            img_id = ann['image_id']
            bbox = ann.get('bbox', None)
            if bbox:
                bbox_map[img_id] = bbox
                
        # Construct final dataframe
        records = []
        for _, row in df.iterrows():
            img_id = row['id']
            img_path = os.path.join(self.data_dir, row['file_name'].replace('/', os.sep))
            turtle_id = row['identity']
            split = row['split_closed']
            
            # Fix for 'valid' instead of 'val'
            if split == 'valid':
                split = 'val'
            
            bbox = bbox_map.get(img_id, None)
            
            records.append({
                'image_path': img_path,
                'turtle_id': turtle_id,
                'bbox': bbox,
                'split': split
            })
            
        return pd.DataFrame(records)

    def get_dataloaders(self):
        """
        Main method to get train, val, and test dataloaders.
        """
        df = self.load_metadata()
        
        if USE_SUBSET:
            print(f"[DataAgent] CPU detected. Using subset of {SUBSET_SIZE} turtles.")
            subset_ids = df['turtle_id'].unique()[:SUBSET_SIZE]
            df = df[df['turtle_id'].isin(subset_ids)]
            
        # Filter by split
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'val']
        test_df = df[df['split'] == 'test']

        print(f"[DataAgent] Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        train_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = SeaTurtleDataset(train_df, transform=train_transforms)
        val_dataset = SeaTurtleDataset(val_df, id_to_label=train_dataset.id_to_label, transform=val_transforms)
        test_dataset = SeaTurtleDataset(test_df, id_to_label=train_dataset.id_to_label, transform=val_transforms)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
            'transforms': {
                'train': train_transforms,
                'val': val_transforms
            },
            'num_classes': len(train_dataset.unique_ids)
        }
    def get_sample_image_for_id(self, turtle_id):
        """Returns the first image path found for a specific turtle ID."""
        try:
            df = self.load_metadata()
            samples = df[df['turtle_id'] == turtle_id]
            if not samples.empty:
                return samples.iloc[0]['image_path']
            return None
        except Exception as e:
            print(f"[DataAgent] Error fetching sample for {turtle_id}: {e}")
            return None
