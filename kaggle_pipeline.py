# %% [markdown]
# # Step 1: Install Dependencies
# %%
!pip install timm

# %% [markdown]
# # Step 2: Imports and Configuration
# %%
import os
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import timm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DATA_DIR = "/kaggle/input/seaturtleid2022" 
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 2 # Kaggle allows multiprocessing, set to 2 or 4.

print(f"Using device: {DEVICE}")

# %% [markdown]
# # Step 3: Agent Definitions (SOLID Principles)
# %%

class SeaTurtleDataset(Dataset):
    """
    Base PyTorch Dataset. Loads original images and bounding boxes.
    """
    def __init__(self, df, id_to_label=None, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.unique_ids = sorted(self.df['turtle_id'].unique())
        if id_to_label is None:
            self.id_to_label = {t_id: i for i, t_id in enumerate(self.unique_ids)}
        else:
            self.id_to_label = id_to_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'image_path': row['image_path'],
            'bbox': row['bbox'],
            'label': self.id_to_label.get(row['turtle_id'], -1),
            'turtle_id': row['turtle_id']
        }

class DataAgent:
    """
    Single Responsibility: Provisioning dataset metadata and building DataLoaders.
    """
    def __init__(self, data_dir=BASE_DATA_DIR):
        self.data_dir = data_dir
        
    def load_metadata(self):
        print(f"[DataAgent] Searching for metadata_splits.csv in /kaggle/input/...")
        # Dynamically find metadata_splits.csv to handle any Kaggle folder structure
        found_dir = None
        for root, dirs, files in os.walk("/kaggle/input"):
            if "metadata_splits.csv" in files or "metadata.csv" in files:
                found_dir = root
                break
                
        if found_dir is None:
            # Let's list the directory contents for debugging if not found
            import glob
            print("Contents of /kaggle/input/:")
            for p in glob.glob("/kaggle/input/*"):
                print(p)
            raise FileNotFoundError(f"Could not find metadata_splits.csv anywhere inside /kaggle/input/. Lütfen sağ panelden veri setini eklediğinize emin olun.")
            
        self.data_dir = found_dir
        
        splits_path = os.path.join(self.data_dir, "metadata_splits.csv")
        if not os.path.exists(splits_path):
            splits_path = os.path.join(self.data_dir, "metadata.csv")
            
        annotations_path = os.path.join(self.data_dir, "annotations.json")

        df = pd.read_csv(splits_path)
        
        anno_data = {}
        if os.path.exists(annotations_path):
            with open(annotations_path, 'r') as f:
                anno_data = json.load(f)
            
        bbox_map = {}
        for ann in anno_data.get('annotations', []):
            if ann.get('bbox'):
                bbox_map[ann['image_id']] = ann['bbox']
                
        records = []
        
        # Identify correct columns
        col_file = 'file_name' if 'file_name' in df.columns else 'image_name'
        if 'image_path' in df.columns: col_file = 'image_path'
            
        col_id = 'identity' if 'identity' in df.columns else 'turtle_id'
        if 'name' in df.columns: col_id = 'name'
            
        col_split = 'split_closed' if 'split_closed' in df.columns else 'split'
        if col_split not in df.columns:
            # Fake split
            df['split'] = 'train'
            col_split = 'split'
            
        for _, row in df.iterrows():
            # Extract
            fname = str(row[col_file]).replace('/', os.sep)
            img_path = os.path.join(self.data_dir, fname)
            
            # Map split
            split_val = str(row[col_split]).lower().strip()
            if split_val in ['valid', 'validation', 'query']: 
                split_val = 'val'
            elif split_val in ['database', 'train']:
                split_val = 'train'
            else:
                split_val = 'test'
                
            img_id = row.get('id', row.get('image_id', None))
            
            records.append({
                'image_path': img_path,
                'turtle_id': row[col_id],
                'bbox': bbox_map.get(img_id, None),
                'split': split_val
            })
            
        return records

    def get_dataloaders(self):
        records = self.load_metadata()
        
        train_records = [r for r in records if r['split'] == 'train']
        val_records = [r for r in records if r['split'] == 'val']
        test_records = [r for r in records if r['split'] == 'test']
        
        # Fallbacks
        if not train_records:
            print("WARNING: No 'train' split found. Using everything for train.")
            train_records = records
        if not val_records:
            print("WARNING: No 'val' split found. Using test for val.")
            val_records = test_records if test_records else train_records
        if not test_records:
            print("WARNING: No 'test' split found. Using val for test.")
            test_records = val_records if val_records else train_records
            
        train_df = pd.DataFrame(train_records)
        val_df = pd.DataFrame(val_records)
        test_df = pd.DataFrame(test_records)
        print(f"[DataAgent] Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # No augmentation for extraction, just normalize
        transforms = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = SeaTurtleDataset(train_df, transform=transforms)
        val_dataset = SeaTurtleDataset(val_df, id_to_label=train_dataset.id_to_label, transform=transforms)
        test_dataset = SeaTurtleDataset(test_df, id_to_label=train_dataset.id_to_label, transform=transforms)
        
        return {
            'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS),
            'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS),
            'test': DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS),
            'num_classes': len(train_dataset.unique_ids)
        }

class CroppedTurtleDataset(Dataset):
    """
    Wrapper Dataset.
    Single Responsibility: Crop head regions from the original image using bbox.
    """
    def __init__(self, base_dataset, transform=None):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self): 
        return len(self.base_dataset)
        
    def crop_head(self, image, bbox):
        h_img, w_img = image.shape[:2]
        if bbox is None or len(bbox) != 4:
            return image[h_img//4:3*h_img//4, w_img//4:3*w_img//4]
        
        x, y, w, h = [int(v) for v in bbox]
        pad = 20
        x, y = max(0, x - pad), max(0, y - pad)
        w, h = min(w_img - x, w + 2*pad), min(h_img - y, h + 2*pad)
        
        cropped_img = image[y:y+h, x:x+w]
        return cropped_img if cropped_img.size > 0 else image[h_img//4:3*h_img//4, w_img//4:3*w_img//4]

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        img_path = sample['image_path']
        
        if os.path.exists(img_path):
            image = np.array(Image.open(img_path).convert('RGB'))
        else:
            image = np.zeros((500, 500, 3), dtype=np.uint8)
            cv2.circle(image, (250, 250), 100, (0, 255, 0), -1)

        cropped_head = self.crop_head(image, sample['bbox'])
        cropped_pil = Image.fromarray(cropped_head).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        tensor_img = self.transform(cropped_pil) if self.transform else T.ToTensor()(cropped_pil)

        return {'image': tensor_img, 'label': sample['label'], 'turtle_id': sample['turtle_id']}

class SegmentationAgent:
    """
    Single Responsibility: Apply cropping logic to datasets.
    """
    def process(self, data_dict):
        print("[SegmentationAgent] Wrapping datasets with head cropping logic...")
        heads_data = {'num_classes': data_dict['num_classes']}
        
        for split in ['train', 'val', 'test']:
            ds = data_dict[split].dataset
            cropped_ds = CroppedTurtleDataset(ds, transform=ds.transform)
            heads_data[split] = DataLoader(cropped_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
            
        print("[SegmentationAgent] Processing pipeline configured.")
        return heads_data

class FeatureExtractorAgent:
    """
    Single Responsibility: Feature extraction using pretrained model (MegaDescriptor).
    """
    def __init__(self):
        self.device = DEVICE
        print(f"[FeatureExtractorAgent] Loading MegaDescriptor-T-224...")
        self.model = timm.create_model('hf-hub:BVRA/MegaDescriptor-T-224', num_classes=0, pretrained=True).to(self.device)

    def extract(self, heads_data):
        print(f"[FeatureExtractorAgent] Extracting features using PyTorch...")
        embeddings_data = {'num_classes': heads_data['num_classes']}
        
        self.model.eval()
        for split in ['train', 'val', 'test']:
            print(f" - Processing {split} split...")
            loader = heads_data[split]
            
            features_list = []
            labels_list = []
            
            with torch.no_grad():
                for batch in loader:
                    images = batch['image'].to(self.device)
                    feats = self.model(images)
                    features_list.append(feats.cpu())
                    labels_list.extend(batch['label'].tolist())
            
            if features_list:
                features = torch.cat(features_list, dim=0)
            else:
                features = torch.empty(0)
            
            embeddings_data[split] = {
                'embeddings': features,
                'labels': torch.tensor(labels_list)
            }
        return embeddings_data

class MatchingAgent:
    """
    Single Responsibility: Calculate similarity and predict closest match (kNN).
    """
    def predict(self, embeddings_data):
        print("[MatchingAgent] Calculating Cosine Similarity...")
        train_features = embeddings_data['train']['embeddings']
        train_labels = embeddings_data['train']['labels']
        test_features = embeddings_data['test']['embeddings']
        test_labels = embeddings_data['test']['labels']
        
        # Calculate Cosine Similarity Matrix
        test_norm = torch.nn.functional.normalize(test_features, p=2, dim=1)
        train_norm = torch.nn.functional.normalize(train_features, p=2, dim=1)
        similarity_matrix = torch.mm(test_norm, train_norm.t())
        
        print("[MatchingAgent] Running kNN Classification (k=1)...")
        preds_top1_arr = []
        preds_top5 = []
        
        for i in range(len(test_features)):
            similarities = similarity_matrix[i]
            sorted_indices = torch.argsort(similarities, descending=True)
            sorted_labels = train_labels[sorted_indices]
            
            unique_labels = []
            for lbl in sorted_labels:
                lbl_val = lbl.item()
                if lbl_val not in unique_labels: 
                    unique_labels.append(lbl_val)
                if len(unique_labels) == 5: 
                    break
            
            preds_top1_arr.append(unique_labels[0])
            preds_top5.append(unique_labels)
            
        return {
            'y_true': test_labels.numpy(),
            'y_pred_top1': preds_top1_arr,
            'y_pred_top5': preds_top5
        }

class EvaluationAgent:
    """
    Single Responsibility: Metrics calculation and visualization.
    """
    def evaluate(self, predictions):
        y_true = predictions['y_true']
        y_pred_top1 = predictions['y_pred_top1']
        y_pred_top5 = predictions['y_pred_top5']

        top1_acc = accuracy_score(y_true, y_pred_top1)
        top5_correct = sum([1 for t, p5 in zip(y_true, y_pred_top5) if t in p5])
        top5_acc = top5_correct / len(y_true) if len(y_true) > 0 else 0.0

        print(f"\n[EvaluationAgent] Evaluation Results:")
        print(f" - Top-1 Accuracy: {top1_acc:.4f} ({top1_acc*100:.1f}%)")
        print(f" - Top-5 Accuracy: {top5_acc:.4f} ({top5_acc*100:.1f}%)")

        cm = confusion_matrix(y_true, y_pred_top1)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, cmap='Blues', fmt='g')
        plt.title('Confusion Matrix (Top-1)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

# %% [markdown]
# # Step 4: Main Pipeline Execution
# %%
def main():
    import gc
    print("="*50)
    print("Kaggle Sea Turtle Identification Pipeline - Starting")
    print("="*50)
    
    data = DataAgent().get_dataloaders()
    gc.collect()
    
    heads = SegmentationAgent().process(data)
    gc.collect()
    
    features = FeatureExtractorAgent().extract(heads)
    gc.collect()
    
    predictions = MatchingAgent().predict(features)
    gc.collect()
    
    EvaluationAgent().evaluate(predictions)
    gc.collect()
    
    print("="*50)
    print("Pipeline Execution Completed. MegaDescriptor Strategy is Done.")
    print("="*50)

# %% [markdown]
# # Step 5: Run
# %%
if __name__ == "__main__":
    main()
