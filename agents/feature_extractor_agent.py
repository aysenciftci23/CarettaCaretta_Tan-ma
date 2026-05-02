import timm
import torch
from torch.utils.data import Dataset
from wildlife_tools.features import DeepFeatures
from config import DEVICE, BATCH_SIZE, NUM_WORKERS

class DictToTupleDataset(Dataset):
    """
    wildlife_tools DeepFeatures usually expects a dataset returning (image, label) tuples.
    This wrapper converts our dictionary outputs to tuples.
    """
    def __init__(self, original_dataset):
        self.original = original_dataset
        
    def __len__(self):
        return len(self.original)
        
    def __getitem__(self, idx):
        item = self.original[idx]
        return item['image'], item['label']

class FeatureExtractorAgent:
    """
    Agent responsible for extracting features using pretrained MegaDescriptor.
    Single Responsibility: Feature extraction from images.
    """
    def __init__(self):
        self.device = DEVICE
        print(f"[FeatureExtractorAgent] Loading MegaDescriptor-T-224...")
        name = 'hf-hub:BVRA/MegaDescriptor-T-224'
        # MegaDescriptor is pretrained, so we set pretrained=True and num_classes=0 to get features
        self.model = timm.create_model(name, num_classes=0, pretrained=True)
        self.model = self.model.to(self.device)
        self.extractor = DeepFeatures(self.model, device=self.device, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    def extract(self, heads_data):
        print(f"[FeatureExtractorAgent] Extracting features using pretrained model...")
        
        embeddings_data = {}
        embeddings_data['num_classes'] = heads_data['num_classes']
        
        for split in ['train', 'val', 'test']:
            print(f"[FeatureExtractorAgent] Processing {split} split...")
            loader = heads_data[split]
            dataset = loader.dataset
            
            # Wrap dataset for DeepFeatures
            wrapped_dataset = DictToTupleDataset(dataset)
            
            # Extract features using wildlife_tools
            features = self.extractor(wrapped_dataset)
            
            # Convert features to tensor if it's a numpy array
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features)
            
            # Extract labels instantly without loading images
            labels = []
            ids = []
            base_df = dataset.base_dataset.df
            id_to_label = dataset.base_dataset.id_to_label
            
            for i in range(len(base_df)):
                turtle_id = base_df.iloc[i]['turtle_id']
                label = id_to_label.get(turtle_id, -1)
                labels.append(label)
                ids.append(turtle_id)
                
            labels = torch.tensor(labels)
            
            embeddings_data[split] = {
                'embeddings': features,
                'labels': labels,
                'turtle_ids': ids
            }
            
        print("[FeatureExtractorAgent] Feature extraction complete.")
        return embeddings_data
