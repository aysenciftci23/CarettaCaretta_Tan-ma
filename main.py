from agents.data_agent import DataAgent
from agents.segmentation_agent import SegmentationAgent
from agents.feature_extractor_agent import FeatureExtractorAgent
from agents.matching_agent import MatchingAgent
from agents.evaluation_agent import EvaluationAgent

def main():
    import gc
    print("="*50)
    print("Sea Turtle Identification Pipeline - Starting")
    print("="*50)
    
    # Adım 1: Veri Yükleme ve Ayrıştırma
    data_agent = DataAgent()
    data = data_agent.get_dataloaders()
    gc.collect()
    
    # Adım 2: Baş Bölgesinin Kırpılması (Segmentation)
    heads = SegmentationAgent().process(data)
    gc.collect()
    
    # Adım 3: Feature Extraction (Pretrained MegaDescriptor)
    feature_extractor = FeatureExtractorAgent()
    features = feature_extractor.extract(heads)
    gc.collect()
    
    # Adım 4: Matching (Cosine + kNN)
    matching_agent = MatchingAgent()
    predictions = matching_agent.predict(features)
    gc.collect()
    
    # Adım 5: Sonuçların Değerlendirilmesi
    EvaluationAgent().evaluate(predictions)
    gc.collect()
    
    print("="*50)
    print("Pipeline Execution Completed.")
    print("="*50)

if __name__ == "__main__":
    main()
