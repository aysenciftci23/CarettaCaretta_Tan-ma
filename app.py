import streamlit as st
from PIL import Image
import numpy as np

# Import Agents
from agents.segmentation_agent import SegmentationAgent
from agents.model_agent import ModelAgent
from agents.matching_agent import MatchingAgent

# Initialize Agents
@st.cache_resource
def load_agents():
    seg_agent = SegmentationAgent()
    model_agent = ModelAgent()
    match_agent = MatchingAgent()
    return seg_agent, model_agent, match_agent

def main():
    st.set_page_config(page_title="Sea Turtle ID", page_icon="🐢", layout="centered")
    
    st.title("🐢 Sea Turtle Identification System")
    st.write("Lütfen tanımlamak istediğiniz deniz kaplumbağası fotoğrafını yükleyin.")

    # File uploader
    uploaded_file = st.file_uploader("Bir Görsel Seçin (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Load agents safely
            seg_agent, model_agent, match_agent = load_agents()
            
            # 1. Load image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Yüklenen Fotoğraf", use_container_width=True)
            
            with st.spinner("Görüntü analiz ediliyor... Lütfen bekleyin."):
                # 2. Preprocess (SegmentationAgent -> 224x224)
                preprocessed_img = seg_agent.preprocess_image(image)
                
                # 3. Model Prediction (ModelAgent -> EfficientNet Inference)
                probabilities = model_agent.predict(preprocessed_img)
                
                # 4. ID Matching (MatchingAgent -> Argmax & Map)
                result = match_agent.match(probabilities)
            
            # Display Result clearly
            st.success("✅ Analiz Tamamlandı!")
            st.markdown(f"### 🐢 Tespit Edilen Kaplumbağa: ID #{result['turtle_id']}")
            st.markdown(f"### 📊 Güven Skoru: %{result['confidence'] * 100:.2f}")

        except Exception as e:
            st.error(f"Sistem Hatası: Model henüz yüklenmemiş veya dosya bozuk olabilir. Detay: {str(e)}")

if __name__ == "__main__":
    main()
