import streamlit as st
from PIL import Image
import numpy as np
import os

# Import Agents
from agents.segmentation_agent import SegmentationAgent
from agents.model_agent import ModelAgent
from agents.matching_agent import MatchingAgent
from agents.data_agent import DataAgent

# Initialize Agents
@st.cache_resource
def load_agents():
    seg_agent = SegmentationAgent()
    model_agent = ModelAgent()
    match_agent = MatchingAgent()
    data_agent = DataAgent() # Added for reference images
    return seg_agent, model_agent, match_agent, data_agent

def main():
    st.set_page_config(page_title="Sea Turtle ID", page_icon="🐢", layout="wide")
    
    # Custom CSS for a premium feel
    st.markdown("""
        <style>
        .main {
            background-color: #f0f2f6;
        }
        .stAlert {
            border-radius: 10px;
        }
        .turtle-card {
            background-color: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🐢 Sea Turtle Identification System")
    st.write("Bu sistem, kaplumbağaların yüzündeki scute (kabuksu yapı) desenlerini analiz ederek birey tanılama yapar.")

    # Sidebar for instructions
    with st.sidebar:
        st.header("🛠️ Başarıyı Artırmak İçin")
        st.info("""
        **1. Yakın Çekim Önemlidir:** Kaplumbağanın sadece kafa bölgesini (yan profil) yüklemeye çalışın. Model, desenleri (scutes) bu bölgeden tanır.
        
        **2. Bakış Açısı:** Mümkünse yan profilden çekilmiş fotoğraflar kullanın. Üstten çekimler başarıyı düşürür.
        
        **3. Netlik:** Bulanık veya çok karanlık fotoğraflarda model yanılabilir.
        """)
        
        st.warning("""
        **Not:** Eğer kaplumbağa veri setinde olmasına rağmen tanınmıyorsa, lütfen fotoğrafı kafa bölgesine odaklanacak şekilde kırparak tekrar deneyin.
        """)

    # File uploader
    uploaded_file = st.file_uploader("Bir Görsel Seçin (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Load agents safely
            seg_agent, model_agent, match_agent, data_agent = load_agents()
            
            # 1. Load image
            image = Image.open(uploaded_file).convert('RGB')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📸 Yüklenen Fotoğraf")
                st.image(image, use_container_width=True)
            
            with st.spinner("AI analiz ediyor... Bu işlem scute desenlerini karşılaştırmayı içerir."):
                # 2. Preprocess
                preprocessed_img = seg_agent.preprocess_image(image)
                
                # 3. Model Prediction
                probabilities = model_agent.predict(preprocessed_img)
                
                # 4. ID Matching
                result = match_agent.match(probabilities)
                turtle_id = result['turtle_id']
                confidence = result['confidence']
            
            with col2:
                st.subheader("🔍 En Yakın Eşleşme")
                # Try to get reference image
                ref_img_path = data_agent.get_sample_image_for_id(turtle_id)
                
                if ref_img_path and os.path.exists(ref_img_path):
                    ref_image = Image.open(ref_img_path)
                    st.image(ref_image, caption=f"Veri Setindeki Referans: {turtle_id}", use_container_width=True)
                else:
                    st.warning(f"Referans görsel bulunamadı (ID: {turtle_id}).")
                    st.write("Veri seti yolu yapılandırılmamış olabilir.")

            # Display Results in a nice card
            st.markdown("---")
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.metric(label="Tespit Edilen Kimlik", value=f"ID #{turtle_id}")
            
            with res_col2:
                st.write(f"**Güven Skoru: %{confidence * 100:.2f}**")
                st.progress(min(confidence, 1.0))
                
                if confidence > 0.8:
                    st.success("🌟 Yüksek Güven: Model bu kaplumbağadan oldukça emin.")
                elif confidence > 0.5:
                    st.info("ℹ️ Orta Güven: Desenler benzerlik gösteriyor.")
                else:
                    st.warning("⚠️ Düşük Güven: Bu kaplumbağa veri setinde olmayabilir veya fotoğraf net değil.")

        except Exception as e:
            st.error(f"Sistem Hatası: {str(e)}")
            st.info("İpucu: Veri setinin 'C:\\CarettaProje' altında olduğundan emin olun.")

if __name__ == "__main__":
    main()
