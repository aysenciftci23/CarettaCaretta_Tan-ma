# 🐢 Sea Turtle Identification System

Bu proje, Caretta caretta bireylerini yüzlerindeki scute (pul) desenlerinden tanımaya yarayan yapay zeka tabanlı bir **Streamlit Web Uygulamasıdır**. 

Önceden eğitilmiş (Kaggle ortamında) **EfficientNetB3** tabanlı bir Keras modeli (`sea_turtle_model_v1.keras`) kullanılarak, kullanıcıların yüklediği kaplumbağa fotoğrafları saniyeler içinde analiz edilir ve kaplumbağanın ID'si (% güven skoru ile birlikte) tespit edilir.

## 🏗️ Mimari ve SOLID Prensipleri

Sistem, S (Single Responsibility) ve D (Dependency Inversion) başta olmak üzere SOLID prensiplerine ve Clean Code standartlarına tam uyumlu olarak tasarlanmıştır:

1. **Arayüz Katmanı (Predictor - `agents/predictor.py`)**
   - *Dependency Inversion:* Sistemin çekirdeği, modelin TensorFlow mu yoksa PyTorch mu olduğunu bilmez. `Predictor` soyut sınıfı üzerinden haberleşir.

2. **ModelAgent (`agents/model_agent.py`)**
   - Sorumluluk: `sea_turtle_model_v1.keras` dosyasını yükler, görüntüyü `(1, 224, 224, 3)` boyutuna getirir ve `EfficientNetB3.preprocess_input` adımını uygulayarak tahmin yapar.

3. **SegmentationAgent (`agents/segmentation_agent.py`)**
   - Sorumluluk: Kullanıcının yüklediği orijinal görseli analiz için 224x224 boyutuna (IMG_SIZE) getirip ön işlemlerden geçirir.

4. **MatchingAgent (`agents/matching_agent.py`)**
   - Sorumluluk: Modelden dönen olasılık dağılım dizisindeki (probabilities) en yüksek skoru (argmax) bulur ve bunu `models/id_to_label.json` sözlüğü yardımıyla gerçek Turtle ID'sine dönüştürür.

5. **Web Arayüzü (`app.py`)**
   - Sorumluluk: Kullanıcı ile sistemin etkileşimini sağlar. Streamlit altyapısı ile sadece dosya yükleme ve sonuç gösterme işlemlerini üstlenir.

## 🚀 Kurulum ve Çalıştırma

Proje Python 3.10+ üzerinde çalışmaktadır. Aşağıdaki adımları takip ederek projeyi kendi bilgisayarınızda ayağa kaldırabilirsiniz:

### 1. Gereksinimleri Yükleyin
```bash
pip install streamlit tensorflow pillow numpy pandas
```

### 2. Modeli Projeye Ekleyin
Projenin çalışabilmesi için Kaggle'da eğitilen **100MB** boyutlarındaki model dosyasına ihtiyacınız var.
- `sea_turtle_model_v1.keras` dosyasını indirin ve projenin ana dizinine yerleştirin.

### 3. Uygulamayı Başlatın
Terminal veya komut satırından projenin bulunduğu klasöre gidin ve şu komutu çalıştırın:
```bash
streamlit run app.py
```

Tarayıcınızda otomatik olarak açılan ekrandan "Bir Görsel Seçin" butonuna tıklayarak kaplumbağa fotoğrafı yükleyebilir ve sonucu görebilirsiniz!

## 📌 Notlar
- `models/id_to_label.json` dosyası, modelin 0-437 arası döndürdüğü sınıfları gerçek ID'lere dönüştürür. Eğer bu dosya eksikse sistem otomatik olarak MOCK bir JSON dosyası oluşturur.
- Model ağırlıkları GitHub dosya boyutu limitlerinden dolayı `.gitignore` ile yoksayılmıştır. Kodları clone'ladıktan sonra modelinizi manuel eklemeyi unutmayın.
