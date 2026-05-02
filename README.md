# Sea Turtle Identification System

Bu proje, Caretta caretta bireylerini yüzlerindeki scute (pul) desenlerinden tanımaya yarayan multi-agent tabanlı bir yapay zeka sistemidir.

## Mimari: Multi-Agent Pipeline

Proje, S (Single Responsibility) ve O (Open/Closed) SOLID prensiplerini uygulamak üzere 5 farklı ajan (agent) üzerinden tasarlanmıştır:

1. **DataAgent (`agents/data_agent.py`)**
   - Sorumluluk: Kaggle API üzerinden `seaturtleid2022` veri setini indirir, train/val/test ayrımlarını (zamana duyarlı) yapar ve Data Augmentation uygular.
   
2. **SegmentationAgent (`agents/segmentation_agent.py`)**
   - Sorumluluk: Orijinal maskeleri kullanarak kaplumbağa kafasını kırpar (crop) ve modele girmeden önce 224x224 boyutuna getirir.

3. **FeatureExtractorAgent (`agents/feature_extractor_agent.py`)**
   - Sorumluluk: Kırpılmış resimlerden 512 boyutlu embedding çıkarır. 
   - *Open/Closed Prensibi (O)*: `BaseFeatureExtractor` soyut sınıfı ile genişlemeye açık bir yapı kurulmuştur.

4. **MatchingAgent (`agents/matching_agent.py`)**
   - Sorumluluk: `ArcFace` loss kullanarak özellik uzayında kaplumbağaların daha iyi ayrışmasını sağlar. Test kümesinde ise Cosine Similarity (Kosinüs Benzerliği) ile 1-NN eşleştirmesi yapar.

5. **EvaluationAgent (`agents/evaluation_agent.py`)**
   - Sorumluluk: Tahminleri gerçek etiketlerle karşılaştırarak Top-1 ve Top-5 doğruluğunu hesaplar. Karışıklık matrisi (Confusion Matrix) oluşturur ve `results/` klasörüne kaydeder.

## Gereksinimler ve Kurulum

Proje Python 3.10+ üzerinde çalışmaktadır. Aşağıdaki adımları takip ederek kurabilirsiniz:

1. Gereksinimleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
   *(Not: requirements.txt dosyası CUDA 11.8 ile PyTorch yükleyecek şekilde ayarlanmıştır.)*

2. Kaggle veri setini indirmek için Kaggle API gereklidir:
   Proje ana dizininde (`C:\Users\aysen\OneDrive\Masaüstü\CarettaCaretta_Tanıma`) bulunan `kaggle.json` dosyasının geçerli bir Kaggle API token barındırdığından emin olun.
   Eğer veriyi kendiniz indirdiyseniz, veri klasörünü proje altındaki `data/` klasörüne kopyalayabilirsiniz.

## Çalıştırma

Tüm pipeline'ı çalıştırmak için:

```bash
python main.py
```

Sistem otomatik olarak GPU varlığını kontrol eder:
- GPU varsa: Tüm veri setiyle tam iterasyon çalıştırılır.
- GPU yoksa: CPU üzerinde sadece 50 kaplumbağadan oluşan bir alt küme (subset) ile eğitim yapılır.

## Çıktılar

Sonuçlar `results/` klasörü altına kaydedilir:
- `metrics.json`: Top-1 ve Top-5 doğruluk oranları.
- `confusion_matrix.png`: Modelin tahminlerine ait hata matrisi grafiği.
