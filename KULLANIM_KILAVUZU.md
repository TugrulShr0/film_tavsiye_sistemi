# Film Tavsiye Sistemi - Kullanım Kılavuzu

**Tuğrul Şahar (233255027) - Burak Yetişer (233255007)**

## Hızlı Başlangıç

### 1. Kurulum

```bash
# Projeyi klonlayın veya indirin
cd film_tavsiye_sistemi

# Sanal ortam oluşturun (önerilen)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows

# Gerekli paketleri yükleyin
pip install -r requirements.txt
```

### 2. Veri Setini İndirin

```bash
# MovieLens 100K veri setini otomatik indir
python download_data.py

# Veya manuel olarak:
# 1. https://files.grouplens.org/datasets/movielens/ml-100k.zip adresinden indirin
# 2. data/raw/ klasörüne çıkarın
```

### 3. Demo'yu Çalıştırın (Opsiyonel)

Model mimarisini ve eğitim sürecini test edin:

```bash
python demo.py
```

### 4. Veri Analizi

Jupyter Notebook ile veri setini keşfedin:

```bash
jupyter notebook notebooks/01_veri_analizi.ipynb
```

### 5. Model Eğitimi

Varsayılan parametrelerle:

```bash
python train.py
```

Özel parametrelerle:

```bash
python train.py --epochs 100 --batch_size 512 --lr 0.0005 --embedding_dim 128
```

### 6. Film Önerileri Alın

Bir kullanıcı için top-10 öneri:

```bash
python predict.py --user_id 123 --top_k 10
```

Belirli bir film için puan tahmini:

```bash
python predict.py --user_id 123 --movie_id 456
```

---

## Detaylı Kullanım

### Model Eğitimi Parametreleri

```bash
python train.py \
  --data_dir data/raw/ml-100k/ \
  --epochs 50 \
  --batch_size 256 \
  --lr 0.001 \
  --embedding_dim 64
```

**Parametreler:**
- `--data_dir`: Veri dizini
- `--epochs`: Epoch sayısı (varsayılan: 50)
- `--batch_size`: Batch boyutu (varsayılan: 256)
- `--lr`: Öğrenme oranı (varsayılan: 0.001)
- `--embedding_dim`: Embedding boyutu (varsayılan: 64)
- `--resume`: Devam edilecek checkpoint yolu

**Örnek Çıktı:**
```
========================================
MODEL EĞİTİMİ BAŞLIYOR
========================================
Cihaz: cuda
Epoch Sayısı: 50
Batch Size: 256
Learning Rate: 0.001
========================================

Epoch 1/50 - 12.34s
Train Loss: 1.2345 | RMSE: 1.1234 | MAE: 0.8765
Val   Loss: 1.1234 | RMSE: 1.0543 | MAE: 0.8234
✓ En iyi model kaydedildi!
...
```

### Tahmin ve Öneri

**Kullanıcı için top-K öneri:**

```bash
python predict.py --user_id 1 --top_k 10
```

Çıktı:
```
========================================
KULLANICI 1 İÇİN FİLM ÖNERİLERİ
========================================

Kullanıcının En Beğendiği Filmler:
+------+--------------------------------+-------+
|   ID | Film Adı                       | Puan  |
+======+================================+=======+
|  123 | Star Wars (1977)              | 5.0★  |
|  456 | The Empire Strikes Back (1980)| 5.0★  |
|  789 | Return of the Jedi (1983)     | 4.0★  |
+------+--------------------------------+-------+

Önerilen Filmler (Top 10):
+---+------+--------------------------------+--------------+
| # |   ID | Film Adı                       | Tahmini Puan |
+===+======+================================+==============+
| 1 |  234 | Raiders of the Lost Ark (1981) | 4.87★        |
| 2 |  567 | The Matrix (1999)              | 4.76★        |
...
```

**Belirli film için tahmin:**

```bash
python predict.py --user_id 1 --movie_id 100
```

**Kullanıcı tahminlerini değerlendir:**

```bash
python predict.py --user_id 1 --evaluate
```

### Checkpoint'tan Devam Etme

```bash
# Belirli bir checkpoint'tan devam et
python train.py --resume models/checkpoint_epoch_25.pth

# En iyi modelden devam et
python train.py --resume models/best_model.pth
```

---

## Proje Yapısı Detayları

```
film_tavsiye_sistemi/
│
├── config.py                  # Konfigürasyon ayarları
├── ncf_model.py              # Model mimarileri (NCF, MF)
├── train.py                   # Eğitim scripti
├── predict.py                 # Tahmin ve öneri scripti
├── demo.py                    # Demo ve test scripti
├── download_data.py           # Veri indirme scripti
├── requirements.txt           # Gerekli paketler
│
├── data/                      # Veri dizini
│   ├── raw/                  # Ham veri
│   │   └── ml-100k/          # MovieLens veri seti
│   └── processed/            # İşlenmiş veri
│
├── models/                    # Kaydedilmiş modeller
│   ├── best_model.pth        # En iyi model
│   └── checkpoint_epoch_*.pth # Checkpoint'lar
│
├── notebooks/                 # Jupyter Notebook'lar
│   ├── 01_veri_analizi.ipynb
│   ├── 02_veri_onisleme.ipynb
│   └── 03_model_egitim.ipynb
│
├── utils/                     # Yardımcı modüller
│   ├── __init__.py
│   ├── data_loader.py        # Veri yükleme
│   ├── preprocessing.py      # Veri ön işleme (gelecek)
│   └── evaluation.py         # Model değerlendirme
│
└── results/                   # Sonuçlar
    ├── training_history.png  # Eğitim grafikleri
    └── predictions/          # Tahmin sonuçları
```

---

## Sık Karşılaşılan Sorunlar

### 1. CUDA/GPU Hataları

**Sorun:** CUDA bulunamadı veya GPU kullanılamıyor

**Çözüm:**
- CPU ile çalıştırmak için `config.py` dosyasında `DEVICE = "cpu"` yapın
- PyTorch CUDA versiyonunu kontrol edin: `torch.cuda.is_available()`

### 2. Veri Seti Bulunamadı

**Sorun:** `FileNotFoundError: [Errno 2] No such file or directory`

**Çözüm:**
```bash
# Veriyi indirin
python download_data.py

# Veya manuel olarak data/raw/ml-100k/ klasörüne yerleştirin
```

### 3. Bellek Hatası

**Sorun:** `RuntimeError: CUDA out of memory`

**Çözüm:**
- Batch size'ı küçültün: `--batch_size 128`
- Embedding boyutunu küçültün: `--embedding_dim 32`
- CPU kullanın

### 4. Import Hataları

**Sorun:** `ModuleNotFoundError: No module named 'torch'`

**Çözüm:**
```bash
# Gereksinimleri yeniden yükleyin
pip install -r requirements.txt
```

---

## İleri Seviye Kullanım

### Hiperparametre Optimizasyonu

`config.py` dosyasını düzenleyerek:

```python
# Daha derin model
EMBEDDING_DIM = 128
HIDDEN_LAYERS = [256, 128, 64, 32]

# Daha agresif öğrenme
LEARNING_RATE = 0.003
BATCH_SIZE = 512

# Regularization
DROPOUT_RATE = 0.3
WEIGHT_DECAY = 1e-4
```

### Kendi Modelinizi Ekleyin

`ncf_model.py` dosyasına yeni model sınıfı ekleyin:

```python
class MyCustomModel(nn.Module):
    def __init__(self, num_users, num_movies):
        super().__init__()
        # Model katmanlarınız
        
    def forward(self, user_ids, movie_ids):
        # Forward pass
        return predictions
```

### Web Uygulaması (Gelecek Geliştirme)

Flask veya FastAPI ile REST API oluşturabilirsiniz:

```python
from flask import Flask, request, jsonify
from predict import MovieRecommender

app = Flask(__name__)
recommender = MovieRecommender('models/best_model.pth', data_loader)

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = request.json['user_id']
    recommendations = recommender.recommend_movies(user_id, top_k=10)
    return jsonify(recommendations.to_dict())
```

---

## Performans İpuçları

1. **GPU Kullanımı:** Mümkünse CUDA destekli GPU kullanın
2. **Batch Size:** GPU belleğine göre optimize edin (genelde 256-1024)
3. **Num Workers:** DataLoader için CPU çekirdek sayınıza göre ayarlayın
4. **Mixed Precision:** Daha hızlı eğitim için `torch.cuda.amp` kullanın
5. **Gradient Accumulation:** Küçük GPU'larda büyük batch effect için

---

## Katkıda Bulunma

Proje açık kaynaklıdır. Geliştirme önerileri:

- [ ] Attention mechanism ekleme
- [ ] Çok katmanlı model denemeleri
- [ ] Film içerik bilgilerini kullanma
- [ ] Kullanıcı demografik özelliklerini ekleme
- [ ] A/B testing framework'ü
- [ ] Web arayüzü

---

## Lisans ve İletişim

**Proje Ekibi:**
- Tuğrul Şahar (233255027)
- Burak Yetişer (233255007)

**Veri Seti:**
- F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: 
  History and Context. ACM Transactions on Interactive Intelligent 
  Systems (TiiS) 5, 4: 19:1–19:19.

---

## Kaynaklar

- PyTorch Documentation: https://pytorch.org/docs/
- MovieLens Dataset: https://grouplens.org/datasets/movielens/
- Neural Collaborative Filtering Paper: https://arxiv.org/abs/1708.05031
- Recommender Systems Handbook
