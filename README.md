# 🎬 Film Tavsiye Sistemi

Makine öğrenmesi tabanlı içerik bazlı (Content-Based Filtering) film öneri sistemi.
Kullanıcının seçtiği filme benzer yapımları analiz ederek öneriler sunar.

---

# 🚀 Proje Hakkında

Bu proje, kullanıcıların sevdiği filmlere benzer filmleri önermek amacıyla geliştirilmiş bir öneri sistemidir.

Sistem:

* Film özelliklerini analiz eder
* Filmler arası benzerlik hesaplar
* Kullanıcıya benzer içerikte film önerileri sunar

Öneri sistemi yaklaşımı olarak **Content-Based Recommendation System** kullanılmıştır.

---

# 🧠 Kullanılan Teknolojiler

* Python
* Pandas
* NumPy
* Scikit-learn
* Cosine Similarity
* Jupyter Notebook

---

# ⚙️ Çalışma Mantığı

Sistem aşağıdaki adımlarla çalışır:

1. Film verileri yüklenir
2. Film özellikleri işlenir
3. Özellik vektörleri oluşturulur
4. Filmler arası benzerlik skorları hesaplanır
5. Seçilen filme en yakın filmler önerilir

Benzerlik hesabında:

* Cosine Similarity
* Feature Vectorization

yöntemleri kullanılmıştır.

---

# 📂 Proje Yapısı

```bash
film_tavsiye_sistemi/
│
├── FilmTavsiyeSistemi.ipynb
├── README.md
└── uygulama.png
```

---

# 📸 Uygulama Görseli

![Uygulama Görseli](uygulama.png)

---

# ▶️ Kurulum

Projeyi çalıştırmak için:

```bash
git clone https://github.com/TugrulShr0/film_tavsiye_sistemi.git
cd film_tavsiye_sistemi
```

Gerekli kütüphaneleri yükleyin:

```bash
pip install pandas numpy scikit-learn
```

Jupyter Notebook'u başlatın:

```bash
jupyter notebook
```

---

# 💡 Örnek Kullanım

Kullanıcının seçtiği bir film:

```python
recommend("Interstellar")
```

Örnek çıktı:

```bash
1. The Martian
2. Gravity
3. Inception
4. Arrival
5. Contact
```

---

# 🎯 Projenin Kazanımları

Bu proje kapsamında:

* Öneri sistemleri mantığı
* Veri ön işleme
* Feature engineering
* Benzerlik algoritmaları
* Makine öğrenmesi temelleri

konularında pratik yapılmıştır.

---

# 📈 Geliştirme Fikirleri

* Web arayüzü ekleme (Flask / FastAPI)
* Kullanıcı puanlama sistemi
* Collaborative Filtering desteği
* Gerçek zamanlı öneri sistemi
* TMDB API entegrasyonu

---

# 👨‍💻 Geliştiriciler

Burak Yetişer
Tuğrul Şahar

---
