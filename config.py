# “Projenin kontrol paneli”

import torch

class Config:
    """Model ve eğitim için genel konfigürasyonlar"""
    
    # Veri yolları
    DATA_DIR = "data/"
    RAW_DATA_DIR = "data/raw/ml-100k/"
    PROCESSED_DATA_DIR = "data/processed/"
    MODEL_DIR = "models/"
    RESULTS_DIR = "results/"
    
    # Veri dosyaları
    RATINGS_FILE = "u.data"
    MOVIES_FILE = "u.item"
    USERS_FILE = "u.user"
    
    # Model parametreleri
    EMBEDDING_DIM = 64          # Embedding boyutu
    HIDDEN_LAYERS = [128, 64, 32]  # Gizli katman boyutları
    DROPOUT_RATE = 0.2          # Dropout oranı
    
    # Eğitim parametreleri
    BATCH_SIZE = 256
    EPOCHS = 50
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    
    # Veri ayrımı
    TRAIN_RATIO = 0.8
    VAL_RATIO = 0.1
    TEST_RATIO = 0.1
    
    # Cihaz
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Rastgelelik kontrolü
    RANDOM_SEED = 42
    
    # Değerlendirme
    TOP_K = 10  # Top-K önerileri
    
    # Model kaydetme
    SAVE_MODEL = True
    MODEL_NAME = "ncf_model.pth"
    CHECKPOINT_FREQ = 5  # Her 5 epoch'ta bir kaydet
    
    # Early stopping
    PATIENCE = 10  # Early stopping için sabır
    MIN_DELTA = 0.001  # Minimum iyileşme miktarı
    
    # Logging
    VERBOSE = True
    LOG_INTERVAL = 100  # Her 100 batch'te bir log
    
    def __repr__(self):
        """Konfigürasyon bilgilerini yazdır"""
        config_str = "=" * 50 + "\n"
        config_str += "Film Tavsiye Sistemi Konfigürasyonu\n"
        config_str += "=" * 50 + "\n"
        config_str += f"Embedding Boyutu: {self.EMBEDDING_DIM}\n"
        config_str += f"Gizli Katmanlar: {self.HIDDEN_LAYERS}\n"
        config_str += f"Batch Size: {self.BATCH_SIZE}\n"
        config_str += f"Epoch Sayısı: {self.EPOCHS}\n"
        config_str += f"Öğrenme Oranı: {self.LEARNING_RATE}\n"
        config_str += f"Cihaz: {self.DEVICE}\n"
        config_str += "=" * 50
        return config_str

# Singleton pattern
config = Config()

if __name__ == "__main__":
    print(config)
