"""
Veri Ön İşleme Modülü
Tuğrul Şahar (233255027) - Burak Yetişer (233255007)

Bu modül veri temizleme, normalizasyon ve dönüşüm fonksiyonları içerir.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from datetime import datetime


class DataPreprocessor:
    """
    Veri ön işleme için yardımcı sınıf
    """
    
    def __init__(self):
        self.user_scaler = None
        self.movie_scaler = None
        self.rating_scaler = None
    
    @staticmethod
    def remove_duplicates(df):
        """
        Tekrar eden kayıtları kaldır
        
        Args:
            df (DataFrame): Veri çerçevesi
            
        Returns:
            DataFrame: Temizlenmiş veri
        """
        initial_len = len(df)
        df = df.drop_duplicates()
        removed = initial_len - len(df)
        
        if removed > 0:
            print(f"✓ {removed} tekrar eden kayıt kaldırıldı")
        
        return df
    
    @staticmethod
    def remove_missing_values(df, columns=None, strategy='drop'):
        """
        Eksik değerleri işle
        
        Args:
            df (DataFrame): Veri çerçevesi
            columns (list): İşlenecek sütunlar (None ise tümü)
            strategy (str): 'drop', 'mean', 'median', 'mode'
            
        Returns:
            DataFrame: İşlenmiş veri
        """
        if columns is None:
            columns = df.columns
        
        initial_len = len(df)
        
        if strategy == 'drop':
            df = df.dropna(subset=columns)
            removed = initial_len - len(df)
            if removed > 0:
                print(f"✓ {removed} eksik değerli satır kaldırıldı")
        
        elif strategy == 'mean':
            for col in columns:
                if df[col].dtype in [np.float64, np.int64]:
                    df[col].fillna(df[col].mean(), inplace=True)
        
        elif strategy == 'median':
            for col in columns:
                if df[col].dtype in [np.float64, np.int64]:
                    df[col].fillna(df[col].median(), inplace=True)
        
        elif strategy == 'mode':
            for col in columns:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    @staticmethod
    def filter_rare_items(df, user_col='user_id', item_col='movie_id', 
                         min_user_ratings=5, min_item_ratings=5):
        """
        Az puanlanan kullanıcı ve filmleri filtrele
        
        Args:
            df (DataFrame): Puanlama verileri
            user_col (str): Kullanıcı sütunu
            item_col (str): Film sütunu
            min_user_ratings (int): Min kullanıcı puanlama sayısı
            min_item_ratings (int): Min film puanlama sayısı
            
        Returns:
            DataFrame: Filtrelenmiş veri
        """
        initial_len = len(df)
        
        # Az puanlayan kullanıcıları kaldır
        user_counts = df[user_col].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        df = df[df[user_col].isin(valid_users)]
        
        # Az puanlanan filmleri kaldır
        item_counts = df[item_col].value_counts()
        valid_items = item_counts[item_counts >= min_item_ratings].index
        df = df[df[item_col].isin(valid_items)]
        
        removed = initial_len - len(df)
        print(f"✓ {removed} nadir puanlama kaldırıldı")
        print(f"  Kalan kullanıcı: {df[user_col].nunique()}")
        print(f"  Kalan film: {df[item_col].nunique()}")
        
        return df
    
    def normalize_ratings(self, ratings, method='minmax'):
        """
        Puanları normalize et
        
        Args:
            ratings (array): Puanlar
            method (str): 'minmax' veya 'standard'
            
        Returns:
            array: Normalize edilmiş puanlar
        """
        ratings = np.array(ratings).reshape(-1, 1)
        
        if method == 'minmax':
            if self.rating_scaler is None:
                self.rating_scaler = MinMaxScaler(feature_range=(0, 1))
                self.rating_scaler.fit(ratings)
            
            normalized = self.rating_scaler.transform(ratings)
        
        elif method == 'standard':
            if self.rating_scaler is None:
                self.rating_scaler = StandardScaler()
                self.rating_scaler.fit(ratings)
            
            normalized = self.rating_scaler.transform(ratings)
        
        else:
            raise ValueError(f"Bilinmeyen normalizasyon metodu: {method}")
        
        return normalized.flatten()
    
    def denormalize_ratings(self, normalized_ratings):
        """
        Normalize edilmiş puanları orijinal aralığa döndür
        
        Args:
            normalized_ratings (array): Normalize edilmiş puanlar
            
        Returns:
            array: Orijinal aralıktaki puanlar
        """
        if self.rating_scaler is None:
            raise ValueError("Scaler henüz fit edilmemiş!")
        
        normalized_ratings = np.array(normalized_ratings).reshape(-1, 1)
        denormalized = self.rating_scaler.inverse_transform(normalized_ratings)
        
        return denormalized.flatten()
    
    @staticmethod
    def extract_temporal_features(df, timestamp_col='timestamp'):
        """
        Zaman damgasından özellikler çıkar
        
        Args:
            df (DataFrame): Veri çerçevesi
            timestamp_col (str): Zaman damgası sütunu
            
        Returns:
            DataFrame: Zamansal özellikleri eklenmiş veri
        """
        # Unix timestamp'i datetime'a çevir
        df['datetime'] = pd.to_datetime(df[timestamp_col], unit='s')
        
        # Özellikler çıkar
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['hour'] = df['datetime'].dt.hour
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        print("✓ Zamansal özellikler eklendi")
        
        return df
    
    @staticmethod
    def create_user_features(ratings_df):
        """
        Kullanıcı bazlı özellikler oluştur
        
        Args:
            ratings_df (DataFrame): Puanlama verileri
            
        Returns:
            DataFrame: Kullanıcı özellikleri
        """
        user_features = ratings_df.groupby('user_id').agg({
            'rating': ['mean', 'std', 'count', 'min', 'max'],
            'movie_id': 'nunique'
        }).reset_index()
        
        user_features.columns = [
            'user_id', 
            'avg_rating', 
            'std_rating', 
            'num_ratings',
            'min_rating',
            'max_rating',
            'num_unique_movies'
        ]
        
        # Ek özellikler
        user_features['rating_range'] = user_features['max_rating'] - user_features['min_rating']
        user_features['is_generous'] = (user_features['avg_rating'] > 3.5).astype(int)
        user_features['is_active'] = (user_features['num_ratings'] > ratings_df.groupby('user_id').size().median()).astype(int)
        
        print("✓ Kullanıcı özellikleri oluşturuldu")
        
        return user_features
    
    @staticmethod
    def create_movie_features(ratings_df):
        """
        Film bazlı özellikler oluştur
        
        Args:
            ratings_df (DataFrame): Puanlama verileri
            
        Returns:
            DataFrame: Film özellikleri
        """
        movie_features = ratings_df.groupby('movie_id').agg({
            'rating': ['mean', 'std', 'count', 'min', 'max'],
            'user_id': 'nunique'
        }).reset_index()
        
        movie_features.columns = [
            'movie_id',
            'avg_rating',
            'std_rating',
            'num_ratings',
            'min_rating',
            'max_rating',
            'num_unique_users'
        ]
        
        # Popülerlik skoru
        movie_features['popularity_score'] = (
            movie_features['num_ratings'] / movie_features['num_ratings'].max()
        ) * movie_features['avg_rating']
        
        # Tartışmalılık skoru (yüksek std = tartışmalı)
        movie_features['controversy_score'] = movie_features['std_rating'].fillna(0)
        
        print("✓ Film özellikleri oluşturuldu")
        
        return movie_features
    
    @staticmethod
    def encode_categorical(df, columns, method='label'):
        """
        Kategorik değişkenleri encode et
        
        Args:
            df (DataFrame): Veri çerçevesi
            columns (list): Encode edilecek sütunlar
            method (str): 'label' veya 'onehot'
            
        Returns:
            DataFrame: Encode edilmiş veri
        """
        from sklearn.preprocessing import LabelEncoder
        
        if method == 'label':
            for col in columns:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col])
            
            print(f"✓ {len(columns)} kategorik sütun label encode edildi")
        
        elif method == 'onehot':
            df = pd.get_dummies(df, columns=columns, prefix=columns)
            print(f"✓ {len(columns)} kategorik sütun one-hot encode edildi")
        
        return df
    
    @staticmethod
    def balance_dataset(df, target_col='rating', method='undersample'):
        """
        Dengesiz veri setini dengele
        
        Args:
            df (DataFrame): Veri çerçevesi
            target_col (str): Hedef sütun
            method (str): 'undersample' veya 'oversample'
            
        Returns:
            DataFrame: Dengelenmiş veri
        """
        value_counts = df[target_col].value_counts()
        
        if method == 'undersample':
            # En az olan sınıfın sayısına göre örnekle
            min_samples = value_counts.min()
            
            balanced_dfs = []
            for rating in df[target_col].unique():
                rating_df = df[df[target_col] == rating]
                sampled_df = rating_df.sample(n=min_samples, random_state=42)
                balanced_dfs.append(sampled_df)
            
            df = pd.concat(balanced_dfs, ignore_index=True)
            print(f"✓ Undersample ile dengelendi: {len(df)} örnek")
        
        elif method == 'oversample':
            # En çok olan sınıfın sayısına göre örnekle
            max_samples = value_counts.max()
            
            balanced_dfs = []
            for rating in df[target_col].unique():
                rating_df = df[df[target_col] == rating]
                sampled_df = rating_df.sample(n=max_samples, replace=True, random_state=42)
                balanced_dfs.append(sampled_df)
            
            df = pd.concat(balanced_dfs, ignore_index=True)
            print(f"✓ Oversample ile dengelendi: {len(df)} örnek")
        
        return df
    
    def full_preprocessing_pipeline(self, ratings_df, config=None):
        """
        Tam veri ön işleme pipeline'ı
        
        Args:
            ratings_df (DataFrame): Ham puanlama verileri
            config (dict): Ön işleme konfigürasyonu
            
        Returns:
            DataFrame: İşlenmiş veri
        """
        if config is None:
            config = {
                'remove_duplicates': True,
                'remove_missing': True,
                'filter_rare': True,
                'min_user_ratings': 5,
                'min_item_ratings': 5,
                'extract_temporal': False,
                'normalize': False
            }
        
        print("\n" + "=" * 60)
        print("VERİ ÖN İŞLEME PIPELINE")
        print("=" * 60)
        
        df = ratings_df.copy()
        
        # Tekrar eden kayıtları kaldır
        if config.get('remove_duplicates', True):
            df = self.remove_duplicates(df)
        
        # Eksik değerleri kaldır
        if config.get('remove_missing', True):
            df = self.remove_missing_values(df)
        
        # Nadir kullanıcı/filmleri filtrele
        if config.get('filter_rare', True):
            df = self.filter_rare_items(
                df,
                min_user_ratings=config.get('min_user_ratings', 5),
                min_item_ratings=config.get('min_item_ratings', 5)
            )
        
        # Zamansal özellikler
        if config.get('extract_temporal', False) and 'timestamp' in df.columns:
            df = self.extract_temporal_features(df)
        
        # Normalizasyon
        if config.get('normalize', False):
            df['rating_normalized'] = self.normalize_ratings(df['rating'].values)
        
        print("=" * 60)
        print(f"✓ Ön işleme tamamlandı: {len(df)} kayıt")
        print("=" * 60 + "\n")
        
        return df


if __name__ == "__main__":
    # Test
    print("Preprocessing Modülü Test Ediliyor...")
    
    # Dummy veri
    np.random.seed(42)
    test_data = pd.DataFrame({
        'user_id': np.random.randint(1, 100, 1000),
        'movie_id': np.random.randint(1, 50, 1000),
        'rating': np.random.choice([1, 2, 3, 4, 5], 1000),
        'timestamp': np.random.randint(900000000, 1000000000, 1000)
    })
    
    preprocessor = DataPreprocessor()
    processed = preprocessor.full_preprocessing_pipeline(test_data)
    
    print(f"Ham veri: {len(test_data)} kayıt")
    print(f"İşlenmiş veri: {len(processed)} kayıt")
