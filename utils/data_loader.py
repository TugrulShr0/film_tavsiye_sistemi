
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os


class MovieLensDataset(Dataset):

    
    def __init__(self, user_ids, movie_ids, ratings):
        self.user_ids = torch.LongTensor(user_ids)
        self.movie_ids = torch.LongTensor(movie_ids)
        self.ratings = torch.FloatTensor(ratings)
    
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        return self.user_ids[idx], self.movie_ids[idx], self.ratings[idx]


class DataLoader_ML:
 
    
    def __init__(self, data_dir="data/raw/ml-100k/", test_size=0.2, random_state=42):
       
        self.data_dir = data_dir
        self.test_size = test_size
        self.random_state = random_state
        
        
        self.ratings_df = None
        self.movies_df = None
        self.users_df = None
        
      
        self.train_data = None
        self.test_data = None
        self.val_data = None
        
        self.user_id_map = {}
        self.movie_id_map = {}
        self.reverse_user_map = {}
        self.reverse_movie_map = {}
        
        
        self.num_users = 0
        self.num_movies = 0
        self.num_ratings = 0
    
    def load_ratings(self):
       
        ratings_file = os.path.join(self.data_dir, "u.data")
        
        print(f"Puanlama verileri yükleniyor: {ratings_file}")
        
        self.ratings_df = pd.read_csv(
            ratings_file,
            sep='\t',
            names=['user_id', 'movie_id', 'rating', 'timestamp'],
            encoding='latin-1'
        )
        
        self.num_ratings = len(self.ratings_df)
        print(f"✓ {self.num_ratings:,} puanlama yüklendi")
        
        return self.ratings_df
    
    def load_movies(self):
       
        movies_file = os.path.join(self.data_dir, "u.item")
        
        print(f"Film bilgileri yükleniyor: {movies_file}")
        
        movie_cols = ['movie_id', 'title', 'release_date', 'video_release_date',
                      'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                      'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                      'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                      'Thriller', 'War', 'Western']
        
        self.movies_df = pd.read_csv(
            movies_file,
            sep='|',
            names=movie_cols,
            encoding='latin-1'
        )
        
        print(f"✓ {len(self.movies_df)} film yüklendi")
        
        return self.movies_df
    
    def load_users(self):
        users_file= os.path.join(self.data_dir, "u.user")
        
        if os.path.exists(users_file):
            print(f"Kullanıcı bilgileri yükleniyor: {users_file}")
            
            self.users_df = pd.read_csv(
                users_file,
                sep='|',
                names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
                encoding='latin-1'
            )
            
            print(f"✓ {len(self.users_df)} kullanıcı yüklendi")
            
            return self.users_df
        else:
            print("Kullanıcı bilgisi dosyası bulunamadı")
            return None
    
    def create_id_mappings(self):
       
        print("\nID mapping'leri oluşturuluyor...")
        
       
        unique_user_ids = self.ratings_df['user_id'].unique()
        unique_movie_ids = self.ratings_df['movie_id'].unique()
        
       
        self.user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_user_ids)}
        self.movie_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_movie_ids)}
        
      
        self.reverse_user_map = {v: k for k, v in self.user_id_map.items()}
        self.reverse_movie_map = {v: k for k, v in self.movie_id_map.items()}
        
       
        self.num_users = len(self.user_id_map)
        self.num_movies = len(self.movie_id_map)
        
        print(f"✓ {self.num_users} kullanıcı")
        print(f"✓ {self.num_movies} film")
        
        self.ratings_df['user_idx'] = self.ratings_df['user_id'].map(self.user_id_map)
        self.ratings_df['movie_idx'] = self.ratings_df['movie_id'].map(self.movie_id_map)
    
    def split_data(self, val_size=0.1):
     
        print(f"\nVeri ayrılıyor (Train/Val/Test)...")
        
        train_val_df, test_df = train_test_split(
            self.ratings_df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.ratings_df['rating'] 
        )
        
        val_ratio = val_size / (1 - self.test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=train_val_df['rating']
        )

        self.train_data = MovieLensDataset(
            train_df['user_idx'].values,
            train_df['movie_idx'].values,
            train_df['rating'].values
        )
        
        self.val_data = MovieLensDataset(
            val_df['user_idx'].values,
            val_df['movie_idx'].values,
            val_df['rating'].values
        )
        
        self.test_data = MovieLensDataset(
            test_df['user_idx'].values,
            test_df['movie_idx'].values,
            test_df['rating'].values
        )
        
        print(f"✓ Train: {len(self.train_data):,} örnek ({len(self.train_data)/len(self.ratings_df)*100:.1f}%)")
        print(f"✓ Validation: {len(self.val_data):,} örnek ({len(self.val_data)/len(self.ratings_df)*100:.1f}%)")
        print(f"✓ Test: {len(self.test_data):,} örnek ({len(self.test_data)/len(self.ratings_df)*100:.1f}%)")
    
    def get_data_loaders(self, batch_size=256, num_workers=2):
    
        train_loader = DataLoader(
            self.train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            self.test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def get_statistics(self):
   
        stats = {
            'num_users': self.num_users,
            'num_movies': self.num_movies,
            'num_ratings': self.num_ratings,
            'sparsity': 1 - (self.num_ratings / (self.num_users * self.num_movies)),
            'avg_rating': self.ratings_df['rating'].mean(),
            'std_rating': self.ratings_df['rating'].std(),
            'min_rating': self.ratings_df['rating'].min(),
            'max_rating': self.ratings_df['rating'].max(),
        }
        
        # Kullanıcı başına ortalama puanlama
        user_rating_counts = self.ratings_df.groupby('user_id').size()
        stats['avg_ratings_per_user'] = user_rating_counts.mean()
        stats['min_ratings_per_user'] = user_rating_counts.min()
        stats['max_ratings_per_user'] = user_rating_counts.max()
        
        # Film başına ortalama puanlama
        movie_rating_counts = self.ratings_df.groupby('movie_id').size()
        stats['avg_ratings_per_movie'] = movie_rating_counts.mean()
        stats['min_ratings_per_movie'] = movie_rating_counts.min()
        stats['max_ratings_per_movie'] = movie_rating_counts.max()
        
        return stats
    
    def print_statistics(self):
       
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("VERİ SETİ İSTATİSTİKLERİ")
        print("=" * 60)
        print(f"Toplam Kullanıcı Sayısı: {stats['num_users']:,}")
        print(f"Toplam Film Sayısı: {stats['num_movies']:,}")
        print(f"Toplam Puanlama Sayısı: {stats['num_ratings']:,}")
        print(f"Veri Seyrekliği: {stats['sparsity']:.2%}")
        print(f"\nOrtalama Puan: {stats['avg_rating']:.2f} (±{stats['std_rating']:.2f})")
        print(f"Puan Aralığı: [{stats['min_rating']}, {stats['max_rating']}]")
        print(f"\nKullanıcı Başına Ortalama Puanlama: {stats['avg_ratings_per_user']:.1f}")
        print(f"Kullanıcı Başına Min/Max Puanlama: {stats['min_ratings_per_user']}/{stats['max_ratings_per_user']}")
        print(f"\nFilm Başına Ortalama Puanlama: {stats['avg_ratings_per_movie']:.1f}")
        print(f"Film Başına Min/Max Puanlama: {stats['min_ratings_per_movie']}/{stats['max_ratings_per_movie']}")
        print("=" * 60)
    
    def get_movie_title(self, movie_id):
     
        if self.movies_df is not None:
            movie = self.movies_df[self.movies_df['movie_id'] == movie_id]
            if not movie.empty:
                return movie.iloc[0]['title']
        return f"Film {movie_id}"
    
    def prepare_data(self):
 
        print("\n" + "=" * 60)
        print("VERİ HAZIRLAMA BAŞLIYOR")
        print("=" * 60)
        
        self.load_ratings()
        self.load_movies()
        self.load_users()
        self.create_id_mappings()
        self.split_data()
        self.print_statistics()
        
        print("\n✓ Veri hazırlama tamamlandı!")
        print("=" * 60)


if __name__ == "__main__":
   
    print("Data Loader Test Ediliyor...")
    
    data_loader = DataLoader_ML(data_dir="data/raw/ml-100k/")
    data_loader.prepare_data()
  
    train_loader, val_loader, test_loader = data_loader.get_data_loaders(batch_size=32)
    
    for user_ids, movie_ids, ratings in train_loader:
        print(f"\nBatch şekli:")
        print(f"User IDs: {user_ids.shape}")
        print(f"Movie IDs: {movie_ids.shape}")
        print(f"Ratings: {ratings.shape}")
        print(f"\nÖrnek veriler:")
        print(f"Users: {user_ids[:5]}")
        print(f"Movies: {movie_ids[:5]}")
        print(f"Ratings: {ratings[:5]}")
        break
