"""
Neural Collaborative Filtering (NCF) Model Mimarisi


Bu modül Neural Collaborative Filtering modelini içerir.
Model, kullanıcı ve film embeddinglərini birleştirerek puan tahmini yapar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NCFModel(nn.Module):
    """
    Neural Collaborative Filtering Modeli
    
    Bu model iki ana bileşenden oluşur:
    1. Embedding Katmanları: Kullanıcı ve filmleri vektörel uzayda temsil eder
    2. Fully Connected Katmanlar: Non-linear etkileşimleri öğrenir
    
    Args:
        num_users (int): Toplam kullanıcı sayısı
        num_movies (int): Toplam film sayısı
        embedding_dim (int): Embedding vektör boyutu
        hidden_layers (list): Gizli katmanların boyutları
        dropout_rate (float): Dropout oranı
    """
    
    def __init__(self, num_users, num_movies, embedding_dim=64, 
                 hidden_layers=[128, 64, 32], dropout_rate=0.2):
        super(NCFModel, self).__init__()
        
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_dim = embedding_dim
        
        # Embedding katmanları
        # Her kullanıcı ve film için öğrenilebilir vektörler
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # Embedding başlatma (Xavier initialization)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.movie_embedding.weight)
        
        # Fully Connected katmanlar
        layers = []
        input_dim = embedding_dim * 2  # User + Movie embeddings birleştirilir
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        
        # Output katmanı - Puan tahmini (1-5 arası)
        layers.append(nn.Linear(input_dim, 1))
        
        self.fc_layers = nn.Sequential(*layers)
        
        # Output normalizasyonu için sigmoid
        self.output_range = (1.0, 5.0)
    
    def forward(self, user_ids, movie_ids):
        """
        Forward pass
        
        Args:
            user_ids (Tensor): Kullanıcı ID'leri [batch_size]
            movie_ids (Tensor): Film ID'leri [batch_size]
            
        Returns:
            Tensor: Tahmin edilen puanlar [batch_size]
        """
        # Embedding'leri al
        user_embedded = self.user_embedding(user_ids)  # [batch_size, embedding_dim]
        movie_embedded = self.movie_embedding(movie_ids)  # [batch_size, embedding_dim]
        
        # Embedding'leri birleştir
        concatenated = torch.cat([user_embedded, movie_embedded], dim=1)  # [batch_size, embedding_dim*2]
        
        # Fully connected katmanlardan geçir
        output = self.fc_layers(concatenated)  # [batch_size, 1]
        
        # Output'u squeeze et ve 1-5 aralığına getir
        output = output.squeeze()  # [batch_size]
        
        # Sigmoid ile 0-1 arasına getir, sonra 1-5 arasına scale et
        output = torch.sigmoid(output)
        output = output * (self.output_range[1] - self.output_range[0]) + self.output_range[0]
        
        return output
    
    def get_embedding(self, user_id=None, movie_id=None):
        """
        Belirli bir kullanıcı veya film için embedding vektörünü döndür
        
        Args:
            user_id (int, optional): Kullanıcı ID
            movie_id (int, optional): Film ID
            
        Returns:
            Tensor: Embedding vektörü
        """
        if user_id is not None:
            return self.user_embedding(torch.tensor([user_id]))
        if movie_id is not None:
            return self.movie_embedding(torch.tensor([movie_id]))
        return None
    
    def predict_for_user(self, user_id, movie_ids):
        """
        Bir kullanıcı için birden fazla film tahmini yap
        
        Args:
            user_id (int): Kullanıcı ID
            movie_ids (list or Tensor): Film ID'leri listesi
            
        Returns:
            Tensor: Tahmin edilen puanlar
        """
        self.eval()
        with torch.no_grad():
            if isinstance(movie_ids, list):
                movie_ids = torch.tensor(movie_ids)
            
            user_ids = torch.tensor([user_id] * len(movie_ids))
            predictions = self.forward(user_ids, movie_ids)
            
        return predictions
    
    def recommend_top_k(self, user_id, all_movie_ids, k=10, exclude_movies=None):
        """
        Bir kullanıcı için top-k film önerisi yap
        
        Args:
            user_id (int): Kullanıcı ID
            all_movie_ids (list): Tüm film ID'leri
            k (int): Önerilecek film sayısı
            exclude_movies (list, optional): Dışarıda bırakılacak filmler (zaten izlenenler)
            
        Returns:
            tuple: (önerilen_film_ids, tahmin_edilen_puanlar)
        """
        # Tüm filmler için tahmin yap
        predictions = self.predict_for_user(user_id, all_movie_ids)
        
        # Zaten izlenen filmleri çıkar
        if exclude_movies:
            mask = torch.ones(len(all_movie_ids), dtype=torch.bool)
            for movie_id in exclude_movies:
                if movie_id in all_movie_ids:
                    idx = all_movie_ids.index(movie_id)
                    mask[idx] = False
            
            predictions = predictions[mask]
            filtered_movie_ids = [m for i, m in enumerate(all_movie_ids) if mask[i]]
        else:
            filtered_movie_ids = all_movie_ids
        
        # En yüksek puanlı k filmi seç
        top_k_values, top_k_indices = torch.topk(predictions, k)
        
        recommended_movies = [filtered_movie_ids[i] for i in top_k_indices.tolist()]
        
        return recommended_movies, top_k_values.tolist()
    
    def count_parameters(self):
        """Model parametrelerini say"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self):
        """Model bilgilerini yazdır"""
        model_info = f"NCF Model\n"
        model_info += f"- Kullanıcı Sayısı: {self.num_users}\n"
        model_info += f"- Film Sayısı: {self.num_movies}\n"
        model_info += f"- Embedding Boyutu: {self.embedding_dim}\n"
        model_info += f"- Toplam Parametre Sayısı: {self.count_parameters():,}\n"
        return model_info


class MatrixFactorization(nn.Module):
    """
    Basit Matrix Factorization Modeli (Karşılaştırma için)
    
    Bu model daha basit bir yaklaşımdır ve sadece embedding'lerin
    dot product'ını kullanır.
    """
    
    def __init__(self, num_users, num_movies, embedding_dim=64):
        super(MatrixFactorization, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # Bias terimleri
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Başlatma
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.movie_embedding.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)
    
    def forward(self, user_ids, movie_ids):
        """Forward pass"""
        user_embedded = self.user_embedding(user_ids)
        movie_embedded = self.movie_embedding(movie_ids)
        
        # Dot product
        dot_product = (user_embedded * movie_embedded).sum(dim=1)
        
        # Bias'ları ekle
        user_bias = self.user_bias(user_ids).squeeze()
        movie_bias = self.movie_bias(movie_ids).squeeze()
        
        # Toplam tahmin
        prediction = dot_product + user_bias + movie_bias + self.global_bias
        
        # 1-5 aralığına sınırla
        prediction = torch.clamp(prediction, 1.0, 5.0)
        
        return prediction


if __name__ == "__main__":
    # Test
    print("NCF Model Test Ediliyor...")
    
    num_users = 1000
    num_movies = 500
    
    model = NCFModel(num_users, num_movies)
    print(model)
    
    # Dummy veri
    user_ids = torch.randint(0, num_users, (32,))
    movie_ids = torch.randint(0, num_movies, (32,))
    
    predictions = model(user_ids, movie_ids)
    print(f"\nÖrnek tahminler: {predictions[:5]}")
    print(f"Tahmin aralığı: [{predictions.min():.2f}, {predictions.max():.2f}]")
