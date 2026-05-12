"""
Toplu Film Önerisi Scripti
Tuğrul Şahar (233255027) - Burak Yetişer (233255007)

Bu script birden fazla kullanıcı için toplu film önerileri oluşturur.
"""

import torch
import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm
from datetime import datetime

from config import config
from ncf_model import NCFModel
from utils.data_loader import DataLoader_ML


class BatchRecommender:
    """
    Toplu öneri sistemi
    """
    
    def __init__(self, model_path, data_loader):
        """
        Args:
            model_path (str): Model checkpoint yolu
            data_loader (DataLoader_ML): Veri yükleyici
        """
        self.data_loader = data_loader
        self.device = config.DEVICE
        
        # Modeli yükle
        print(f"Model yükleniyor: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device,weights_only=False)
        
        model_config = checkpoint['config']
        
        self.model = NCFModel(
            num_users=model_config['num_users'],
            num_movies=model_config['num_movies'],
            embedding_dim=model_config['embedding_dim'],
            hidden_layers=model_config['hidden_layers'],
            dropout_rate=model_config['dropout_rate']
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"✓ Model yüklendi!")
    
    def generate_recommendations_for_users(self, user_ids, top_k=10, 
                                          exclude_rated=True):
        """
        Birden fazla kullanıcı için öneriler oluştur
        
        Args:
            user_ids (list): Kullanıcı ID listesi
            top_k (int): Her kullanıcı için öneri sayısı
            exclude_rated (bool): Puanlanan filmleri çıkar
            
        Returns:
            dict: Kullanıcı ID -> öneriler mapping'i
        """
        all_recommendations = {}
        all_movie_ids = list(self.data_loader.movie_id_map.keys())
        
        print(f"\n{len(user_ids)} kullanıcı için öneriler oluşturuluyor...")
        
        for user_id in tqdm(user_ids, desc="Öneriler oluşturuluyor"):
            # Kullanıcı index'ini al
            user_idx = self.data_loader.user_id_map.get(user_id)
            
            if user_idx is None:
                continue
            
            # Puanlanan filmleri bul
            if exclude_rated:
                user_ratings = self.data_loader.ratings_df[
                    self.data_loader.ratings_df['user_id'] == user_id
                ]
                rated_movie_ids = set(user_ratings['movie_id'].values)
                candidate_movie_ids = [mid for mid in all_movie_ids 
                                      if mid not in rated_movie_ids]
            else:
                candidate_movie_ids = all_movie_ids
            
            if len(candidate_movie_ids) == 0:
                continue
            
            # Tahminler yap (batch processing)
            batch_size = 1000
            all_predictions = []
            all_candidate_ids = []
            
            with torch.no_grad():
                for i in range(0, len(candidate_movie_ids), batch_size):
                    batch_movie_ids = candidate_movie_ids[i:i+batch_size]
                    batch_movie_indices = [self.data_loader.movie_id_map[mid] 
                                          for mid in batch_movie_ids]
                    
                    user_tensor = torch.LongTensor([user_idx] * len(batch_movie_indices)).to(self.device)
                    movie_tensor = torch.LongTensor(batch_movie_indices).to(self.device)
                    
                    batch_predictions = self.model(user_tensor, movie_tensor)
                    
                    all_predictions.extend(batch_predictions.cpu().numpy())
                    all_candidate_ids.extend(batch_movie_ids)
            
            # En yüksek puanlı filmleri seç
            top_indices = np.argsort(all_predictions)[-top_k:][::-1]
            
            recommendations = []
            for idx in top_indices:
                movie_id = all_candidate_ids[idx]
                score = all_predictions[idx]
                movie_title = self.data_loader.get_movie_title(movie_id)
                
                recommendations.append({
                    'movie_id': movie_id,
                    'title': movie_title,
                    'predicted_score': float(score)
                })
            
            all_recommendations[user_id] = recommendations
        
        print(f"✓ {len(all_recommendations)} kullanıcı için öneriler oluşturuldu!")
        
        return all_recommendations
    
    def generate_recommendations_for_all_users(self, top_k=10):
        """
        Tüm kullanıcılar için öneriler oluştur
        
        Args:
            top_k (int): Her kullanıcı için öneri sayısı
            
        Returns:
            dict: Tüm kullanıcı önerileri
        """
        all_user_ids = self.data_loader.ratings_df['user_id'].unique()
        return self.generate_recommendations_for_users(all_user_ids, top_k=top_k)
    
    def save_recommendations_to_csv(self, recommendations, output_file):
        """
        Önerileri CSV dosyasına kaydet
        
        Args:
            recommendations (dict): Öneriler dictionary
            output_file (str): Çıktı dosya yolu
        """
        rows = []
        
        for user_id, recs in recommendations.items():
            for rank, rec in enumerate(recs, 1):
                rows.append({
                    'user_id': user_id,
                    'rank': rank,
                    'movie_id': rec['movie_id'],
                    'title': rec['title'],
                    'predicted_score': rec['predicted_score']
                })
        
        df = pd.DataFrame(rows)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_csv(output_file, index=False)
        
        print(f"✓ Öneriler kaydedildi: {output_file}")
        print(f"  Toplam satır: {len(df)}")
    
    def save_recommendations_to_json(self, recommendations, output_file):
        """
        Önerileri JSON dosyasına kaydet
        
        Args:
            recommendations (dict): Öneriler dictionary
            output_file (str): Çıktı dosya yolu
        """
        import json
        """
        # JSON'a dönüştürülebilir hale getir
        json_recommendations = {
            str(user_id): recs 
            for user_id, recs in recommendations.items()
        }
        """
        json_recommendations = {
        str(user_id): [
         {
            "movie_id": int(rec["movie_id"]),
            "title": rec["title"],
           # "score": float(rec["score"])
           "score": float(rec["predicted_score"])
         }
         for rec in recs
        ]
        for user_id, recs in recommendations.items()
        }
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_recommendations, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Öneriler kaydedildi: {output_file}")
    
    def generate_user_similarity_matrix(self, sample_size=100):
        """
        Kullanıcı benzerlik matrisi oluştur (embedding tabanlı)
        
        Args:
            sample_size (int): Örneklem boyutu
            
        Returns:
            DataFrame: Benzerlik matrisi
        """
        # Rastgele kullanıcılar seç
        all_users = list(self.data_loader.user_id_map.keys())
        sample_users = np.random.choice(all_users, 
                                       min(sample_size, len(all_users)), 
                                       replace=False)
        
        # Embedding'leri al
        user_embeddings = []
        user_ids_sampled = []
        
        for user_id in sample_users:
            user_idx = self.data_loader.user_id_map[user_id]
            embedding = self.model.user_embedding(torch.tensor([user_idx])).detach().cpu().numpy()
            user_embeddings.append(embedding[0])
            user_ids_sampled.append(user_id)
        
        user_embeddings = np.array(user_embeddings)
        
        # Cosine similarity hesapla
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(user_embeddings)
        
        # DataFrame'e dönüştür
        similarity_df = pd.DataFrame(
            similarity_matrix,
            index=user_ids_sampled,
            columns=user_ids_sampled
        )
        
        return similarity_df
    
    def find_similar_users(self, user_id, top_k=5):
        """
        Benzer kullanıcıları bul
        
        Args:
            user_id (int): Kullanıcı ID
            top_k (int): En benzer kullanıcı sayısı
            
        Returns:
            list: Benzer kullanıcılar ve benzerlik skorları
        """
        user_idx = self.data_loader.user_id_map.get(user_id)
        
        if user_idx is None:
            return []
        
        # Bu kullanıcının embedding'ini al
        user_embedding = self.model.user_embedding(
            torch.tensor([user_idx])
        ).detach().cpu().numpy()
        
        # Tüm kullanıcı embedding'lerini al
        all_embeddings = self.model.user_embedding.weight.data.cpu().numpy()
        
        # Cosine similarity hesapla
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(user_embedding, all_embeddings)[0]
        
        # En benzer kullanıcıları bul (kendisi hariç)
        top_indices = np.argsort(similarities)[-top_k-1:-1][::-1]
        
        similar_users = []
        for idx in top_indices:
            if idx == user_idx:
                continue
            
            original_user_id = self.data_loader.reverse_user_map[idx]
            similarity_score = similarities[idx]
            
            similar_users.append({
                'user_id': original_user_id,
                'similarity': float(similarity_score)
            })
        
        return similar_users


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description='Toplu Film Önerisi')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                        help='Model checkpoint yolu')
    parser.add_argument('--data_dir', type=str, default='data/raw/ml-100k/',
                        help='Veri dizini')
    parser.add_argument('--mode', type=str, default='all', 
                        choices=['all', 'sample', 'specific'],
                        help='Öneri modu')
    parser.add_argument('--user_ids', type=str, default=None,
                        help='Virgülle ayrılmış kullanıcı ID listesi (specific mode için)')
    parser.add_argument('--sample_size', type=int, default=100,
                        help='Sample mode için kullanıcı sayısı')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Her kullanıcı için öneri sayısı')
    parser.add_argument('--output_csv', type=str, 
                        default='results/recommendations/batch_recommendations.csv',
                        help='CSV çıktı yolu')
    parser.add_argument('--output_json', type=str,
                        default='results/recommendations/batch_recommendations.json',
                        help='JSON çıktı yolu')
    
    args = parser.parse_args()
    
    # Veriyi yükle
    print("Veri yükleniyor...")
    data_loader = DataLoader_ML(data_dir=args.data_dir)
    data_loader.prepare_data()
    
    # Recommender oluştur
    recommender = BatchRecommender(args.model, data_loader)
    
    # Kullanıcıları belirle
    if args.mode == 'all':
        print("\nTüm kullanıcılar için öneriler oluşturuluyor...")
        recommendations = recommender.generate_recommendations_for_all_users(top_k=args.top_k)
    
    elif args.mode == 'sample':
        print(f"\n{args.sample_size} rastgele kullanıcı için öneriler oluşturuluyor...")
        all_users = data_loader.ratings_df['user_id'].unique()
        sample_users = np.random.choice(all_users, args.sample_size, replace=False)
        recommendations = recommender.generate_recommendations_for_users(
            sample_users, top_k=args.top_k
        )
    
    elif args.mode == 'specific':
        if args.user_ids is None:
            print("Hata: --user_ids parametresi gerekli!")
            return
        
        user_ids = [int(uid) for uid in args.user_ids.split(',')]
        print(f"\n{len(user_ids)} kullanıcı için öneriler oluşturuluyor...")
        recommendations = recommender.generate_recommendations_for_users(
            user_ids, top_k=args.top_k
        )
    
    # Sonuçları kaydet
    if args.output_csv:
        recommender.save_recommendations_to_csv(recommendations, args.output_csv)
    
    if args.output_json:
        recommender.save_recommendations_to_json(recommendations, args.output_json)
    
    print("\n✓ Toplu öneri işlemi tamamlandı!")
    
    # Örnek öneri göster
    sample_user = list(recommendations.keys())[0]
    print(f"\nÖrnek: Kullanıcı {sample_user} için öneriler:")
    for i, rec in enumerate(recommendations[sample_user][:5], 1):
        print(f"  {i}. {rec['title']}: {rec['predicted_score']:.2f}★")


if __name__ == "__main__":
    main()
