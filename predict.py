
import torch
import pandas as pd
import argparse
import numpy as np
from tabulate import tabulate

from config import config
from ncf_model import NCFModel
from utils.data_loader import DataLoader_ML


class MovieRecommender:

    def __init__(self, model_path, data_loader):
        self.data_loader = data_loader
        self.device = config.DEVICE

        print(f"Model yükleniyor: {model_path}")

        # 🔥 KRİTİK FIX: safe loading
        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False
        )

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

        print("✓ Model başarıyla yüklendi")

    def predict_rating(self, user_id, movie_id):

        user_idx = self.data_loader.user_id_map.get(user_id)
        movie_idx = self.data_loader.movie_id_map.get(movie_id)

        if user_idx is None or movie_idx is None:
            return None

        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx]).to(self.device)
            movie_tensor = torch.LongTensor([movie_idx]).to(self.device)

            pred = self.model(user_tensor, movie_tensor)

        return pred.item()

    def recommend_movies(self, user_id, top_k=10):

        user_idx = self.data_loader.user_id_map.get(user_id)

        if user_idx is None:
            print("Kullanıcı bulunamadı!")
            return None

        rated = self.data_loader.ratings_df[
            self.data_loader.ratings_df['user_id'] == user_id
        ]['movie_id'].values

        rated_set = set(rated)

        candidates = [
            mid for mid in self.data_loader.movie_id_map.keys()
            if mid not in rated_set
        ]

        results = []

        with torch.no_grad():
            for mid in candidates:

                m_idx = self.data_loader.movie_id_map[mid]

                user_tensor = torch.LongTensor([user_idx]).to(self.device)
                movie_tensor = torch.LongTensor([m_idx]).to(self.device)

                pred = self.model(user_tensor, movie_tensor).item()

                results.append((mid, pred))

        results.sort(key=lambda x: x[1], reverse=True)
        top = results[:top_k]

        df = pd.DataFrame(top, columns=["movie_id", "score"])

        if self.data_loader.movies_df is not None:
            df = df.merge(self.data_loader.movies_df, on="movie_id", how="left")

        return df

    def print_recommendations(self, user_id, top_k=10):

        print("\n" + "=" * 60)
        print(f"KULLANICI {user_id} ÖNERİLERİ")
        print("=" * 60)

        recs = self.recommend_movies(user_id, top_k)

        if recs is None:
            print("Öneri yok")
            return

        table = []
        for i, row in recs.iterrows():
            table.append([
                i + 1,
                row['movie_id'],
                row.get('title', 'Bilinmiyor'),
                f"{row['score']:.3f}"
            ])

        print(tabulate(table, headers=["#", "ID", "Film", "Skor"], tablefmt="grid"))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best_model.pth")
    parser.add_argument("--data_dir", default="data/raw/ml-100k/")
    parser.add_argument("--user_id", type=int, required=True)
    parser.add_argument("--top_k", type=int, default=10)

    args = parser.parse_args()

    print("Veri yükleniyor...")

    data_loader = DataLoader_ML(data_dir=args.data_dir)
    data_loader.prepare_data()

    recommender = MovieRecommender(args.model, data_loader)

    recommender.print_recommendations(args.user_id, args.top_k)


if __name__ == "__main__":
    main()