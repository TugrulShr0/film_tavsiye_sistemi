"""
Film Tavsiye Sistemi - Demo Script
Tuğrul Şahar (233255027) - Burak Yetişer (233255007)

Bu script projeyi hızlıca test etmek için kullanılır.
"""

import torch
import numpy as np
from ncf_model import NCFModel, MatrixFactorization


def test_model_architecture():
    """Model mimarisini test et"""
    print("\n" + "=" * 70)
    print("MODEL MİMARİSİ TESTİ")
    print("=" * 70)
    
    # Test parametreleri
    num_users = 100
    num_movies = 50
    batch_size = 16
    
    # NCF modeli oluştur
    print("\n1. NCF Model:")
    ncf_model = NCFModel(
        num_users=num_users,
        num_movies=num_movies,
        embedding_dim=32,
        hidden_layers=[64, 32, 16],
        dropout_rate=0.2
    )
    print(ncf_model)
    
    # Matrix Factorization modeli oluştur
    print("\n2. Matrix Factorization Model:")
    mf_model = MatrixFactorization(
        num_users=num_users,
        num_movies=num_movies,
        embedding_dim=32
    )
    print(f"MF Model - Parametre Sayısı: {sum(p.numel() for p in mf_model.parameters()):,}")
    
    # Dummy veri oluştur
    user_ids = torch.randint(0, num_users, (batch_size,))
    movie_ids = torch.randint(0, num_movies, (batch_size,))
    
    # Forward pass test
    print("\n3. Forward Pass Testi:")
    with torch.no_grad():
        ncf_predictions = ncf_model(user_ids, movie_ids)
        mf_predictions = mf_model(user_ids, movie_ids)
    
    print(f"NCF Tahminler: {ncf_predictions[:5]}")
    print(f"NCF Tahmin Aralığı: [{ncf_predictions.min():.2f}, {ncf_predictions.max():.2f}]")
    print(f"\nMF Tahminler: {mf_predictions[:5]}")
    print(f"MF Tahmin Aralığı: [{mf_predictions.min():.2f}, {mf_predictions.max():.2f}]")
    
    # Embedding test
    print("\n4. Embedding Testi:")
    user_embedding = ncf_model.get_embedding(user_id=0)
    movie_embedding = ncf_model.get_embedding(movie_id=0)
    print(f"User Embedding Shape: {user_embedding.shape}")
    print(f"Movie Embedding Shape: {movie_embedding.shape}")
    
    # Öneri testi
    print("\n5. Öneri Fonksiyonu Testi:")
    test_user_id = 5
    all_movie_ids = list(range(num_movies))
    
    recommended_movies, scores = ncf_model.recommend_top_k(
        user_id=test_user_id,
        all_movie_ids=all_movie_ids,
        k=5
    )
    
    print(f"Kullanıcı {test_user_id} için Top-5 Öneriler:")
    for i, (movie_id, score) in enumerate(zip(recommended_movies, scores), 1):
        print(f"  {i}. Film {movie_id}: {score:.2f}★")
    
    print("\n✓ Model testleri başarılı!")


def test_training_loop():
    """Basit bir eğitim döngüsü test et"""
    print("\n" + "=" * 70)
    print("EĞİTİM DÖNGÜSÜ TESTİ (Küçük Veri)")
    print("=" * 70)
    
    # Küçük bir veri seti oluştur
    num_users = 50
    num_movies = 30
    num_samples = 500
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Sentetik veri
    user_ids = torch.randint(0, num_users, (num_samples,))
    movie_ids = torch.randint(0, num_movies, (num_samples,))
    ratings = torch.rand(num_samples) * 4 + 1  # 1-5 arası
    
    # Model
    model = NCFModel(
        num_users=num_users,
        num_movies=num_movies,
        embedding_dim=16,
        hidden_layers=[32, 16],
        dropout_rate=0.1
    )
    
    # Loss ve optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Mini eğitim
    print("\nModel eğitiliyor (5 epoch)...")
    model.train()
    
    batch_size = 64
    num_epochs = 5
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, num_samples, batch_size):
            batch_users = user_ids[i:i+batch_size]
            batch_movies = movie_ids[i:i+batch_size]
            batch_ratings = ratings[i:i+batch_size]
            
            # Forward pass
            predictions = model(batch_users, batch_movies)
            loss = criterion(predictions, batch_ratings)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/5 - Loss: {avg_loss:.4f}")
    
    # Test tahmini
    print("\nTest Tahmini:")
    model.eval()
    with torch.no_grad():
        test_user = torch.tensor([0])
        test_movie = torch.tensor([5])
        prediction = model(test_user, test_movie)
        print(f"User 0, Movie 5 için tahmin: {prediction.item():.2f}★")
    
    print("\n✓ Eğitim döngüsü testi başarılı!")


def test_recommendation_quality():
    """Öneri kalitesini test et"""
    print("\n" + "=" * 70)
    print("ÖNERİ KALİTESİ TESTİ")
    print("=" * 70)
    
    # Basit bir senaryo: Kullanıcılar belirli film türlerini seviyor
    num_users = 20
    num_movies = 40
    
    # Her kullanıcıya bir "profil" ata
    # İlk 10 film "aksiyon", son 10 film "komedi", ortadakiler "karma"
    
    np.random.seed(42)
    
    # Sentetik puanlamalar oluştur
    user_ids_list = []
    movie_ids_list = []
    ratings_list = []
    
    for user in range(num_users):
        # Her kullanıcı rastgele 15-25 film puanlasın
        num_ratings = np.random.randint(15, 26)
        
        for _ in range(num_ratings):
            movie = np.random.randint(0, num_movies)
            
            # İlk 10 kullanıcı aksiyon sever (film 0-9)
            # Son 10 kullanıcı komedi sever (film 30-39)
            if user < 10:  # Aksiyon seven
                if movie < 10:  # Aksiyon filmi
                    rating = np.random.uniform(3.5, 5.0)
                elif movie >= 30:  # Komedi filmi
                    rating = np.random.uniform(1.0, 2.5)
                else:
                    rating = np.random.uniform(2.0, 4.0)
            else:  # Komedi seven
                if movie < 10:  # Aksiyon filmi
                    rating = np.random.uniform(1.0, 2.5)
                elif movie >= 30:  # Komedi filmi
                    rating = np.random.uniform(3.5, 5.0)
                else:
                    rating = np.random.uniform(2.0, 4.0)
            
            user_ids_list.append(user)
            movie_ids_list.append(movie)
            ratings_list.append(rating)
    
    # Tensor'lere dönüştür
    user_ids = torch.LongTensor(user_ids_list)
    movie_ids = torch.LongTensor(movie_ids_list)
    ratings = torch.FloatTensor(ratings_list)
    
    print(f"Toplam {len(ratings)} puanlama oluşturuldu")
    
    # Model eğit
    model = NCFModel(
        num_users=num_users,
        num_movies=num_movies,
        embedding_dim=16,
        hidden_layers=[32, 16],
        dropout_rate=0.1
    )
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("\nModel eğitiliyor...")
    model.train()
    
    for epoch in range(20):
        optimizer.zero_grad()
        predictions = model(user_ids, movie_ids)
        loss = criterion(predictions, ratings)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/20 - Loss: {loss.item():.4f}")
    
    # Öneri testi
    print("\nÖneri Testi:")
    model.eval()
    
    # Aksiyon seven bir kullanıcı (user 0)
    print("\nAKSİYON SEVEN KULLANICI (User 0):")
    action_recs, action_scores = model.recommend_top_k(
        user_id=0,
        all_movie_ids=list(range(num_movies)),
        k=5
    )
    
    action_count = sum(1 for m in action_recs if m < 10)
    print(f"Top-5 öneride {action_count} aksiyon filmi var (beklenen: yüksek)")
    for i, (movie, score) in enumerate(zip(action_recs, action_scores), 1):
        movie_type = "Aksiyon" if movie < 10 else ("Komedi" if movie >= 30 else "Karma")
        print(f"  {i}. Film {movie} ({movie_type}): {score:.2f}★")
    
    # Komedi seven bir kullanıcı (user 15)
    print("\nKOMEDİ SEVEN KULLANICI (User 15):")
    comedy_recs, comedy_scores = model.recommend_top_k(
        user_id=15,
        all_movie_ids=list(range(num_movies)),
        k=5
    )
    
    comedy_count = sum(1 for m in comedy_recs if m >= 30)
    print(f"Top-5 öneride {comedy_count} komedi filmi var (beklenen: yüksek)")
    for i, (movie, score) in enumerate(zip(comedy_recs, comedy_scores), 1):
        movie_type = "Aksiyon" if movie < 10 else ("Komedi" if movie >= 30 else "Karma")
        print(f"  {i}. Film {movie} ({movie_type}): {score:.2f}★")
    
    print("\n✓ Öneri kalitesi testi tamamlandı!")


def main():
    """Ana test fonksiyonu"""
    print("\n")
    print("█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  FİLM TAVSİYE SİSTEMİ - DEMO VE TEST".center(68) + "█")
    print("█" + "  Tuğrul Şahar (233255027) - Burak Yetişer (233255007)".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    try:
        # Test 1: Model mimarisi
        test_model_architecture()
        
        # Test 2: Eğitim döngüsü
        test_training_loop()
        
        # Test 3: Öneri kalitesi
        test_recommendation_quality()
        
        print("\n" + "=" * 70)
        print("TÜM TESTLER BAŞARILI! ✓")
        print("=" * 70)
        print("\nProje başlatma adımları:")
        print("1. Veri setini indirin: python download_data.py")
        print("2. Modeli eğitin: python train.py --epochs 50")
        print("3. Öneri alın: python predict.py --user_id 1 --top_k 10")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Test hatası: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
