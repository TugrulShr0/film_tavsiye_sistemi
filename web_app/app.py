import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from flask import Flask, render_template, request, jsonify
import torch
import os
import json
from ncf_model import NCFModel
from utils.data_loader import DataLoader_ML
from config import config

app = Flask(__name__)

model = None
data_loader = None
model_loaded = False


def load_model_and_data():
    global model, data_loader, model_loaded
    
    if model_loaded:
        return True
    
    try:
        print("Model ve veri yükleniyor...")
        
        
        data_loader = DataLoader_ML(data_dir='data/raw/ml-100k/')
        data_loader.prepare_data()
        
        
        model_path = 'models/best_model.pth'
        if not os.path.exists(model_path):
            print(f"Model bulunamadı: {model_path}")
            return False
        
        checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
        model_config = checkpoint['config']
        
        model = NCFModel(
            num_users=model_config['num_users'],
            num_movies=model_config['num_movies'],
            embedding_dim=model_config['embedding_dim'],
            hidden_layers=model_config['hidden_layers'],
            dropout_rate=model_config['dropout_rate']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(config.DEVICE)
        model.eval()
        
        model_loaded = True
        print("✓ Model ve veri yüklendi!")
        return True
        
    except Exception as e:
        print(f"Yükleme hatası: {e}")
        return False


@app.route('/')
def index():
    """Ana sayfa - index.html şablonunun beklediği verileri gönderir"""
    if not model_loaded:
        load_model_and_data()
        
    # index.html'deki Jinja şablon değişkenlerini besliyoruz
    return render_template(
        'index.html',
        total_users=int(data_loader.num_users) if data_loader else 943,
        total_movies=int(data_loader.num_movies) if data_loader else 1682,
      
    )


@app.route('/recommend', methods=['POST'])
def recommend():
    """index.html'den gelen POST isteğini karşılar ve önerileri döner"""
    if not model_loaded:
        if not load_model_and_data():
            return jsonify({'error': 'Model yüklenemedi'}), 500
            
    try:
        data = request.get_json()
        if not data or 'user_id' not in data:
            return jsonify({'error': 'Geçersiz istek. user_id gerekli.'}), 400
            
        user_id = int(data.get('user_id'))
        top_k = int(data.get('top_k', 10))
        
       
        if user_id not in data_loader.user_id_map:
            return jsonify({'error': f'Kullanıcı ID bulunamadı (1-{data_loader.num_users})'}), 404
            
       
        user_idx = data_loader.user_id_map[user_id]
        
      
        user_ratings = data_loader.ratings_df[
            data_loader.ratings_df['user_id'] == user_id
        ].nlargest(5, 'rating')
        
        history = []
        for _, row in user_ratings.iterrows():
            history.append({
                'movie_id': int(row['movie_id']),
                'title': data_loader.get_movie_title(row['movie_id']),
                'rating': float(row['rating'])
            })
            
    
        all_movie_indices = list(data_loader.movie_id_map.values())
        
   
        rated_movie_ids = data_loader.ratings_df[
            data_loader.ratings_df['user_id'] == user_id
        ]['movie_id'].values
        
        rated_movie_indices = [
            data_loader.movie_id_map[mid] 
            for mid in rated_movie_ids 
            if mid in data_loader.movie_id_map
        ]
    
        recommended_movie_indices, scores = model.recommend_top_k(
            user_id=user_idx,
            all_movie_ids=all_movie_indices,
            k=top_k,
            exclude_movies=rated_movie_indices
        )
        
        
        recommendations = []
        for rank, (movie_idx, score) in enumerate(zip(recommended_movie_indices, scores), 1):
          
            if movie_idx in data_loader.reverse_movie_map:
                original_movie_id = data_loader.reverse_movie_map[movie_idx]
            else:
                original_movie_id = movie_idx
                
            recommendations.append({
                'rank': rank,
                'movie_id': int(original_movie_id),
                'title': data_loader.get_movie_title(original_movie_id),
                'predicted_score': float(score)
            })
            
        return jsonify({
            'user_id': user_id,
            'history': history,
            'recommendations': recommendations
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f"Model/Eşleştirme Hatası: {str(e)}"}), 500



@app.route('/api/stats')
def get_stats():
    if not model_loaded:
        if not load_model_and_data():
            return jsonify({'error': 'Model yüklenemedi'}), 500
    return jsonify({
        'total_users': int(data_loader.num_users),
        'total_movies': int(data_loader.num_movies),
        'total_ratings': int(data_loader.num_ratings),
        'model_loaded': model_loaded
    })


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("FİLM TAVSİYE SİSTEMİ - WEB UYGULAMASI")
    print("=" * 70)
    print("\nUygulama başlatılıyor...")
    
    if load_model_and_data():
        print("\n✓ Hazır!")
        print("\nTarayıcınızda açın: http://127.0.0.1:5000")
        print("\nKapatmak için: CTRL+C")
        print("=" * 70 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n✗ Model yüklenemedi!")
        print("Lütfen önce modeli eğitin: python train.py --epochs 50")