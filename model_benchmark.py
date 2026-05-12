"""
Model Karşılaştırma ve Benchmark

Bu script farklı modelleri karşılaştırır ve benchmark sonuçları oluşturur.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tabulate import tabulate

from config import config
from ncf_model import NCFModel, MatrixFactorization
from utils.data_loader import DataLoader_ML
from utils.evaluation import Evaluator


class ModelBenchmark:
    """
    Model karşılaştırma ve benchmark sınıfı
    """
    
    def __init__(self, data_loader, device):
        """
        Args:
            data_loader: Veri yükleyici
            device: Hesaplama cihazı
        """
        self.data_loader = data_loader
        self.device = device
        self.results = {}
    
    def train_model(self, model, model_name, train_loader, val_loader, 
                    epochs=50, learning_rate=0.001):
        """
        Tek bir modeli eğit ve değerlendir
        
        Args:
            model: PyTorch modeli
            model_name (str): Model adı
            train_loader: Eğitim veri yükleyici
            val_loader: Doğrulama veri yükleyici
            epochs (int): Epoch sayısı
            learning_rate (float): Öğrenme oranı
            
        Returns:
            dict: Model sonuçları
        """
        print(f"\n{'=' * 70}")
        print(f"MODELİ EĞİTİYOR: {model_name}")
        print(f"{'=' * 70}")
        
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        evaluator = Evaluator()
        
        training_time = 0
        best_val_loss = float('inf')
        
        # Eğitim
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training
            model.train()
            train_loss = 0.0
            
            for user_ids, movie_ids, ratings in train_loader:
                user_ids = user_ids.to(self.device)
                movie_ids = movie_ids.to(self.device)
                ratings = ratings.to(self.device)
                
                optimizer.zero_grad()
                predictions = model(user_ids, movie_ids)
                loss = criterion(predictions, ratings)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            val_metrics = evaluator.evaluate_model(model, val_loader, self.device, criterion)
            
            epoch_time = time.time() - epoch_start
            training_time += epoch_time
            
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_metrics = val_metrics
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | "
                      f"Val RMSE: {val_metrics['rmse']:.4f}")
        
        print(f"\n✓ Eğitim tamamlandı - Toplam süre: {training_time:.2f}s")
        
        # Model bilgileri
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'model': model,
            'best_val_loss': best_val_loss,
            'best_val_rmse': best_metrics['rmse'],
            'best_val_mae': best_metrics['mae'],
            'training_time': training_time,
            'avg_epoch_time': training_time / epochs,
            'num_parameters': num_params
        }
    
    def benchmark_all_models(self, epochs=50):
        """
        Tüm modelleri karşılaştır
        
        Args:
            epochs (int): Her model için epoch sayısı
        """
        print("\n" + "=" * 70)
        print("MODEL BENCHMARK BAŞLIYOR")
        print("=" * 70)
        
        # DataLoader'ları hazırla
        train_loader, val_loader, test_loader = self.data_loader.get_data_loaders(
            batch_size=256
        )
        
        # Model 1: NCF (Küçük)
        ncf_small = NCFModel(
            num_users=self.data_loader.num_users,
            num_movies=self.data_loader.num_movies,
            embedding_dim=32,
            hidden_layers=[64, 32],
            dropout_rate=0.2
        )
        self.results['NCF_Small'] = self.train_model(
            ncf_small, 'NCF (Small)', train_loader, val_loader, epochs
        )
        
        # Model 2: NCF (Orta)
        ncf_medium = NCFModel(
            num_users=self.data_loader.num_users,
            num_movies=self.data_loader.num_movies,
            embedding_dim=64,
            hidden_layers=[128, 64, 32],
            dropout_rate=0.2
        )
        self.results['NCF_Medium'] = self.train_model(
            ncf_medium, 'NCF (Medium)', train_loader, val_loader, epochs
        )
        
        # Model 3: NCF (Büyük)
        ncf_large = NCFModel(
            num_users=self.data_loader.num_users,
            num_movies=self.data_loader.num_movies,
            embedding_dim=128,
            hidden_layers=[256, 128, 64, 32],
            dropout_rate=0.2
        )
        self.results['NCF_Large'] = self.train_model(
            ncf_large, 'NCF (Large)', train_loader, val_loader, epochs
        )
        
        # Model 4: Matrix Factorization
        mf_model = MatrixFactorization(
            num_users=self.data_loader.num_users,
            num_movies=self.data_loader.num_movies,
            embedding_dim=64
        )
        self.results['Matrix_Factorization'] = self.train_model(
            mf_model, 'Matrix Factorization', train_loader, val_loader, epochs
        )
        
        # Test setinde değerlendirme
        print("\n" + "=" * 70)
        print("TEST SETİ DEĞERLENDİRMESİ")
        print("=" * 70)
        
        evaluator = Evaluator()
        
        for model_name, result in self.results.items():
            test_metrics = evaluator.evaluate_model(
                result['model'], test_loader, self.device, nn.MSELoss()
            )
            result['test_rmse'] = test_metrics['rmse']
            result['test_mae'] = test_metrics['mae']
            
            print(f"\n{model_name}:")
            print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
            print(f"  Test MAE: {test_metrics['mae']:.4f}")
    
    def print_comparison_table(self):
        """Karşılaştırma tablosu yazdır"""
        print("\n" + "=" * 100)
        print("MODEL KARŞILAŞTIRMA TABLOSU")
        print("=" * 100)
        
        table_data = []
        for model_name, result in self.results.items():
            table_data.append([
                model_name,
                f"{result['num_parameters']:,}",
                f"{result['best_val_rmse']:.4f}",
                f"{result['best_val_mae']:.4f}",
                f"{result['test_rmse']:.4f}",
                f"{result['test_mae']:.4f}",
                f"{result['training_time']:.1f}s",
                f"{result['avg_epoch_time']:.2f}s"
            ])
        
        headers = [
            'Model', 'Parametreler', 'Val RMSE', 'Val MAE', 
            'Test RMSE', 'Test MAE', 'Eğitim Süresi', 'Epoch Süresi'
        ]
        
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    def plot_comparison(self, save_path=None):
        """Karşılaştırma grafikleri oluştur"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        model_names = list(self.results.keys())
        val_rmse = [self.results[m]['best_val_rmse'] for m in model_names]
        test_rmse = [self.results[m]['test_rmse'] for m in model_names]
        test_mae = [self.results[m]['test_mae'] for m in model_names]
        num_params = [self.results[m]['num_parameters'] for m in model_names]
        training_time = [self.results[m]['training_time'] for m in model_names]
        
        # RMSE karşılaştırması
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, val_rmse, width, label='Validation RMSE', alpha=0.8)
        axes[0, 0].bar(x + width/2, test_rmse, width, label='Test RMSE', alpha=0.8)
        axes[0, 0].set_xlabel('Model')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].set_title('RMSE Karşılaştırması')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Parametre sayısı vs Performans
        axes[0, 1].scatter(num_params, test_rmse, s=200, alpha=0.6)
        for i, name in enumerate(model_names):
            axes[0, 1].annotate(name, (num_params[i], test_rmse[i]), 
                               fontsize=8, ha='right')
        axes[0, 1].set_xlabel('Parametre Sayısı')
        axes[0, 1].set_ylabel('Test RMSE')
        axes[0, 1].set_title('Model Karmaşıklığı vs Performans')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Eğitim süresi karşılaştırması
        axes[1, 0].barh(model_names, training_time, alpha=0.8)
        axes[1, 0].set_xlabel('Eğitim Süresi (saniye)')
        axes[1, 0].set_title('Eğitim Süresi Karşılaştırması')
        axes[1, 0].grid(True, alpha=0.3, axis='x')
        
        # MAE karşılaştırması
        axes[1, 1].bar(model_names, test_mae, alpha=0.8, color='coral')
        axes[1, 1].set_ylabel('Test MAE')
        axes[1, 1].set_title('MAE Karşılaştırması')
        axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Grafik kaydedildi: {save_path}")
        
        plt.show()
    
    def get_winner(self):
        """En iyi modeli belirle"""
        best_model = min(self.results.items(), 
                        key=lambda x: x[1]['test_rmse'])
        
        print("\n" + "=" * 70)
        print("🏆 EN İYİ MODEL")
        print("=" * 70)
        print(f"\nModel: {best_model[0]}")
        print(f"Test RMSE: {best_model[1]['test_rmse']:.4f}")
        print(f"Test MAE: {best_model[1]['test_mae']:.4f}")
        print(f"Parametre Sayısı: {best_model[1]['num_parameters']:,}")
        print(f"Eğitim Süresi: {best_model[1]['training_time']:.1f}s")
        print("=" * 70)
        
        return best_model
    
    def save_results(self, filename='model_benchmark_results.csv'):
        """Sonuçları CSV'ye kaydet"""
        import os
        
        output_dir = 'results/benchmarks'
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, filename)
        
        data = []
        for model_name, result in self.results.items():
            data.append({
                'model': model_name,
                'num_parameters': result['num_parameters'],
                'val_rmse': result['best_val_rmse'],
                'val_mae': result['best_val_mae'],
                'test_rmse': result['test_rmse'],
                'test_mae': result['test_mae'],
                'training_time': result['training_time'],
                'avg_epoch_time': result['avg_epoch_time']
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        print(f"\n✓ Sonuçlar kaydedildi: {filepath}")


def main():
    """Ana fonksiyon"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Benchmark')
    parser.add_argument('--data_dir', type=str, default='data/raw/ml-100k/',
                        help='Veri dizini')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Epoch sayısı')
    parser.add_argument('--save_plot', type=str, default='results/benchmarks/model_comparison.png',
                        help='Grafik kaydetme yolu')
    
    args = parser.parse_args()
    
    # Veriyi yükle
    print("Veri yükleniyor...")
    data_loader = DataLoader_ML(data_dir=args.data_dir)
    data_loader.prepare_data()
    
    # Benchmark oluştur
    benchmark = ModelBenchmark(data_loader, config.DEVICE)
    
    # Tüm modelleri test et
    benchmark.benchmark_all_models(epochs=args.epochs)
    
    # Sonuçları yazdır
    benchmark.print_comparison_table()
    
    # En iyi modeli göster
    benchmark.get_winner()
    
    # Grafikleri oluştur
    import os
    os.makedirs(os.path.dirname(args.save_plot), exist_ok=True)
    benchmark.plot_comparison(save_path=args.save_plot)
    
    # Sonuçları kaydet
    benchmark.save_results()
    
    print("\n✓ Benchmark tamamlandı!")


if __name__ == "__main__":
    main()
