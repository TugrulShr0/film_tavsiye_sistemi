"""
Model Değerlendirme Modülü
Tuğrul Şahar (233255027) - Burak Yetişer (233255007)

Bu modül model performansını değerlendirmek için metrikler içerir.
"""

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns


class Evaluator:
    """
    Model değerlendirme sınıfı
    """
    
    def __init__(self):
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_rmse': [],
            'val_rmse': [],
            'train_mae': [],
            'val_mae': [],
        }
    
    @staticmethod
    def calculate_rmse(predictions, targets):
        """
        Root Mean Square Error hesapla
        
        Args:
            predictions (array): Tahmin edilen değerler
            targets (array): Gerçek değerler
            
        Returns:
            float: RMSE değeri
        """
        return np.sqrt(mean_squared_error(targets, predictions))
    
    @staticmethod
    def calculate_mae(predictions, targets):
        """
        Mean Absolute Error hesapla
        
        Args:
            predictions (array): Tahmin edilen değerler
            targets (array): Gerçek değerler
            
        Returns:
            float: MAE değeri
        """
        return mean_absolute_error(targets, predictions)
    
    @staticmethod
    def calculate_mse(predictions, targets):
        """
        Mean Square Error hesapla
        
        Args:
            predictions (array): Tahmin edilen değerler
            targets (array): Gerçek değerler
            
        Returns:
            float: MSE değeri
        """
        return mean_squared_error(targets, predictions)
    
    @staticmethod
    def calculate_accuracy(predictions, targets, threshold=0.5):
        """
        Accuracy hesapla (tahmin ile gerçek değer arasındaki fark threshold'dan küçükse doğru)
        
        Args:
            predictions (array): Tahmin edilen değerler
            targets (array): Gerçek değerler
            threshold (float): Kabul edilebilir hata eşiği
            
        Returns:
            float: Accuracy değeri (0-1 arası)
        """
        correct = np.abs(predictions - targets) <= threshold
        return np.mean(correct)
    
    @staticmethod
    def calculate_precision_at_k(predicted_items, relevant_items, k):
        """
        Precision@K metriği
        
        Args:
            predicted_items (list): Önerilen itemler
            relevant_items (list): İlgili (beğenilen) itemler
            k (int): Top-K değeri
            
        Returns:
            float: Precision@K değeri
        """
        if k == 0:
            return 0.0
        
        predicted_k = predicted_items[:k]
        relevant_set = set(relevant_items)
        
        hits = sum(1 for item in predicted_k if item in relevant_set)
        
        return hits / k
    
    @staticmethod
    def calculate_recall_at_k(predicted_items, relevant_items, k):
        """
        Recall@K metriği
        
        Args:
            predicted_items (list): Önerilen itemler
            relevant_items (list): İlgili (beğenilen) itemler
            k (int): Top-K değeri
            
        Returns:
            float: Recall@K değeri
        """
        if len(relevant_items) == 0:
            return 0.0
        
        predicted_k = predicted_items[:k]
        relevant_set = set(relevant_items)
        
        hits = sum(1 for item in predicted_k if item in relevant_set)
        
        return hits / len(relevant_items)
    
    @staticmethod
    def calculate_ndcg_at_k(predicted_items, relevant_items, k):
        """
        Normalized Discounted Cumulative Gain@K metriği
        
        Args:
            predicted_items (list): Önerilen itemler
            relevant_items (list): İlgili itemler (sıralı - en alakalıdan en az alakalıya)
            k (int): Top-K değeri
            
        Returns:
            float: NDCG@K değeri
        """
        def dcg_at_k(items, k):
            items_k = items[:k]
            return sum((2**rel - 1) / np.log2(idx + 2) for idx, rel in enumerate(items_k))
        
        # Tahmin edilen itemler için relevance skorları
        predicted_k = predicted_items[:k]
        relevance_scores = [1 if item in relevant_items else 0 for item in predicted_k]
        
        # DCG hesapla
        dcg = dcg_at_k(relevance_scores, k)
        
        # İdeal DCG hesapla (tüm ilgili itemler başta)
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = dcg_at_k(ideal_relevance, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate_model(self, model, data_loader, device, criterion=None):
        """
        Modeli bir veri seti üzerinde değerlendir
        
        Args:
            model: PyTorch modeli
            data_loader: Veri yükleyici
            device: Hesaplama cihazı
            criterion: Loss fonksiyonu (opsiyonel)
            
        Returns:
            dict: Değerlendirme metrikleri
        """
        model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for user_ids, movie_ids, ratings in data_loader:
                user_ids = user_ids.to(device)
                movie_ids = movie_ids.to(device)
                ratings = ratings.to(device)
                
                # Tahmin yap
                predictions = model(user_ids, movie_ids)
                
                # Loss hesapla (eğer verilmişse)
                if criterion is not None:
                    loss = criterion(predictions, ratings)
                    total_loss += loss.item()
                
                # Listeye ekle
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(ratings.cpu().numpy())
        
        # Metrikleri hesapla
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        metrics = {
            'rmse': self.calculate_rmse(all_predictions, all_targets),
            'mae': self.calculate_mae(all_predictions, all_targets),
            'mse': self.calculate_mse(all_predictions, all_targets),
            'accuracy_0.5': self.calculate_accuracy(all_predictions, all_targets, threshold=0.5),
            'accuracy_1.0': self.calculate_accuracy(all_predictions, all_targets, threshold=1.0),
        }
        
        if criterion is not None:
            metrics['loss'] = total_loss / len(data_loader)
        
        return metrics
    
    def update_history(self, epoch, train_metrics, val_metrics):
        """Eğitim geçmişini güncelle"""
        self.metrics_history['train_loss'].append(train_metrics.get('loss', 0))
        self.metrics_history['val_loss'].append(val_metrics.get('loss', 0))
        self.metrics_history['train_rmse'].append(train_metrics['rmse'])
        self.metrics_history['val_rmse'].append(val_metrics['rmse'])
        self.metrics_history['train_mae'].append(train_metrics['mae'])
        self.metrics_history['val_mae'].append(val_metrics['mae'])
    
    def print_metrics(self, metrics, phase='Test'):
        """Metrikleri yazdır"""
        print(f"\n{phase} Sonuçları:")
        print("-" * 40)
        if 'loss' in metrics:
            print(f"Loss: {metrics['loss']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"Accuracy (±0.5): {metrics['accuracy_0.5']:.2%}")
        print(f"Accuracy (±1.0): {metrics['accuracy_1.0']:.2%}")
        print("-" * 40)
    
    def plot_training_history(self, save_path=None):
        """
        Eğitim geçmişini görselleştir
        
        Args:
            save_path (str): Grafiklerin kaydedileceği yol
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.metrics_history['train_loss']) + 1)
        
        # Loss grafiği
        axes[0, 0].plot(epochs, self.metrics_history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.metrics_history['val_loss'], 'r-', label='Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE grafiği
        axes[0, 1].plot(epochs, self.metrics_history['train_rmse'], 'b-', label='Train RMSE')
        axes[0, 1].plot(epochs, self.metrics_history['val_rmse'], 'r-', label='Validation RMSE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_title('Training and Validation RMSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # MAE grafiği
        axes[1, 0].plot(epochs, self.metrics_history['train_mae'], 'b-', label='Train MAE')
        axes[1, 0].plot(epochs, self.metrics_history['val_mae'], 'r-', label='Validation MAE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].set_title('Training and Validation MAE')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Overfitting analizi
        overfitting_score = [val - train for train, val in 
                           zip(self.metrics_history['train_rmse'], 
                               self.metrics_history['val_rmse'])]
        axes[1, 1].plot(epochs, overfitting_score, 'g-', label='Val RMSE - Train RMSE')
        axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Overfitting Score')
        axes[1, 1].set_title('Overfitting Analysis')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grafik kaydedildi: {save_path}")
        
        plt.show()
    
    def plot_prediction_distribution(self, predictions, targets, save_path=None):
        """
        Tahmin dağılımını görselleştir
        
        Args:
            predictions (array): Tahminler
            targets (array): Gerçek değerler
            save_path (str): Grafiklerin kaydedileceği yol
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Scatter plot - Tahmin vs Gerçek
        axes[0].scatter(targets, predictions, alpha=0.3, s=10)
        axes[0].plot([1, 5], [1, 5], 'r--', lw=2, label='Perfect Prediction')
        axes[0].set_xlabel('Actual Rating')
        axes[0].set_ylabel('Predicted Rating')
        axes[0].set_title('Predictions vs Actual Ratings')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Histogram - Tahmin dağılımı
        axes[1].hist(predictions, bins=50, alpha=0.7, label='Predictions', edgecolor='black')
        axes[1].hist(targets, bins=50, alpha=0.7, label='Actual', edgecolor='black')
        axes[1].set_xlabel('Rating')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Ratings')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Hata dağılımı
        errors = predictions - targets
        axes[2].hist(errors, bins=50, alpha=0.7, edgecolor='black', color='red')
        axes[2].axvline(x=0, color='k', linestyle='--', lw=2)
        axes[2].set_xlabel('Prediction Error')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title(f'Error Distribution (Mean: {np.mean(errors):.3f})')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grafik kaydedildi: {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Test
    print("Evaluator Test Ediliyor...")
    
    # Dummy veri
    np.random.seed(42)
    targets = np.random.uniform(1, 5, 1000)
    predictions = targets + np.random.normal(0, 0.5, 1000)
    predictions = np.clip(predictions, 1, 5)
    
    evaluator = Evaluator()
    
    print(f"RMSE: {evaluator.calculate_rmse(predictions, targets):.4f}")
    print(f"MAE: {evaluator.calculate_mae(predictions, targets):.4f}")
    print(f"Accuracy (±0.5): {evaluator.calculate_accuracy(predictions, targets, 0.5):.2%}")
    
    # Precision/Recall test
    predicted_items = [1, 2, 3, 4, 5]
    relevant_items = [2, 4, 6, 8]
    
    print(f"\nPrecision@5: {evaluator.calculate_precision_at_k(predicted_items, relevant_items, 5):.2f}")
    print(f"Recall@5: {evaluator.calculate_recall_at_k(predicted_items, relevant_items, 5):.2f}")
