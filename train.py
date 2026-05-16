
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import os
import argparse
from tqdm import tqdm
import time

from config import config
from ncf_model import NCFModel
from utils.data_loader import DataLoader_ML
from utils.evaluation import Evaluator


class Trainer:
 
    
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, config):
       
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config
        
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
      
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
          
        )
        
       
        self.evaluator = Evaluator()
        
      
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        
        self.epoch_times = []
    
    def train_epoch(self, epoch):
       
        self.model.train()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (user_ids, movie_ids, ratings) in enumerate(pbar):
           
            user_ids = user_ids.to(self.device)
            movie_ids = movie_ids.to(self.device)
            ratings = ratings.to(self.device)
                    
            self.optimizer.zero_grad()
        
            predictions = self.model(user_ids, movie_ids)
    
            loss = self.criterion(predictions, ratings)
   
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            total_loss += loss.item()
            all_predictions.extend(predictions.detach().cpu().numpy())
            all_targets.extend(ratings.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
  
        avg_loss = total_loss / len(self.train_loader)
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        metrics = {
            'loss': avg_loss,
            'rmse': self.evaluator.calculate_rmse(all_predictions, all_targets),
            'mae': self.evaluator.calculate_mae(all_predictions, all_targets),
        }
        
        return metrics
    
    def validate(self):

        metrics = self.evaluator.evaluate_model(
            self.model, 
            self.val_loader, 
            self.device, 
            self.criterion
        )
        return metrics
    
    def test(self):

        metrics = self.evaluator.evaluate_model(
            self.model, 
            self.test_loader, 
            self.device, 
            self.criterion
        )
        return metrics
    
    def train(self, num_epochs):

        print("\n" + "=" * 70)
        print("MODEL EĞİTİMİ BAŞLIYOR")
        print("=" * 70)
        print(f"Cihaz: {self.device}")
        print(f"Epoch Sayısı: {num_epochs}")
        print(f"Batch Size: {self.config.BATCH_SIZE}")
        print(f"Learning Rate: {self.config.LEARNING_RATE}")
        print("=" * 70 + "\n")
        
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            
            
            train_metrics = self.train_epoch(epoch)
  
            val_metrics = self.validate()
  
            self.evaluator.update_history(epoch, train_metrics, val_metrics)
    
            self.scheduler.step(val_metrics['loss'])
     
            epoch_time = time.time() - epoch_start_time
            self.epoch_times.append(epoch_time)
   
            print(f"\nEpoch {epoch}/{num_epochs} - {epoch_time:.2f}s")
            print(f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"RMSE: {train_metrics['rmse']:.4f} | "
                  f"MAE: {train_metrics['mae']:.4f}")
            print(f"Val   Loss: {val_metrics['loss']:.4f} | "
                  f"RMSE: {val_metrics['rmse']:.4f} | "
                  f"MAE: {val_metrics['mae']:.4f}")
            
   
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                
                if self.config.SAVE_MODEL:
                    self.save_checkpoint(epoch, val_metrics, is_best=True)
                    print(f"✓ En iyi model kaydedildi! (Val Loss: {val_metrics['loss']:.4f})")
            else:
                self.patience_counter += 1
            
            if epoch % self.config.CHECKPOINT_FREQ == 0:
                if self.config.SAVE_MODEL:
                    self.save_checkpoint(epoch, val_metrics, is_best=False)
            
         
            if self.patience_counter >= self.config.PATIENCE:
                print(f"\n⚠ Early stopping triggered! ({self.config.PATIENCE} epoch boyunca iyileşme yok)")
                break
            
            print("-" * 70)
        
        print("\n" + "=" * 70)
        print("EĞİTİM TAMAMLANDI")
        print("=" * 70)
        print(f"Ortalama Epoch Süresi: {np.mean(self.epoch_times):.2f}s")
        print(f"En İyi Val Loss: {self.best_val_loss:.4f}")
        print("=" * 70 + "\n")
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
    
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': {
                'num_users': self.model.num_users,
                'num_movies': self.model.num_movies,
                'embedding_dim': self.model.embedding_dim,
                'hidden_layers': self.config.HIDDEN_LAYERS,
                'dropout_rate': self.config.DROPOUT_RATE,
            }
        }
        
        if is_best:
            path = os.path.join(self.config.MODEL_DIR, 'best_model.pth')
        else:
            path = os.path.join(self.config.MODEL_DIR, f'checkpoint_epoch_{epoch}.pth')
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path):
    
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"✓ Checkpoint yüklendi: {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Val Loss: {checkpoint['metrics']['loss']:.4f}")


def main():
   
    parser = argparse.ArgumentParser(description='NCF Model Eğitimi')
    parser.add_argument('--data_dir', type=str, default='data/raw/ml-100k/',
                        help='Veri dizini')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                        help='Epoch sayısı')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                        help='Batch boyutu')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                        help='Öğrenme oranı')
    parser.add_argument('--embedding_dim', type=int, default=config.EMBEDDING_DIM,
                        help='Embedding boyutu')
    parser.add_argument('--resume', type=str, default=None,
                        help='Devam edilecek checkpoint')
    
    args = parser.parse_args()
    
 
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.EMBEDDING_DIM = args.embedding_dim
    
   
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
   
    print("Veri yükleniyor...")
    data_loader = DataLoader_ML(data_dir=args.data_dir)
    data_loader.prepare_data()
    
    # DataLoader'ları oluştur
    train_loader, val_loader, test_loader = data_loader.get_data_loaders(
        batch_size=config.BATCH_SIZE
    )
    

    print("\nModel oluşturuluyor...")
    model = NCFModel(
        num_users=data_loader.num_users,
        num_movies=data_loader.num_movies,
        embedding_dim=config.EMBEDDING_DIM,
        hidden_layers=config.HIDDEN_LAYERS,
        dropout_rate=config.DROPOUT_RATE
    )
    
    print(model)
   
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=config.DEVICE,
        config=config
    )
 
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
  
    trainer.train(num_epochs=config.EPOCHS)
    
   
    print("\nTest setinde değerlendirme yapılıyor...")
    test_metrics = trainer.test()
    trainer.evaluator.print_metrics(test_metrics, phase='Test')
    

    print("\nGrafikler oluşturuluyor...")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    trainer.evaluator.plot_training_history(
        save_path=os.path.join(config.RESULTS_DIR, 'training_history.png')
    )
    
    print("\n Eğitim tamamlandı!")


if __name__ == "__main__":
    main()
