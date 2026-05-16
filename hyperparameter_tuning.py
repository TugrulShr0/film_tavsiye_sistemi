
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
import os
from itertools import product
from datetime import datetime

from config import config
from ncf_model import NCFModel
from utils.data_loader import DataLoader_ML
from utils.evaluation import Evaluator


class HyperparameterTuner:
   
    
    def __init__(self, data_loader, device):
       
        self.data_loader = data_loader
        self.device = device
        self.results = []
    
    def train_with_params(self, params, train_loader, val_loader, max_epochs=30):
        
        
        model = NCFModel(
            num_users=self.data_loader.num_users,
            num_movies=self.data_loader.num_movies,
            embedding_dim=params['embedding_dim'],
            hidden_layers=params['hidden_layers'],
            dropout_rate=params['dropout_rate']
        ).to(self.device)
        
    
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
    
        evaluator = Evaluator()
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 5
        
    
        for epoch in range(max_epochs):
       
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
            
        
            val_metrics = evaluator.evaluate_model(model, val_loader, self.device, criterion)
            
        
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        return {
            'best_val_loss': best_val_loss,
            'best_val_rmse': val_metrics['rmse'],
            'best_val_mae': val_metrics['mae'],
            'epochs_trained': epoch + 1
        }
    
    def grid_search(self, param_grid, max_epochs=30):
   
        print("\n" + "=" * 70)
        print("HİPERPARAMETRE OPTİMİZASYONU - GRID SEARCH")
        print("=" * 70)
        
       
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        all_combinations = list(product(*param_values))
        total_combinations = len(all_combinations)
        
        print(f"\nToplam {total_combinations} kombinasyon test edilecek")
        print(f"Her kombinasyon için max {max_epochs} epoch eğitim\n")
     
        train_loader, val_loader, _ = self.data_loader.get_data_loaders(
            batch_size=256
        )
        
        best_params = None
        best_score = float('inf')
      
        for idx, combination in enumerate(all_combinations, 1):
            params = dict(zip(param_names, combination))
            
            print(f"\n[{idx}/{total_combinations}] Test ediliyor:")
            print(f"  Embedding: {params['embedding_dim']}")
            print(f"  Hidden: {params['hidden_layers']}")
            print(f"  LR: {params['learning_rate']}")
            print(f"  Dropout: {params['dropout_rate']}")
            print(f"  Weight Decay: {params['weight_decay']}")
            
            try:
                results = self.train_with_params(params, train_loader, val_loader, max_epochs)
                
               
                result_entry = {
                    'combination_id': idx,
                    'params': params,
                    'results': results
                }
                self.results.append(result_entry)
                
                print(f"  ✓ Val Loss: {results['best_val_loss']:.4f}")
                print(f"  ✓ Val RMSE: {results['best_val_rmse']:.4f}")
                print(f"  ✓ Epochs: {results['epochs_trained']}")
                
                if results['best_val_loss'] < best_score:
                    best_score = results['best_val_loss']
                    best_params = params
                    print(f"  🌟 YENİ EN İYİ!")
                
            except Exception as e:
                print(f"  ✗ Hata: {e}")
                continue
        
        print("\n" + "=" * 70)
        print("OPTİMİZASYON TAMAMLANDI")
        print("=" * 70)
        print(f"\nEn İyi Parametreler:")
        print(f"  Embedding: {best_params['embedding_dim']}")
        print(f"  Hidden: {best_params['hidden_layers']}")
        print(f"  LR: {best_params['learning_rate']}")
        print(f"  Dropout: {best_params['dropout_rate']}")
        print(f"  Weight Decay: {best_params['weight_decay']}")
        print(f"\nEn İyi Validation Loss: {best_score:.4f}")
        
        return best_params, self.results
    
    def random_search(self, param_distributions, n_iterations=20, max_epochs=30):
       
        print("\n" + "=" * 70)
        print("HİPERPARAMETRE OPTİMİZASYONU - RANDOM SEARCH")
        print("=" * 70)
        print(f"\n{n_iterations} rastgele kombinasyon test edilecek\n")
        
        train_loader, val_loader, _ = self.data_loader.get_data_loaders(batch_size=256)
        
        best_params = None
        best_score = float('inf')
        
        for iteration in range(1, n_iterations + 1):
            # Rastgele parametreler seç
            params = {}
            for param_name, distribution in param_distributions.items():
                if param_name == 'hidden_layers':
                    # Hidden layers için özel seçim
                    params[param_name] = distribution[np.random.randint(len(distribution))]
                elif isinstance(distribution, list):
                    params[param_name] = np.random.choice(distribution)
                elif isinstance(distribution, tuple) and len(distribution) == 2:
                    # Range (min, max)
                    if param_name in ['embedding_dim']:
                        params[param_name] = np.random.randint(distribution[0], distribution[1] + 1)
                    else:
                        params[param_name] = np.random.uniform(distribution[0], distribution[1])
            
            print(f"\n[{iteration}/{n_iterations}] Test ediliyor:")
            for key, value in params.items():
                print(f"  {key}: {value}")
            
            try:
                results = self.train_with_params(params, train_loader, val_loader, max_epochs)
                
                result_entry = {
                    'iteration': iteration,
                    'params': params,
                    'results': results
                }
                self.results.append(result_entry)
                
                print(f"  ✓ Val Loss: {results['best_val_loss']:.4f}")
                print(f"  ✓ Val RMSE: {results['best_val_rmse']:.4f}")
                
                if results['best_val_loss'] < best_score:
                    best_score = results['best_val_loss']
                    best_params = params
                    print(f"  🌟 YENİ EN İYİ!")
                
            except Exception as e:
                print(f"  ✗ Hata: {e}")
                continue
        
        print("\n" + "=" * 70)
        print("RANDOM SEARCH TAMAMLANDI")
        print("=" * 70)
        print(f"\nEn İyi Validation Loss: {best_score:.4f}")
        
        return best_params, self.results
    
    def save_results(self, filename='hyperparameter_results.json'):
        output_dir = 'results/hyperparameter_tuning'
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, filename)
        
        json_results = []
        for result in self.results:
            json_result = {
                'params': {k: (v if not isinstance(v, list) else str(v)) 
                          for k, v in result['params'].items()},
                'results': result['results']
            }
            if 'combination_id' in result:
                json_result['combination_id'] = result['combination_id']
            if 'iteration' in result:
                json_result['iteration'] = result['iteration']
            
            json_results.append(json_result)
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n✓ Sonuçlar kaydedildi: {filepath}")
    
    def print_summary(self):
        if not self.results:
            print("Henüz sonuç yok!")
            return
    
        summary_data = []
        for result in self.results:
            row = {
                'embedding_dim': result['params']['embedding_dim'],
                'learning_rate': result['params']['learning_rate'],
                'dropout_rate': result['params']['dropout_rate'],
                'val_loss': result['results']['best_val_loss'],
                'val_rmse': result['results']['best_val_rmse'],
                'val_mae': result['results']['best_val_mae'],
                'epochs': result['results']['epochs_trained']
            }
            summary_data.append(row)
        
        df = pd.DataFrame(summary_data)
        df = df.sort_values('val_loss')
        
        print("\n" + "=" * 70)
        print("EN İYİ 5 SONUÇ")
        print("=" * 70)
        print(df.head().to_string(index=False))


def main():
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Hiperparametre Optimizasyonu')
    parser.add_argument('--method', type=str, default='grid', 
                        choices=['grid', 'random'],
                        help='Arama metodu')
    parser.add_argument('--data_dir', type=str, default='data/raw/ml-100k/',
                        help='Veri dizini')
    parser.add_argument('--n_iterations', type=int, default=20,
                        help='Random search için deneme sayısı')
    parser.add_argument('--max_epochs', type=int, default=30,
                        help='Her kombinasyon için max epoch')
    
    args = parser.parse_args()
    
   
    print("Veri yükleniyor...")
    data_loader = DataLoader_ML(data_dir=args.data_dir)
    data_loader.prepare_data()
   
    tuner = HyperparameterTuner(data_loader, config.DEVICE)
    
    if args.method == 'grid':
      
        param_grid = {
            'embedding_dim': [32, 64, 128],
            'hidden_layers': [
                [64, 32],
                [128, 64, 32],
                [256, 128, 64]
            ],
            'learning_rate': [0.0005, 0.001, 0.002],
            'dropout_rate': [0.1, 0.2, 0.3],
            'weight_decay': [1e-5, 1e-4]
        }
        
        best_params, results = tuner.grid_search(param_grid, max_epochs=args.max_epochs)
    
    elif args.method == 'random':
       
        param_distributions = {
            'embedding_dim': (32, 128),  # Range
            'hidden_layers': [
                [64, 32],
                [128, 64, 32],
                [256, 128, 64],
                [128, 64],
                [64, 32, 16]
            ],
            'learning_rate': (0.0001, 0.003),
            'dropout_rate': (0.0, 0.5),
            'weight_decay': (1e-6, 1e-3)
        }
        
        best_params, results = tuner.random_search(
            param_distributions, 
            n_iterations=args.n_iterations,
            max_epochs=args.max_epochs
        )
    
   
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'hyperparameter_results_{args.method}_{timestamp}.json'
    tuner.save_results(filename)
    
    tuner.print_summary()
    
    print("\n✓ Optimizasyon tamamlandı!")


if __name__ == "__main__":
    main()
