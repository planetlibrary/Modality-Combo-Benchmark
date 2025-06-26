# CLI-enabled training pipeline
# ===================== Standard Libraries =====================
import os
import json
import argparse
import warnings
import textwrap

# ===================== Third-party Libraries =====================
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, default_collate
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from torchinfo import summary
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
# import wandb
import torchio as tio
import datetime

# ===================== Local Modules =====================
# from config import cfg
# from data_loader import get_dataloaders
# from models import get_model
# from utils.utils import *
# from utils.model_tracker import ModelChangeTracker
# from brain_model_nicara_v2 import get_model
from dataset_nicara_v2 import MultimodalDataset, PreprocessTransform
from adaptive_multimodal_network import AdaptiveMultimodalNetwork

from pathlib import Path
import time
from tqdm import tqdm
import sys
from config import Config
from utils.utils import *
# Monkey-patch tqdm to always write to stdout
# tqdm.__init__ = lambda self, *args, **kwargs: super(tqdm, self).__init__(*args, file=sys.stdout, **kwargs)








    

class ModalityCombinationTrainer:
    """
    Comprehensive trainer for studying multimodal network performance 
    across all 15 modality combinations
    """
    
    def __init__(self, model, save_dir, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

        # self.cfg = cfg
        
        # Results storage
        self.training_results = {}
        self.evaluation_results = {}
        self.combination_metrics = {}
        
        # Get all combinations
        self.all_combinations = model.get_all_combinations()
        print(f"Total combinations to study: {len(self.all_combinations)}")
        
    def train_combination(self, combination, train_loader, test_loader,
                         epochs=50, lr=0.001, patience=10):
        """Train model on specific modality combination"""
        
        print(f"\n{'='*60}")
        print(f"Training combination: {combination}")
        print(f"{'='*60}")
        
        self.cfg = Config(combination)
        # Set model to use this combination
        self.model.set_modality_combination(combination)
        
        # Setup optimizer and criterion
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.7)
        
        
        train_losses = []
        test_losses = []
        train_accuracies = []
        test_accuracies = []

        history = {
        'train_loss': [],
        'test_loss': [],
        'train_accuracy': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1_score': [],
        'test_accuracy': [],
        'test_precision': [],
        'test_recall': [],
        'test_f1_score': [],
        }
        best_test_acc = 0.0
        patience_counter = 0
        
        start_time = time.time()

        os.makedirs(self.cfg.log_dir, exist_ok=True)
        with open(self.cfg.log_dir/"statistics_v1.txt", "w") as file:
            file.write(f'The training for {combination} is started at: {get_date()}\n')
        
        for epoch in range(epochs):
            estart_time = time.time()

            # Training tracking
            epoch_train_logits, epoch_train_targets = [], []
            epoch_train_labels, epoch_train_predictions = [], []
            train_loss, train_correct, train_total = 0, 0, 0

            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', leave=True)
            
            for batch in loop:

                tabular = batch['tabular_data'].to(self.device)
                image = batch['image_data'].to(self.device)
                genetic = batch['genetic_data'].to(self.device)
                atlas = {k: v.to(self.device) for k, v in batch['atlas_data'].items()}
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                # try:
                # Forward pass
                outputs = self.model(tabular, genetic, image, atlas, combination)
                loss = criterion(outputs['final_prediction'], labels)
                    
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                    
                # Statistics
                train_loss += loss.item()
                predictions = torch.argmax(outputs['final_prediction'], dim=1)
                # print(f'prediction: {predictions.shape}, label:{labels.shape}')
                train_correct += (predictions == torch.argmax(labels,dim=1)).sum().item()
                train_total += labels.size(0)

                _, predicted = torch.max(outputs['final_prediction'].data, 1)
                    
                epoch_train_logits.append(outputs['final_prediction'].cpu().detach())
                epoch_train_targets.append(labels.cpu().detach())
                epoch_train_labels.append(torch.max(labels, 1)[1].cpu())
                epoch_train_predictions.append(predicted.cpu())
                
                # except FileNotFoundError  as e:
                # print(f"Error in --line 150: {e}")
                # continue

                loop.set_postfix(loss=loss.item(), accuracy=f'{100 * train_correct / train_total:.2f}%')

            train_loss_avg = train_loss / train_total
            train_acc = 100 * train_correct / train_total
            
            # Test phase
            test_loss_avg, test_acc, epoch_test_labels, epoch_test_predictions = self._validate(combination, test_loader, criterion, epoch, epochs)


            epoch_train_labels = torch.cat(epoch_train_labels)
            epoch_train_predictions = torch.cat(epoch_train_predictions)
            plot_cm(epoch_train_labels, epoch_train_predictions, self.cfg, epoch, 'Train')
            epoch_test_labels = torch.cat(epoch_test_labels)
            epoch_test_predictions = torch.cat(epoch_test_predictions)
            plot_cm(epoch_test_labels, epoch_test_predictions, self.cfg, epoch, 'Test')

            _, train_precision,train_recall,train_f1=  compute_metrics(epoch_train_labels, epoch_train_predictions)
            _, test_precision,test_recall, test_f1=  compute_metrics(epoch_test_labels, epoch_test_predictions)
            
            # train_precision = precision_score(epoch_train_labels, epoch_train_predictions, average='macro')
            # train_recall = recall_score(epoch_train_labels, epoch_train_predictions, average='macro')
            # train_f1 = f1_score(epoch_train_labels, epoch_train_predictions, average='macro')
            # test_precision = precision_score(epoch_test_labels, epoch_test_predictions, average='macro')
            # test_recall = recall_score(epoch_test_labels, epoch_test_predictions, average='macro')
            # test_f1 = f1_score(epoch_test_labels, epoch_test_predictions, average='macro')

            # print('train_loss', type(train_loss),'train_total', type(train_total),'train_loss_avg', type(train_loss_avg),'test_loss_avg', type(test_loss_avg))
            train_losses.append(train_loss_avg)
            test_losses.append(test_loss_avg)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            
            history['train_loss'].append(train_loss_avg)
            history['test_loss'].append(test_loss_avg)
            history['train_accuracy'].append(train_acc)
            history['train_precision'].append(train_precision)
            history['train_recall'].append(train_recall)
            history['train_f1_score'].append(train_f1)
            history['test_accuracy'].append(test_acc)
            history['test_precision'].append(test_precision)
            history['test_recall'].append(test_recall)
            history['test_f1_score'].append(test_f1)
            



                
            # Update learning rate
            scheduler.step(train_loss)
            
            # Record metrics
            # train_losses.append(train_loss / len(train_loader))
            # val_losses.append(val_loss)
            # val_accuracies.append(val_acc)
            os.makedirs(self.cfg.checkpoints_dir, exist_ok=True)
            status  = save_at_n_epoch(self.model, epoch, self.cfg.checkpoints_dir/"model.pth", self.cfg)
            
            # Early stopping check
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                patience_counter = 0
                # Save best model for this combination
                self._save_best_model(combination)
            else:
                patience_counter += 1

            # Print and save results
            # print(f'Epoch {epoch+1}/{epochs}: '
            #     f'Train Loss: {train_loss_avg:.4f}, '
            #     f'Train Acc: {train_acc:.2f}%, '
            #     f'Train F1: {train_f1:.4f}, '
            #     f'Test Loss: {test_loss_avg:.4f}, '
            #     f'Test Acc: {test_acc:.2f}%, '
            #     f'Test F1: {test_f1:.4f}')

            print(
                f'[{get_date()}]: '
                f'Combination: {combination} -> '
                f'Epoch {epoch+1}/{epochs}: '
                f'Train Loss: {train_loss_avg:.4f}, '
                f'Train Acc: {train_acc:.2f}%, '
                f'Train Precision: {train_precision:.4f}, '
                f'Train Recall: {train_recall:.4f}, '
                f'Train F1: {train_f1:.4f}, '
                f'Test Loss: {test_loss_avg:.4f}, '
                f'Test Acc: {test_acc:.2f}%, '
                f'Test Precision: {test_precision:.4f}, '
                f'Test Recall: {test_recall:.4f}, '
                f'Test F1: {test_f1:.4f}, '
                f"Time Taken: {get_time_diff(time.time(),estart_time)}"
            )
            
            # Save results to file
            with open(self.cfg.log_dir/"statistics_v1.txt", "a") as file:
                file.write(
                        f'[{get_date()}]: '
                        f'Combination: {combination} -> '
                        f'Epoch {epoch+1}/{epochs}: '
                        f'Train Loss: {train_loss_avg:.4f}, '
                        f'Train Acc: {train_acc:.2f}%, '
                        f'Train Precision: {train_precision:.4f}, '
                        f'Train Recall: {train_recall:.4f}, '
                        f'Train F1: {train_f1:.4f}, '
                        f'Test Loss: {test_loss_avg:.4f}, '
                        f'Test Acc: {test_acc:.2f}%, '
                        f'Test Precision: {test_precision:.4f}, '
                        f'Test Recall: {test_recall:.4f}, '
                        f'Test F1: {test_f1:.4f}, '
                        f"Time Taken: {get_time_diff(time.time(),estart_time)}\n"
                        ) 
            
            
        training_time = get_time_diff(time.time(),start_time)
            
        
        os.makedirs(self.cfg.metrics_dir, exist_ok = True)
        
        with open(os.path.join(self.cfg.metrics_dir, 'metrics.json'), 'w') as file:
            json.dump(history, file, indent=4)

        plot_save_config(self.cfg)

        plot_metrics(self.cfg)
        plot_single_metrics(self.cfg)



        
        
        # Store results
        self.training_results[combination] = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accuracies': train_accuracies,
            'test_accuracies': test_accuracies,
            'best_test_accuracy': best_test_acc,
            'training_time': training_time,
            'epochs_trained': epoch + 1
        }
        
        print(f"Completed training for {combination}")
        print(f"Best validation accuracy: {best_test_acc:.4f}")
        print(f"Training time: {training_time}")

        plot_save_config(self.cfg)
        
        return best_test_acc
    
    def _validate(self, combination, test_loader, criterion, epoch, epochs):
        """Validate model on specific combination"""
        self.model.eval()
        # val_loss = 0.0
        # correct = 0
        # total = 0
        test_loss, test_correct, test_total = 0.0, 0, 0
        epoch_test_logits, epoch_test_targets = [], []
        epoch_test_labels, epoch_test_predictions = [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f' Testing ... Epoch {epoch+1}/{epochs}', leave=True):
                # Move data to device
                tabular = batch['tabular_data'].to(self.device)
                image = batch['image_data'].to(self.device)
                genetic = batch['genetic_data'].to(self.device)
                atlas = {k: v.to(self.device) for k, v in batch['atlas_data'].items()}
                labels = batch['label'].to(self.device)
                
                try:
                    outputs = self.model(tabular, genetic, image, atlas, combination)
                    loss = criterion(outputs['final_prediction'], labels)
                    
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs['final_prediction'].data, 1)
                    predictions = torch.argmax(outputs['final_prediction'], dim=1)
                    test_correct += (predictions == torch.argmax(labels,dim=1)).sum().item()
                    test_total += labels.size(0)

                    epoch_test_logits.append(outputs['final_prediction'].cpu().detach())
                    epoch_test_targets.append(labels.cpu().detach())
                    epoch_test_labels.append(torch.max(labels, 1)[1].cpu())
                    epoch_test_predictions.append(predicted.cpu())
                    
                except Exception as e:
                    # print(f"Error --line 281: {e}")
                    continue
        
        test_loss_avg = test_loss / test_total
        test_acc = 100* test_correct / test_total 

        # print('test_loss', type(test_loss),'test_total', type(test_total))
        
        return test_loss_avg, test_acc, epoch_test_labels, epoch_test_predictions
    
    def evaluate_combination(self, combination, test_loader):
        """Comprehensive evaluation of a specific combination"""
        
        # Load best model for this combination
        self._load_best_model(combination)
        self.model.set_modality_combination(combination)
        self.model.eval()
        
        test_loss, test_correct, test_total = 0, 0, 0
        epoch_test_logits, epoch_test_targets = [], []
        epoch_test_labels, epoch_test_predictions = [], []
        epoch_test_probabilities = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f' Testing ...', leave=True):
                # Move data to device
                # assert tabular is not None, "tabular_data must not be None"
                # assert genetic is not None, "genetic_data must not be None"
                # assert image is not None, "image_data must not be None"
                # assert atlas is not None, "atlas_data must not be None"

                tabular = batch['tabular_data'].to(self.device)
                image = batch['image_data'].to(self.device)
                genetic = batch['genetic_data'].to(self.device)
                atlas = {k: v.to(self.device) for k, v in batch['atlas_data'].items()}
                labels = batch['label'].to(self.device)
                
                try:
                    outputs = self.model(tabular, genetic, image, atlas, combination)
                    # test_loss = criterion(outputs['final_prediction'], labels)
                    
                    # test_loss += test_loss.item()
                    _, predicted = torch.max(outputs['final_prediction'].data, 1)
                    predictions = torch.argmax(outputs['final_prediction'], dim=1)
                    test_correct += (predictions == torch.argmax(labels,dim=1)).sum().item()
                    test_total += labels.size(0)

                    epoch_test_logits.append(outputs['final_prediction'].cpu().detach())
                    epoch_test_targets.append(labels.cpu().detach())
                    epoch_test_labels.append(torch.max(labels, 1)[1].cpu())
                    epoch_test_predictions.append(predicted.cpu())
                    
                except Exception as e:
                    # print(f"Error --line 334: {e}")
                    continue
        
        # test_loss_avg, test_acc, epoch_test_labels, epoch_test_predictions = self._validate(combination, test_loader, criterion, epoch, epochs)


        # epoch_train_labels = torch.cat(epoch_train_labels)
        # epoch_train_predictions = torch.cat(epoch_train_predictions)
        epoch_test_labels = torch.cat(epoch_test_labels)
        epoch_test_predictions = torch.cat(epoch_test_predictions)
        
        # test_loss_avg = test_loss / test_total
        # test_acc = test_correct / test_total 
        
        # return test_loss_avg, test_acc, epoch_test_labels, epoch_test_predictions
        # Calculate metrics
        # print(f'acc --test labels: {epoch_test_labels}, {len(epoch_test_labels)}')
        # print(f'acc -- test predictions: {epoch_test_predictions}, {len(epoch_test_predictions)}')
        accuracy = accuracy_score(epoch_test_labels, epoch_test_predictions)
        precision = precision_score(epoch_test_labels, epoch_test_predictions, average='macro')
        recall = recall_score(epoch_test_labels, epoch_test_predictions, average='macro')
        f1 = f1_score(epoch_test_labels, epoch_test_predictions, average='macro')
        cm = confusion_matrix(epoch_test_labels, epoch_test_predictions)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'predictions': epoch_test_predictions,
            'labels': epoch_test_labels,
            'probabilities': epoch_test_probabilities
        }
        
        self.evaluation_results[combination] = metrics
        return metrics
    
    def train_all_combinations(self, train_loader, test_loader, **training_kwargs):
        """Train model on all 15 combinations"""
        
        print(f"\nStarting comprehensive study of {len(self.all_combinations)} combinations")
        print("="*80)
        
        combination_performances = {}
        
        for i, combination in enumerate(self.all_combinations, 1):
            print(f"\nProgress: {i}/{len(self.all_combinations)}")
            
            #  try:
            best_acc = self.train_combination(combination, train_loader, test_loader, **training_kwargs)
            combination_performances[combination] = best_acc
                
            # except Exception as e:
            #  print(f"Error training combination {combination}: {e}")
            # combination_performances[combination] = 0.0
            #continue
        
        # Sort combinations by performance
        sorted_combinations = sorted(
            combination_performances.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        print(f"\n{'='*80}")
        print("TRAINING RESULTS SUMMARY")
        print(f"{'='*80}")
        
        for i, (combination, accuracy) in enumerate(sorted_combinations, 1):
            print(f"{i:2d}. {combination:25s} - Test Accuracy: {accuracy:.4f}")
        
        return combination_performances
    
    def evaluate_all_combinations(self, test_loader):
        """Evaluate all combinations on test set"""
        
        print(f"\nEvaluating all {len(self.all_combinations)} combinations on test set")
        print("="*80)
        
        for i, combination in enumerate(self.all_combinations, 1):
            print(f"\nEvaluating {i}/{len(self.all_combinations)}: {combination}")
            
            # try:
            metrics = self.evaluate_combination(combination, test_loader)
            print(f"Test Accuracy: {metrics['accuracy']:.4f}")
            # except Exception as e:
            # print(f"Error evaluating {combination}: {e}")
            # continue
        
    def analyze_modality_importance(self):
        """Analyze importance of individual modalities"""
        
        if not self.evaluation_results:
            print("No evaluation results found. Run evaluate_all_combinations first.")
            return
        
        # Group combinations by number of modalities
        modality_groups = {1: [], 2: [], 3: [], 4: []}
        
        for combination, metrics in self.evaluation_results.items():
            num_modalities = len(combination.split('_'))
            if num_modalities in modality_groups:
                modality_groups[num_modalities].append({
                    'combination': combination,
                    'accuracy': metrics['accuracy']
                })
        
        # Analyze single modality performance
        print("\nSINGLE MODALITY PERFORMANCE:")
        print("-" * 40)
        for item in sorted(modality_groups[1], key=lambda x: x['accuracy'], reverse=True):
            print(f"{item['combination']:15s}: {item['accuracy']:.4f}")
        
        # Analyze modality combinations
        modality_contributions = {
            'tabular': [],
            'genetic': [],
            'image': [],
            'atlas': []
        }
        
        for combination, metrics in self.evaluation_results.items():
            modalities = combination.split('_')
            for modality in ['tabular', 'genetic', 'image', 'atlas']:
                if modality in modalities:
                    modality_contributions[modality].append(metrics['accuracy'])
        
        # Calculate average contribution
        print("\nMODALITY CONTRIBUTION ANALYSIS:")
        print("-" * 40)
        for modality, accuracies in modality_contributions.items():
            if accuracies:
                avg_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                print(f"{modality:10s}: {avg_acc:.4f} ± {std_acc:.4f} (n={len(accuracies)})")
    
    def create_performance_visualizations(self):
        """Create comprehensive visualizations of combination performance"""
        
        if not self.evaluation_results:
            print("No evaluation results found.")
            return
        
        # Prepare data for visualization
        combinations = list(self.evaluation_results.keys())
        accuracies = [self.evaluation_results[combo]['accuracy'] for combo in combinations]
        
        # Sort by performance
        sorted_data = sorted(zip(combinations, accuracies), key=lambda x: x[1], reverse=True)
        sorted_combinations, sorted_accuracies = zip(*sorted_data)
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Multimodal Network Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Bar plot of all combinations
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(sorted_combinations)), sorted_accuracies, 
                      color='skyblue', alpha=0.7)
        ax1.set_xlabel('Modality Combinations')
        ax1.set_ylabel('Test Accuracy')
        ax1.set_title('Performance Across All 15 Combinations')
        ax1.set_xticks(range(len(sorted_combinations)))
        ax1.set_xticklabels(sorted_combinations, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars, sorted_accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 2. Performance by number of modalities
        ax2 = axes[0, 1]
        modality_counts = {}
        for combo, acc in zip(combinations, accuracies):
            count = len(combo.split('_'))
            if count not in modality_counts:
                modality_counts[count] = []
            modality_counts[count].append(acc)
        
        counts = sorted(modality_counts.keys())
        mean_accs = [np.mean(modality_counts[c]) for c in counts]
        std_accs = [np.std(modality_counts[c]) for c in counts]
        
        ax2.bar(counts, mean_accs, yerr=std_accs, capsize=5, 
               color='lightcoral', alpha=0.7)
        ax2.set_xlabel('Number of Modalities')
        ax2.set_ylabel('Average Test Accuracy')
        ax2.set_title('Performance vs Number of Modalities')
        ax2.set_xticks(counts)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Heatmap of pairwise combinations
        ax3 = axes[1, 0]
        modalities = ['tabular', 'genetic', 'image', 'atlas']
        heatmap_data = np.zeros((4, 4))
        
        for combo, acc in zip(combinations, accuracies):
            combo_mods = combo.split('_')
            if len(combo_mods) == 2:
                for i, mod1 in enumerate(modalities):
                    for j, mod2 in enumerate(modalities):
                        if i < j and mod1 in combo_mods and mod2 in combo_mods:
                            heatmap_data[i, j] = acc
                            heatmap_data[j, i] = acc
        
        # Fill diagonal with single modality performance
        for i, mod in enumerate(modalities):
            if mod in self.evaluation_results:
                heatmap_data[i, i] = self.evaluation_results[mod]['accuracy']
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd',
                   xticklabels=modalities, yticklabels=modalities, ax=ax3)
        ax3.set_title('Pairwise Modality Performance Heatmap')
        
        # 4. Training convergence comparison (top 5 combinations)
        ax4 = axes[1, 1]
        top_5_combos = sorted_combinations[:5]
        
        for combo in top_5_combos:
            if combo in self.training_results:
                val_accs = self.training_results[combo]['test_accuracies']
                ax4.plot(val_accs, label=combo, linewidth=2)
        
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Test Accuracy')
        ax4.set_title('Epoch vs Test Accuracy (Top 5 Combinations)')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_detailed_report(self):
        """Generate comprehensive analysis report"""
        
        report = {
            'experiment_summary': {
                'total_combinations': len(self.all_combinations),
                'completed_training': len(self.training_results),
                'completed_evaluation': len(self.evaluation_results)
            },
            'performance_ranking': [],
            'modality_analysis': {},
            'training_efficiency': {},
            'best_practices': []
        }
        
        # Performance ranking
        if self.evaluation_results:
            sorted_results = sorted(
                self.evaluation_results.items(),
                key=lambda x: x[1]['accuracy'],
                reverse=True
            )
            
            for combo, metrics in sorted_results:
                report['performance_ranking'].append({
                    'combination': combo,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1_score': metrics['f1_score']
                })
        
        # Modality analysis
        modality_stats = {
            'tabular': {'present': 0, 'absent': 0, 'avg_acc_present': 0, 'avg_acc_absent': 0},
            'genetic': {'present': 0, 'absent': 0, 'avg_acc_present': 0, 'avg_acc_absent': 0},
            'image': {'present': 0, 'absent': 0, 'avg_acc_present': 0, 'avg_acc_absent': 0},
            'atlas': {'present': 0, 'absent': 0, 'avg_acc_present': 0, 'avg_acc_absent': 0}
        }
        
        for combo, metrics in self.evaluation_results.items():
            combo_mods = combo.split('_')
            acc = metrics['accuracy']
            
            for modality in ['tabular', 'genetic', 'image', 'atlas']:
                if modality in combo_mods:
                    modality_stats[modality]['present'] += 1
                    modality_stats[modality]['avg_acc_present'] += acc
                else:
                    modality_stats[modality]['absent'] += 1
                    modality_stats[modality]['avg_acc_absent'] += acc
        
        # Calculate averages
        for modality, stats in modality_stats.items():
            if stats['present'] > 0:
                stats['avg_acc_present'] /= stats['present']
            if stats['absent'] > 0:
                stats['avg_acc_absent'] /= stats['absent']
            
            stats['contribution'] = stats['avg_acc_present'] - stats['avg_acc_absent']
        
        report['modality_analysis'] = modality_stats
        
        # Training efficiency
        if self.training_results:
            for combo, results in self.training_results.items():
                report['training_efficiency'][combo] = {
                    'training_time': results['training_time'],
                    'epochs_trained': results['epochs_trained'],
                    'best_test_accuracy': results['best_test_accuracy'],
                    'convergence_rate': results['best_test_accuracy'] / results['epochs_trained']
                }
        
        # Generate recommendations
        if self.evaluation_results:
            best_combo = max(self.evaluation_results.items(), key=lambda x: x[1]['accuracy'])
            worst_combo = min(self.evaluation_results.items(), key=lambda x: x[1]['accuracy'])
            
            report['best_practices'] = [
                f"Best performing combination: {best_combo[0]} (Accuracy: {best_combo[1]['accuracy']:.4f})",
                f"Worst performing combination: {worst_combo[0]} (Accuracy: {worst_combo[1]['accuracy']:.4f})",
                f"Performance improvement: {best_combo[1]['accuracy'] - worst_combo[1]['accuracy']:.4f}",
            ]
            
            # Add modality importance insights
            most_important = max(modality_stats.items(), key=lambda x: x[1]['contribution'])
            least_important = min(modality_stats.items(), key=lambda x: x[1]['contribution'])
            
            report['best_practices'].extend([
                f"Most important modality: {most_important[0]} (contribution: +{most_important[1]['contribution']:.4f})",
                f"Least important modality: {least_important[0]} (contribution: +{least_important[1]['contribution']:.4f})"
            ])
        
        # Save report
        with open(self.save_dir / 'detailed_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _save_best_model(self, combination):
        """Save best model for specific combination"""
        comb_folder = self.cfg.checkpoints_dir
        os.makedirs(comb_folder, exist_ok=True)
        model_path = comb_folder / 'best_model.pth'  
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'combination': combination,
        }, model_path)

    
    def _load_best_model(self, combination):
        """Load best model for specific combination"""
        # comb_folder = self.save_dir / combination
        model_path = self.cfg.checkpoints_dir / f'best_model.pth'
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f"Warning: No saved model found for combination {combination}")
    
    def save_results(self):
        """Save all results to files"""
        
        # Save training results
        with open(self.save_dir / 'training_results.json', 'w') as f:
            json.dump(self.training_results, f, indent=2, default=str)
        
        # Save evaluation results
        eval_results_serializable = {}
        for combo, metrics in self.evaluation_results.items():
            eval_results_serializable[combo] = {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'confusion_matrix': metrics['confusion_matrix']
            }
        
        with open(self.save_dir / 'evaluation_results.json', 'w') as f:
            json.dump(eval_results_serializable, f, indent=2)
        
        print(f"Results saved to {self.save_dir}")




def pad_sequence(sequences):
    max_size = max([s.size() for s in sequences])
    padded_sequences = [F.pad(s, (0, max_size[2] - s.size(2), 0, max_size[1] - s.size(1), 0, max_size[0] - s.size(0))) for s in sequences]
    return torch.stack(padded_sequences)

def my_collate_fn(batch):
    batch = {k: [d[k] for d in batch] for k in batch[0]}
    for k in batch:
        if k == 'image_data':
            batch[k] = pad_sequence(batch[k])
        elif k == 'atlas_data':
            atlas_data = {atlas: [] for atlas in batch[k][0].keys()}
            for sample in batch[k]:
                for atlas, data in sample.items():
                    atlas_data[atlas].append(data)
            for atlas in atlas_data:
                atlas_data[atlas] = default_collate(atlas_data[atlas])
            batch[k] = atlas_data
        else:
            batch[k] = default_collate(batch[k])
    return batch

def train_test_split(dataset, train_ratio=0.7):
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])

def check_shapes(dictionary):
    for key, value in dictionary.items():
        # Check if the value has a shape attribute (like numpy arrays or pandas DataFrames)
        if hasattr(value, 'shape'):
            print(f"Key: {key}, Shape: {value.shape}")
        # If the value is a list, check its length and the length of inner lists if nested
        elif isinstance(value, list):
            try:
                print(f"Key: {key}, Shape: ({len(value)}, {len(value[0]) if isinstance(value[0], list) else 'n/a'})")
            except IndexError:
                print(f"Key: {key}, Shape: ({len(value)},)")
        # For anything else, just print the type
        else:
            print(f"Key: {key}, Type: {type(value)}")

def visualize_attention_weights(attention_weights, title, save_path=None):
    """
    Safely visualize attention weights with proper reshaping
    
    Args:
        attention_weights: torch.Tensor or numpy array
        title: str, title for the plot
        save_path: str, optional path to save the figure
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    # Ensure 2D shape
    if len(attention_weights.shape) == 1:
        attention_weights = attention_weights.reshape(-1, 1)
    elif len(attention_weights.shape) > 2:
        attention_weights = attention_weights.mean(axis=tuple(range(len(attention_weights.shape)-2)))
    
    plt.figure(figsize=(10, 7))
    sns.heatmap(attention_weights, cmap="viridis")
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        return save_path
    
    return plt.gcf()

# Then in the training loop:
# def log_attention_maps(attention_info, epoch, wandb_run=None):
#     """Log attention maps to wandb"""
#     if attention_info is None:
#         return
        
#     for key, value in attention_info.items():
#         if isinstance(value, dict):
#             for sub_key, attention_map in value.items():
#                 if isinstance(attention_map, torch.Tensor):
#                     fig_path = visualize_attention_weights(
#                         attention_map,
#                         f'{key}/{sub_key} Attention Weights',
#                         f'attention_{key}_{sub_key}_epoch_{epoch+1}.png'
#                     )
#                     if wandb_run:
#                         wandb_run.log({f"attention_maps/{key}/{sub_key}": wandb.Image(fig_path)})
        
#         elif isinstance(value, (list, tuple)):
#             for i, tensor in enumerate(value):
#                 if isinstance(tensor, torch.Tensor):
#                     fig_path = visualize_attention_weights(
#                         tensor,
#                         f'{key} Effect {i+1}',
#                         f'attention_{key}_effect_{i+1}_epoch_{epoch+1}.png'
#                     )
#                     if wandb_run:
#                         wandb_run.log({f"attention_maps/{key}/effect_{i+1}": wandb.Image(fig_path)})
        
#         elif isinstance(value, torch.Tensor):
#             fig_path = visualize_attention_weights(
#                 value,
#                 f'{key} Attention Weights',
#                 f'attention_{key}_epoch_{epoch+1}.png'
#             )
#             if wandb_run:
#                 wandb_run.log({f"attention_maps/{key}": wandb.Image(fig_path)})



# Fetching the dataset

def main():
    """Main execution function"""
    
    print("Initializing Multimodal Combination Study")
    print("="*60)
    
    # Set device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f"Using device: {device}")
    
    # Initialize model
    model = AdaptiveMultimodalNetwork(tabular_data_size=85, n_classes=3)

    print('model loaded')

    cfg = Config()
    
    # Initialize trainer
    trainer = ModalityCombinationTrainer(model, save_dir=cfg.result_dir, device=cfg.device)
    print('trainer loaded')
    # Creating DataLoaders
    dataset = MultimodalDataset(csv_file=cfg.csv_file,
                                img_folder=cfg.img_folder,#'/home/sayantan/img_v6_mount/',
                                genetic_folder_path=cfg.genetic_folder_path,
                                atlas_folder_path=cfg.atlas_folder_path,
                                transform=PreprocessTransform(cfg.transform_shape),
                                validation=False)
    print('dataset loaded')

    # Splitting the dataset
    train_dataset, test_dataset = train_test_split(dataset, train_ratio=cfg.split_ratio)
    train_label_dist = data_dist(train_dataset)
    test_label_dist = data_dist(test_dataset)
    print_dist(train_label_dist, test_label_dist, config=cfg)

    print(len(train_dataset), len(test_dataset))
    print('dataloader loaded')
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=my_collate_fn, pin_memory=True, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=my_collate_fn, pin_memory=True, num_workers=cfg.num_workers)

    
    # print(f"Data loaded: Train={len(train_loader.dataset)}, "
    #       f"Val={len(test_loader.dataset)}, Test={len(test_loader.dataset)}")
    
    
    
    # Option 1: Train all combinations (full study)
    print("\nStarting comprehensive combination study...")
    combination_performances = trainer.train_all_combinations(
        train_loader, test_loader, **cfg.training_config
    )
    
    # Option 2: Train specific combinations (for quick testing)
    # selected_combinations = ['tabular', 'genetic', 'tabular_genetic', 
    #                         'tabular_genetic_image_atlas']
    # for combo in selected_combinations:
    #     trainer.train_combination(combo, train_loader, test_loader, **training_config)
    
    # Evaluate all combinations
    print("\nEvaluating all combinations on test set...")
    trainer.evaluate_all_combinations(test_loader)
    
    # Analyze results
    print("\nAnalyzing modality importance...")
    trainer.analyze_modality_importance()
    
    # Create visualizations
    print("\nCreating performance visualizations...")
    trainer.create_performance_visualizations()
    
    # Generate detailed report
    print("\nGenerating detailed report...")
    report = trainer.generate_detailed_report()
    
    # Save all results
    trainer.save_results()
    
    # Print final summary
    print("\n" + "="*80)
    print("STUDY COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Results saved in: {trainer.save_dir}")
    print("\nKey findings:")
    for practice in report['best_practices']:
        print(f"  - {practice}")
    
    return trainer, report


if __name__ == "__main__":
    trainer, report = main()

