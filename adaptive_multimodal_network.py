import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union
import itertools
from config import Config

class AdaptiveMultimodalNetwork(nn.Module):
    """
    Enhanced multimodal network that can handle different combinations of modalities
    for studying performance across 15 different data combinations.
    """
    
    def __init__(self, tabular_data_size=64, n_classes=3):
        super(AdaptiveMultimodalNetwork, self).__init__()
        
        # Define modality names and their corresponding sizes
        self.modality_names = ['tabular', 'genetic', 'image', 'atlas']
        self.modality_feature_sizes = {
            'tabular': 16,
            'genetic': 64,
            'image': 512,  # After reduction
            'atlas': 128,  # 4 atlases * 32 each
            'bio': 32      # Biological interaction features
        }
        
        # Modality availability flags
        self.modality_availability = {
            'tabular': True,
            'genetic': True,
            'image': True,
            'atlas': True
        }
        
        # Generate all possible combinations (2^4 - 1 = 15 combinations)
        self.modality_combinations = self._generate_combinations()
        
        # Learnable modality weights
        self.modality_weights = nn.ParameterDict({
            combo_name: nn.Parameter(torch.ones(len(combo)))
            for combo_name, combo in self.modality_combinations.items()
        })
        
        # Individual modality branches (from your original model)
        self._build_individual_branches(tabular_data_size, n_classes)
        
        # Adaptive fusion layers for different combinations
        self._build_adaptive_fusion_layers(n_classes)
        
        # Biological Brain Interaction (from your original model)
        self.brain_interaction = EnhancedBiologicalBrainInteraction()
        
    def _generate_combinations(self):
        """Generate all possible combinations of modalities (15 total)"""
        combinations = {}
        modalities = list(self.modality_names)
        
        # Generate all non-empty subsets
        for r in range(1, len(modalities) + 1):
            for combo in itertools.combinations(modalities, r):
                combo_name = '_'.join(sorted(combo))
                combinations[combo_name] = list(combo)
                
        return combinations
    
    def _build_individual_branches(self, tabular_data_size, n_classes):
        """Build individual modality processing branches"""
        
        # Tabular branch
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_data_size, 32),
            nn.InstanceNorm1d(32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.InstanceNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.InstanceNorm1d(16),
            nn.ReLU()
        )
        self.tabular_classifier = nn.Linear(16, n_classes)
        
        # Genetic branch
        self.genetic_branch = nn.Sequential(
            nn.Linear(500*6, 512),
            nn.InstanceNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.InstanceNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.InstanceNorm1d(64),
            nn.ReLU()
        )
        self.genetic_classifier = nn.Linear(64, n_classes)
        
        # Image branch
        self.image_branch = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Dropout3d(0.3),
            nn.Flatten()
        )
        
        # Image feature reduction
        self.image_flat_size = 64 * 8 * 16 * 16
        self.image_reduction = nn.Sequential(
            nn.Linear(self.image_flat_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.image_classifier = nn.Linear(self.image_flat_size, n_classes)
        
        # Atlas branches
        self.atlas_feature_size = 50 * 5
        self.atlas_branches = nn.ModuleDict({
            'Schaefer600Parc17Net': nn.Sequential(
                nn.Linear(self.atlas_feature_size, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 32)
            ),
            'Glasser': nn.Sequential(
                nn.Linear(self.atlas_feature_size, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 32)
            ),
            'DK': nn.Sequential(
                nn.Linear(self.atlas_feature_size, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 32)
            ),
            'Destrieux': nn.Sequential(
                nn.Linear(self.atlas_feature_size, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 32)
            )
        })
        
        self.atlas_classifiers = nn.ModuleDict({
            atlas: nn.Linear(32, n_classes) for atlas in self.atlas_branches
        })
        self.atlas_classifier_pooled = nn.Sequential(
            nn.Linear(32*4, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )
        
    def _build_adaptive_fusion_layers(self, n_classes):
        """Build adaptive fusion layers for different modality combinations"""
        
        self.fusion_layers = nn.ModuleDict()
        self.final_classifiers = nn.ModuleDict()
        
        for combo_name, combo in self.modality_combinations.items():
            # Calculate input size for this combination
            input_size = 0
            for modality in combo:
                if modality == 'atlas':
                    input_size += self.modality_feature_sizes['atlas']
                else:
                    input_size += self.modality_feature_sizes[modality]
            
            # Add biological features if atlas is present
            if 'atlas' in combo:
                input_size += self.modality_feature_sizes['bio']
            
            # Create fusion layer for this combination
            hidden_size = max(128, input_size // 4)
            self.fusion_layers[combo_name] = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
            self.final_classifiers[combo_name] = nn.Sequential(
                nn.Linear(hidden_size // 2, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, n_classes)
            )
    
    def set_modality_combination(self, combination: Union[str, List[str]]):
        """Set which modalities to use for forward pass"""
        if isinstance(combination, list):
            combination = '_'.join(sorted(combination))
        
        if combination not in self.modality_combinations:
            raise ValueError(f"Invalid combination: {combination}")
        
        self.current_combination = combination
        self.active_modalities = self.modality_combinations[combination]
        
        # Update availability flags
        for modality in self.modality_names:
            self.modality_availability[modality] = modality in self.active_modalities
    
    def process_modalities(self, tabular_data=None, genetic_data=None, 
                          image_data=None, atlas_data=None):
        """Process individual modalities based on availability"""
        
        processed_features = {}
        individual_predictions = {}
        
        # Process tabular data
        if self.modality_availability['tabular'] and tabular_data is not None:
            tabular_out = self.tabular_branch(tabular_data)
            processed_features['tabular'] = tabular_out
            individual_predictions['tabular'] = self.tabular_classifier(tabular_out)
        
        # Process genetic data
        if self.modality_availability['genetic'] and genetic_data is not None:
            genetic_flat = genetic_data.reshape(genetic_data.size(0), -1)
            genetic_out = self.genetic_branch(genetic_flat)
            processed_features['genetic'] = genetic_out
            individual_predictions['genetic'] = self.genetic_classifier(genetic_out)
        
        # Process image data
        if self.modality_availability['image'] and image_data is not None:
            image_features = self.image_branch(image_data)
            image_reduced = self.image_reduction(image_features)
            processed_features['image'] = image_reduced
            individual_predictions['image'] = self.image_classifier(image_features)
        
        # Process atlas data
        if self.modality_availability['atlas'] and atlas_data is not None:
            atlas_features = []
            atlas_predictions = {}
            
            for atlas_name, data in atlas_data.items():
                flat_data = data.view(data.size(0), -1)
                atlas_out = self.atlas_branches[atlas_name](flat_data)
                atlas_features.append(atlas_out)
                atlas_predictions[atlas_name] = self.atlas_classifiers[atlas_name](atlas_out)
            
            processed_features['atlas'] = torch.cat(atlas_features, dim=1)
            atlas_predictions_pooled = self.atlas_classifier_pooled(processed_features['atlas'])
            # print('processd featurs atlas: ', processed_features['atlas'].shape)
            individual_predictions['atlas'] = atlas_predictions_pooled
            # individual_predictions['atlas'] = atlas_predictions
            
            # Get biological interaction features if atlas data is available
            if image_data is not None or genetic_data is not None:
                bio_features, attention_info = self.brain_interaction(
                    atlas_data, image_data, genetic_data
                )
                processed_features['bio'] = bio_features
        
        return processed_features, individual_predictions
    
    def forward(self, tabular_data=None, genetic_data=None, image_data=None, 
                atlas_data=None, combination=None):
        """
        Forward pass with specified modality combination
        
        Args:
            tabular_data, genetic_data, image_data, atlas_data: Input data
            combination: String or list specifying which modalities to use
        """
        
        if combination is not None:
            self.set_modality_combination(combination)
        
        if not hasattr(self, 'current_combination'):
            raise ValueError("No modality combination specified. Use set_modality_combination() first.")
        
        # Process individual modalities
        processed_features, individual_predictions = self.process_modalities(
            tabular_data, genetic_data, image_data, atlas_data
        )
        
        # Combine features for current combination
        combined_features = []
        active_features = []
        
        for modality in self.active_modalities:
            if modality in processed_features:
                combined_features.append(processed_features[modality])
                active_features.append(modality)
        
        # Add biological features if atlas is active
        if 'atlas' in self.active_modalities and 'bio' in processed_features:
            combined_features.append(processed_features['bio'])
        
        if not combined_features:
            raise ValueError("No valid features found for current combination")
        
        # Fuse features
        fused_features = torch.cat(combined_features, dim=1)
        
        # Apply combination-specific fusion and classification
        combo_name = self.current_combination
        fusion_out = self.fusion_layers[combo_name](fused_features)
        final_prediction = self.final_classifiers[combo_name](fusion_out)
        
        # Apply modality weights to individual predictions
        weighted_predictions = {}
        if hasattr(self, 'modality_weights') and combo_name in self.modality_weights:
            weights = torch.softmax(self.modality_weights[combo_name], dim=0)
            for i, modality in enumerate(active_features):
                if modality in individual_predictions:
                    # print('weights[i]: ', weights[i], type(weights[i]))
                    # print(f'individual_predictions[{modality}] ', individual_predictions[modality], type(individual_predictions[modality]))
                    weighted_predictions[modality] = weights[i] * individual_predictions[modality]
        
        return {
            'final_prediction': final_prediction,
            'individual_predictions': individual_predictions,
            'weighted_predictions': weighted_predictions,
            'active_modalities': self.active_modalities,
            'combination_name': combo_name
        }
    
    def get_all_combinations(self):
        """Return all possible modality combinations"""
        return list(self.modality_combinations.keys())
    
    def evaluate_all_combinations(self, tabular_data=None, genetic_data=None,
                                image_data=None, atlas_data=None):
        """Evaluate model performance on all 15 combinations"""
        
        results = {}
        for combo_name in self.modality_combinations.keys():
            try:
                self.set_modality_combination(combo_name)
                result = self.forward(tabular_data, genetic_data, image_data, atlas_data)
                results[combo_name] = result
            except Exception as e:
                results[combo_name] = {'error': str(e)}
        
        return results

class EnhancedBiologicalBrainInteraction(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define brain networks for each atlas
        self.brain_networks = {
            'high_gyr_depth': list(range(0, 12)),    # High gyrification and depth
            'high_volume': list(range(12, 24)),       # High grey matter volume
            'high_thickness': list(range(24, 36)),    # High thickness
            'mixed_features': list(range(36, 50))     # Mixed patterns
        }
        
        # Atlas-specific network interactions
        self.atlas_network_processors = nn.ModuleDict({
            'Schaefer600Parc17Net': self._create_network_processor(),
            'Glasser': self._create_network_processor(),
            'DK': self._create_network_processor(),
            'Destrieux': self._create_network_processor()
        })
        
        # Cross-atlas attention
        self.atlas_attention = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )
        
        # Cross-network attention within each atlas
        self.network_attention = nn.MultiheadAttention(
            embed_dim=32,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )
        
        # Genetic pathway processing
        self.genetic_encoder = nn.Sequential(
            nn.Linear(500*6, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 192)
        )
        
        self.genetic_modulators = nn.ModuleDict({
            'amyloid_effect': nn.Sequential(
                nn.Linear(64, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, 32)
            ),
            'tau_effect': nn.Sequential(
                nn.Linear(64, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, 32)
            ),
            'apoe_effect': nn.Sequential(
                nn.Linear(64, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, 32)
            )
        })
        
        # Fixed Structure-Function Coupling
        self.structure_function_coupling = nn.Sequential(
            nn.Linear(544, 256),  # Changed to match actual input size (128 + 416)
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64)
        )
        self.ncombined_features = 96
        # Final integration layer remains the same
        self.final_integration = nn.Sequential(
            nn.Linear(self.ncombined_features, 64),  # 64(structure-function) + 32(genetic)
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    
    # def _create_network_processor(self):
    #     return nn.ModuleDict({
    #         network: nn.Sequential(
    #             nn.Linear(5, 64),
    #             nn.LayerNorm(64),
    #             nn.ReLU(),
    #             nn.Linear(64, 32)
    #         ) for network in self.brain_networks.keys()
    #     })

    def _create_network_processor(self):
        return nn.ModuleDict({
            network: nn.Sequential(
                nn.Linear(5, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, 32)
            ) for network in self.brain_networks.keys()
        })

    def process_atlas(self, atlas_data, atlas_name):
        network_states = {}
        network_processor = self.atlas_network_processors[atlas_name]
        
        # Now each network represents feature-based groups
        for network_name, region_indices in self.brain_networks.items():
            # These regions now represent similar feature patterns
            network_regions = atlas_data[:, region_indices, :]
            network_state = network_processor[network_name](network_regions.mean(dim=1))
            network_states[network_name] = network_state
        
        # Stack and apply network attention
        network_stack = torch.stack(list(network_states.values()), dim=1)  # [batch, n_networks, 32]
        network_attended, network_attention = self.network_attention(
            network_stack, network_stack, network_stack
        )
        
        return network_attended, network_attention
    
    def forward(self, atlas_data =None, mri_features=None, genetic_features=None):
        batch_size = genetic_features.size(0)
        
        # 1. Process each atlas
        atlas_features = {}
        atlas_attentions = {}
        for atlas_name in atlas_data.keys():
            atlas_features[atlas_name], atlas_attentions[atlas_name] = self.process_atlas(
                atlas_data[atlas_name], atlas_name
            )
        
        # 2. Cross-atlas attention
        atlas_stack = torch.stack(list(atlas_features.values()), dim=1)  # [batch, n_atlases, n_networks, 32]
        atlas_integrated, atlas_cross_attention = self.atlas_attention(
            atlas_stack.view(-1, atlas_stack.size(2), atlas_stack.size(3)),
            atlas_stack.view(-1, atlas_stack.size(2), atlas_stack.size(3)),
            atlas_stack.view(-1, atlas_stack.size(2), atlas_stack.size(3))
        )
        atlas_integrated = atlas_integrated.reshape(batch_size, len(atlas_data), -1)  # [batch, n_atlases, 32]

        genetic_effects,genetic_modulation = None, None
        # 3. Process genetic pathways
        if genetic_features is not None:
            genetic_flat = genetic_features.reshape(batch_size, -1)
            genetic_encoded = self.genetic_encoder(genetic_flat)
            
            genetic_pathways = {
                'amyloid': genetic_encoded[:, :64],
                'tau': genetic_encoded[:, 64:128],
                'apoe': genetic_encoded[:, 128:]
            }
        
            # 4. Apply genetic modulation
            genetic_effects = []
            for pathway_name, pathway_features in genetic_pathways.items():
                modulator = self.genetic_modulators[f'{pathway_name}_effect']
                modulation = modulator(pathway_features)
                genetic_effects.append(modulation)
            
            genetic_modulation = sum(genetic_effects)  # [batch, 32]

        # 5. Structure-Function Integration
        if mri_features is not None:
            mri_state = mri_features.view(batch_size, -1)[:, :32]  # [batch, 32]
        
        atlas_flat = atlas_integrated.view(batch_size, -1)  # [batch, n_atlases*32]
        
        # Combine structural and functional information
        structure_function_combined = None
        if mri_features is not None:
            structure_function = torch.cat([atlas_flat, mri_state], dim=1)  # [batch, (n_atlases*32 + 32)]
            structure_function_combined = self.structure_function_coupling(structure_function)  # [batch, 64]
        
        # Final integration with genetic effects
        if structure_function_combined is not None and genetic_modulation is not None:
            final_features = torch.cat([structure_function_combined, genetic_modulation], dim=1)  # [batch, 96]
            final_state = self.final_integration(final_features)  # [batch, 32]

        elif structure_function_combined is not None:
            final_features = structure_function_combined.copy()  # [batch, 64]
            self.ncombined_features = 64
            final_state = self.final_integration(final_features)  # [batch, 32]

        elif genetic_modulation is not None:
            final_features = genetic_modulation.copy()  # [batch, 32]
            self.ncombined_features = 32
            final_state = self.final_integration(final_features)  # [batch, 32]
        else:
            raise ValueError("Both of the features (MRI and Genetic) can not be none")
        
        return final_state, {
            'atlas_attentions': atlas_attentions,
            'cross_atlas_attention': atlas_cross_attention,
            'genetic_effects': genetic_effects,
            'structure_function': structure_function_combined
        }


# Example usage and training script
class MultimodalExperiment:
    """Class to manage experiments across different modality combinations"""
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.results = {}
        
    def run_combination_study(self, data_loader, combinations=None):
        """Run experiments on specified combinations or all combinations"""
        
        if combinations is None:
            combinations = self.model.get_all_combinations()
        
        for combination in combinations:
            print(f"Testing combination: {combination}")
            self.model.set_modality_combination(combination)
            
            # Run evaluation for this combination
            combo_results = self._evaluate_combination(data_loader, combination)
            self.results[combination] = combo_results
            
            print(f"Accuracy for {combination}: {combo_results['accuracy']:.4f}")
        
        return self.results
    
    def _evaluate_combination(self, data_loader, combination):
        """Evaluate model on specific combination"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                tabular, genetic, image, atlas, labels = batch
                
                # Move to device
                tabular = tabular.to(self.device) if tabular is not None else None
                genetic = genetic.to(self.device) if genetic is not None else None
                image = image.to(self.device) if image is not None else None
                atlas = {k: v.to(self.device) for k, v in atlas.items()} if atlas else None
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(tabular, genetic, image, atlas, combination)
                predictions = torch.argmax(outputs['final_prediction'], dim=1)
                
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
        
        accuracy = correct / total
        return {'accuracy': accuracy, 'correct': correct, 'total': total}
    
    def get_best_combination(self):
        """Get the best performing combination"""
        if not self.results:
            return None
        
        best_combo = max(self.results.keys(), 
                        key=lambda x: self.results[x].get('accuracy', 0))
        return best_combo, self.results[best_combo]
    
    def print_summary(self):
        """Print summary of all combination results"""
        print("\n" + "="*60)
        print("MODALITY COMBINATION STUDY RESULTS")
        print("="*60)
        
        sorted_results = sorted(self.results.items(), 
                               key=lambda x: x[1].get('accuracy', 0), 
                               reverse=True)
        
        for i, (combo, result) in enumerate(sorted_results, 1):
            if 'accuracy' in result:
                print(f"{i:2d}. {combo:20s} - Accuracy: {result['accuracy']:.4f}")
            else:
                print(f"{i:2d}. {combo:20s} - Error: {result.get('error', 'Unknown')}")
        
        print("="*60)


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = AdaptiveMultimodalNetwork(tabular_data_size=64, n_classes=3)
    
    # Create experiment manager
    experiment = MultimodalExperiment(model)
    
    # Example: Test specific combinations
    test_combinations = ['tabular_genetic', 'image_atlas', 'tabular_genetic_image_atlas']
    
    # Print all available combinations
    print("All available combinations:")
    for combo in model.get_all_combinations():
        print(f"  - {combo}")