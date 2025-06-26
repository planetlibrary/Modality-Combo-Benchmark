import torch
import pandas as pd
import numpy as np
import os
import fnmatch
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from torch.utils.data import Dataset
import torchio as tio
import warnings 
warnings.filterwarnings('ignore')
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# from utils.utils import pad_sequence, my_collate_fn, train_test_split

class PreprocessTransform:
    def __init__(self, new_shape):
        self.new_shape = new_shape  # New shape as a tuple (D, H, W)

    def __call__(self, img):
        resize_transform = tio.Resize(self.new_shape)
        img_resized = resize_transform(img)
        img_normalized = (img_resized - img_resized.mean()) / img_resized.std()
        return img_normalized
        
class MultimodalDataset(Dataset):
    def __init__(self, csv_file, img_folder, genetic_folder_path, atlas_folder_path, transform=None, validation=True):
        self.tabular_frame = pd.read_csv(csv_file)
        self.img_folder = img_folder
        self.transform = transform
        self.genetic_folder_path = genetic_folder_path
        self.validation = validation
        self.atlas_folder_path = atlas_folder_path
        self.atlas_types = ['Schaefer600Parc17Net', 'Glasser', 'DK', 'Destrieux']
        
        # First filter: Find valid samples with all modalities
        self.valid_samples = self.filter_valid_samples()
        
        # Keep only valid samples in the tabular frame
        self.tabular_frame = self.tabular_frame[self.tabular_frame['IMAGEUID'].isin([x[1] for x in self.valid_samples])]
        
        # Process the data for valid samples only
        self.features, self.features_dash, self.target, self.feature_size = self.preprocess_tabular_data()
        self.atlas_data = self.preprocess_all_atlas_data()
        
        print(f"Dataset initialized with {len(self.valid_samples)} valid samples")

    def filter_valid_samples(self):
        valid_samples = []
        invalid_count = {'genetic': 0, 'image': 0, 'atlas': 0}
        missing_atlas_samples = []  # New list to store samples missing only atlas data
        
        for _, row in self.tabular_frame.iterrows():
            imageuid = int(row['IMAGEUID'])
            ptid = row['PTID']
            
            # Check all modalities
            genetic_path = os.path.join(self.genetic_folder_path, f"{ptid}.csv")
            image_path = self.find_image_file(ptid, imageuid)
            atlas_path = os.path.join(self.atlas_folder_path, str(imageuid), "Morphometrics_SBM.csv")
            
            # Count missing modalities
            if not os.path.exists(genetic_path):
                invalid_count['genetic'] += 1
            if not image_path:
                invalid_count['image'] += 1
            if not os.path.exists(atlas_path):
                invalid_count['atlas'] += 1
                # Check if only atlas is missing
                if os.path.exists(genetic_path) and image_path:
                    missing_atlas_samples.append((ptid, imageuid))
            
            # Only include if all modalities are present
            if os.path.exists(genetic_path) and image_path and os.path.exists(atlas_path):
                valid_samples.append((ptid, imageuid))
        
        print(f"Total samples: {len(self.tabular_frame)}")
        print(f"Missing genetic data: {invalid_count['genetic']}")
        print(f"Missing image data: {invalid_count['image']}")
        print(f"Missing atlas data: {invalid_count['atlas']}")
        print(f"Valid samples with all modalities: {len(valid_samples)}")
        print(f"Samples missing only atlas data: {len(missing_atlas_samples)}")

        # File path to save the information
        file_path = "data_summary.txt"

        # Data to save
        total_samples = len(self.tabular_frame)
        missing_genetic_data = invalid_count['genetic']
        missing_image_data = invalid_count['image']
        missing_atlas_data = invalid_count['atlas']
        valid_samples_count = len(valid_samples)
        missing_atlas_samples_count = len(missing_atlas_samples)

        # Writing to a file
        with open(file_path, "w") as file:
            file.write(f"Total samples: {total_samples}\n")
            file.write(f"Missing genetic data: {missing_genetic_data}\n")
            file.write(f"Missing image data: {missing_image_data}\n")
            file.write(f"Missing atlas data: {missing_atlas_data}\n")
            file.write(f"Valid samples with all modalities: {valid_samples_count}\n")
            file.write(f"Samples missing only atlas data: {missing_atlas_samples_count}\n")

        print(f"Information saved to {file_path}")

        
        # Save samples missing only atlas data to a file
        with open('missing_atlas_samples.csv', 'w') as f:
            f.write("PTID,IMAGEUID\n")  # Header
            for ptid, imageuid in missing_atlas_samples:
                f.write(f"{ptid},{imageuid}\n")
        
        print("Saved missing atlas samples to 'missing_atlas_samples.csv'")
        return valid_samples

    def preprocess_tabular_data(self):
        one_hot_mapping = {
            'CN': [1, 0, 0],
            'MCI': [0, 1, 0],
            'Dementia': [0, 0, 1]
        }

        # print(self.tabular_frame.shape)
        
        features = self.tabular_frame.drop(columns=['IMAGEUID', 'IMAGEUID_bl', 'EXAMDATE_bl'])
        
        # Print class distribution
        class_counts = features['DX'].value_counts()
        print("\nClass distribution:")
        for cls, count in class_counts.items():
            print(f"{cls}: {count}")
        
        target_column = 'DX'
        feature_columns = [col for col in features.columns if col != target_column]
        
        categorical_columns = ['PTID', 'DX_bl', 'PTGENDER', 'PTETHCAT', 'PTRACCAT', 'PTMARRY', 'APOE4', 'FSVERSION', 'FLDSTRENG', 'FSVERSION_bl', 'FLDSTRENG_bl']

        # print("Unique vals")
        # for i in categorical_columns:
        #     print(features[i].value_counts())        
        # Encode categorical features
        # for col in categorical_columns:
        #     if col in feature_columns:
        #         features[col] = LabelEncoder().fit_transform(features[col])
        
        # Scale numerical features
        numerical_columns = list(set(feature_columns) - set(categorical_columns))
        # scaler = StandardScaler()
        # features[numerical_columns] = scaler.fit_transform(features[numerical_columns])
        
        # Process target and create features
        target = features[target_column].map(one_hot_mapping)
        X = features.drop(columns=[target_column])
        X_dash = X.drop(columns=['PTID'])
        y = target.tolist()
        
        # scaling

        if self.validation:
            with open('tabular_scaler.pkl','rb') as f:
                preprocessor = pickle.load(f)

            X_dash = preprocessor.transform(X_dash)

        else:
            # Create the ColumnTransformer to apply transformations
            # Apply the transformation
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_columns),   # Apply StandardScaler to numerical columns
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns[1:])    # Apply OneHotEncoder to categorical columns
                ],
                remainder='passthrough'  # Retain other columns as they are (though in this case, there are no other columns)
            )
            X_dash = preprocessor.fit_transform(X_dash)

            with open('tabular_scaler.pkl', 'wb') as f:
                pickle.dump(preprocessor, f)

        # Convert to tensors
        features_t = X.values
        features_t_dash = torch.tensor(X_dash, dtype=torch.float32)
        labels = torch.tensor(y, dtype=torch.float32)
        
        feature_size = features_t_dash.shape[1]
        print(f"\nFeature shapes: Full: {features_t.shape}, Without PTID: {features_t_dash.shape}")
        
        # print(features_t_dash[0].shape)
        return features_t, features_t_dash, labels, feature_size

    def preprocess_genetic_data(self, df):
        nucleotide_to_number = {'A': 1, 'G': 2, 'C': 3, 'T': 4}
        
        # Convert nucleotides to numbers
        for column in df.columns:
            df[column] = df[column].apply(lambda x: nucleotide_to_number[x] if x in nucleotide_to_number else np.random.randint(1, 5))
        
        # Scale the data
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_features, columns=df.columns)
        
        return scaled_df

    def load_genetic_data(self, ptid):
        genetic_file_path = os.path.join(self.genetic_folder_path, f"{ptid}.csv")
        genetic_df = pd.read_csv(genetic_file_path)
        scaled_gen_df = self.preprocess_genetic_data(genetic_df)
        genetic_data = torch.tensor(scaled_gen_df.values, dtype=torch.float32)
        return genetic_data

    def preprocess_all_atlas_data(self):
        atlas_data_dict = {}
        feature_cols = ['Gyrification', 'Depth', 'Area', 'GreyMatterVolume', 'Thickness']
        
        for ptid, imageuid in self.valid_samples:
            imageuid = int(float(imageuid))
            atlas_file_path = os.path.join(self.atlas_folder_path, str(imageuid), "Morphometrics_SBM.csv")
            
            # Read the atlas file
            df = pd.read_csv(atlas_file_path, delimiter='\t')
            atlas_features = {}
            
            # Define anatomical groups for each atlas
            atlas_regions = {
                'Schaefer600Parc17Net': {
                    'default_mode': [...],  # List of region indices
                    'salience': [...],
                    'executive': [...],
                    'memory': [...]
                },
                'Glasser': { ... },
                'DK': { ... },
                'Destrieux': { ... }
            }
            
            for atlas in self.atlas_types:
                # Get data for specific atlas
                atlas_df = df[df['Atlas'] == atlas]
                
                # Filter out rows with invalid data
                valid_data = atlas_df[feature_cols].dropna()  # Remove rows with NaN
                valid_data = valid_data[(valid_data != 0).all(axis=1)]  # Remove rows with all zeros
                valid_data = valid_data[~valid_data.isin([np.inf, -np.inf]).any(axis=1)]  # Remove rows with infinity
                valid_data = valid_data[~(valid_data < 0).any(axis=1)]  # Remove rows with negative values
                
                if len(valid_data) >= 50:
                    # If we have more than 50 valid rows, randomly sample 50
                    atlas_data = valid_data.sample(n=50, random_state=42).values
                else:
                    # If we have less than 50 valid rows, pad with median values
                    print(f"Warning: {atlas} for imageuid {imageuid} has only {len(valid_data)} valid rows")
                    atlas_data = valid_data.values
                    remaining_rows = 50 - len(valid_data)
                    median_values = valid_data.median().values.reshape(1, -1)
                    padding = np.tile(median_values, (remaining_rows, 1))
                    atlas_data = np.vstack([atlas_data, padding])
                
                # Apply MinMax normalization
                min_vals = atlas_data.min(axis=0)
                max_vals = atlas_data.max(axis=0)
                normalized = (atlas_data - min_vals) / (max_vals - min_vals + 1e-8)
                
                atlas_features[atlas] = normalized
            
            atlas_data_dict[imageuid] = atlas_features
        
        print(f"\nProcessed atlas data for {len(atlas_data_dict)} samples")
        print(f"Each atlas has shape: (50, {len(feature_cols)})")
        return atlas_data_dict

    def load_atlas_data(self, imageuid):
        """Load preprocessed atlas data for a specific imageuid with error handling"""
        try:
            imageuid = int(float(imageuid))
            if imageuid not in self.atlas_data:
                raise KeyError(f"imageuid {imageuid} not found in atlas data.")
            
            atlas_dict = {}
            for atlas in self.atlas_types:
                try:
                    atlas_dict[atlas] = torch.tensor(self.atlas_data[imageuid][atlas], dtype=torch.float32)
                except Exception as e:
                    print(f"Error processing atlas '{atlas}' for imageuid {imageuid}: {e}")
            
            return atlas_dict
        except Exception as e:
            print(f"Failed to load atlas data for imageuid {imageuid}: {e}")
            return None  # Or return an empty dict {}

                
    def find_image_file(self, ptid, imageuid):
        # pattern = f'ADNI_{ptid}_*_*_I{int(imageuid)}_mni_norm.nii.gz'
        pattern = f'ADNI_{ptid}_*_*_I{int(imageuid)}.nii'
        for file in os.listdir(self.img_folder):
            if fnmatch.fnmatch(file, pattern):
                return os.path.join(self.img_folder, file)
        return None

    def __len__(self):
        return len(self.tabular_frame)

    def __getitem__(self, idx):
        # Get identifiers
        ptid = self.tabular_frame.iloc[idx]['PTID']
        imageuid = self.tabular_frame.iloc[idx]['IMAGEUID']
        
        # Get tabular data
        tabular_data = self.features_dash[idx]
        
        # Get image data
        img_path = self.find_image_file(ptid, imageuid)
        img = tio.ScalarImage(img_path)
        img_data = torch.tensor(img.data, dtype=torch.float32)
        img_data = self.transform(img_data) if self.transform else img_data
        
        # Get genetic data
        genetic_data = self.load_genetic_data(ptid)
        
        # Get atlas data
        atlas_data = self.load_atlas_data(imageuid)

        return {
            'tabular_data': tabular_data,
            'image_data': img_data,
            'genetic_data': genetic_data,
            'atlas_data': atlas_data,
            'label': self.target[idx],
            'img_path': img_path,
            'genetic_path': os.path.join(self.genetic_folder_path, f"{ptid}.csv"),
            'tabular_row': self.tabular_frame.iloc[idx].drop(columns=['DX']).to_string()
        }

# def get_dataloaders(csv_file,img_folder, genetic_folder_path, atlas_folder_path, config, transforms=None):
#     dataset = MultimodalDataset(csv_file=csv_file,
#                                 img_folder=img_folder,
#                                 genetic_folder_path=genetic_folder_path,
#                                 atlas_folder_path=atlas_folder_path,
#                                 transform=PreprocessTransform((64, 128, 128)))

#     # Splitting the dataset
#     train_dataset, test_dataset = train_test_split(dataset, train_ratio=config.split_ratio)
#     print(len(train_dataset), len(test_dataset))

#     # Creating DataLoaders
#     train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=my_collate_fn, pin_memory=True, num_workers=config.num_workers)
#     test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=my_collate_fn, pin_memory=True, num_workers=config.num_workers)

#     return train_loader, test_loader,# (train_dist, test_dist, val_dist)


# Example usage:
if __name__ == "__main__":
    dataset = MultimodalDataset(csv_file='/home/sayantan/Alzheimer/ADNIMERGE_18Sep2023_final3.csv',
                            img_folder='/home/sayantan/Alzheimer/images_Adni_final_v5',
                            genetic_folder_path="/home/sayantan/Alzheimer/ADNI_Genetic_Merge",
                            atlas_folder_path='/home/sayantan/Alzheimer/nicara_preprocessed_iuids',
                            transform=PreprocessTransform((64, 128, 128)))
    
    # Test the first sample
    sample = dataset[0]
    print("\nSample shapes:")
    print(f"Tabular data: {sample['tabular_data'].shape}")
    print(f"Image data: {sample['image_data'].shape}")
    print(f"Genetic data: {sample['genetic_data'].shape}")
    for atlas_name, atlas_data in sample['atlas_data'].items():
        print(f"{atlas_name} data: {atlas_data.shape}")