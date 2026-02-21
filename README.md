# Alzheimer's Disease Stage Classification

## An Integrated DWT–LBP–DenseNet Framework for Multimodal Brain Image Analysis

### Project Overview
This project implements a multimodal classification framework for Alzheimer's Disease (AD) stage identification using MRI, PET, and CT brain images. The framework achieves high accuracy through targeted preprocessing and advanced feature extraction techniques.

### Key Features
- **Preprocessing Pipeline**: Median filtering, skull stripping, min-max normalization, and image resizing
- **Multimodal Feature Extraction**:
  - **DWT (Discrete Wavelet Transform)**: Extracts multiresolution spatial features from MRI
  - **LBP (Local Binary Pattern)**: Captures metabolic texture distributions from PET
  - **DenseNet**: Obtains high-level anatomical features from CT
- **Feature Fusion**: Combines features from all modalities with PCA dimensionality reduction
- **Classification**: Fully connected neural network with dropout and batch normalization

### Classification Categories
1. Normal Cognition
2. Mild Cognitive Impairment
3. Early Alzheimer's Disease
4. Moderate Alzheimer's Disease

### Target Performance
- **Accuracy**: 99.42%
- **F1 Score**: 98.28%

## Project Structure
```
Alzheimer-s-Disease-Stage-Classification/
├── config.py                 # Configuration parameters
├── preprocessing.py          # Image preprocessing pipeline
├── feature_extraction.py     # DWT, LBP, DenseNet extractors
├── feature_fusion.py         # Feature fusion and PCA
├── model.py                  # Classification model
├── utils.py                  # Visualization and metrics
├── main.py                   # Main training script
├── requirements.txt          # Python dependencies
├── models/                   # Saved models
└── outputs/                  # Results and visualizations
    ├── processed_images/     # Preprocessing step images
    └── *.png                 # Plots and curves
```

## Installation

### Option 1: Docker (Recommended)

#### Prerequisites
- Docker Desktop installed and running
- NVIDIA GPU with Docker GPU support (optional, for faster training)
- Dataset extracted to `../dataset/Data/`

#### Quick Start with Docker
1. **Build and run the container:**

**With GPU:**
```powershell
.\run_docker.ps1
```

Or manually:
```bash
# Build the Docker image (GPU)
docker-compose build

# Run the container (GPU)
docker-compose up
```

**CPU-only (no GPU required):**
```bash
# Build the Docker image (CPU)
docker-compose -f docker-compose.cpu.yml build

# Run the container (CPU)
docker-compose -f docker-compose.cpu.yml up
```

2. **Access results:**
   - Training outputs: `./outputs/`
   - Saved models: `./models/`

#### Docker Benefits
- ✅ No dependency conflicts
- ✅ Isolated environment
- ✅ Easy deployment
- ✅ GPU support included
- ✅ Reproducible results

### Option 2: Local Installation

#### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster training)

#### Setup
1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Ensure dataset is in the correct location:
```
../dataset/Data/
├── Non Demented/
├── Mild Cognitive Impairment/
├── Early Alzheimer's Disease/
└── Moderate Dementia/
```

## Usage

### Using Docker (Recommended)

#### Run Training
```powershell
# Windows PowerShell
.\run_docker.ps1
```

```bash
# Linux/Mac
docker-compose up
```

#### Interactive Shell (for debugging)
```bash
docker-compose run --rm alzheimer-classifier bash
```

#### Stop Container
```bash
docker-compose down
```

### Local Execution

#### Training
Run the complete training pipeline with multiple learning rates:
```bash
python main.py
```

This will:
1. Load and split the dataset
2. Preprocess all images
3. Extract multimodal features (DWT, LBP, DenseNet)
4. Apply feature fusion with PCA
5. Train models with learning rates: 0.1, 0.01, 0.001, 0.0001
6. Generate all visualizations and metrics

### Output Files
The script generates:
- **Training curves**: Accuracy and loss plots for each learning rate
- **Confusion matrices**: Classification confusion for each model
- **ROC curves**: Multi-class ROC analysis
- **Learning rate comparison**: Comprehensive comparison across all LRs
- **Preprocessing steps**: Visualization of preprocessing pipeline
- **Classification reports**: Detailed precision, recall, F1 scores
- **Saved models**: Trained model weights (.h5 files)

## Configuration
Edit `config.py` to modify:
- Image dimensions
- Learning rates to test
- Number of epochs
- Batch size
- PCA components
- Network architecture (dense units, dropout)

## Methodology

### 1. Preprocessing Pipeline
- **Median Filtering**: Removes noise while preserving edges
- **Skull Stripping**: Isolates brain tissue using morphological operations
- **Min-Max Normalization**: Standardizes intensity values to [0, 1]
- **Resizing**: Ensures uniform 224x224 input dimensions

### 2. Feature Extraction
#### MRI - Discrete Wavelet Transform (DWT)
- Extracts multiresolution spatial features
- Uses Haar wavelet with 3 decomposition levels
- Captures both approximation and detail coefficients

#### PET - Local Binary Pattern (LBP)
- Represents regional texture distributions
- Radius: 3, Points: 24
- Generates rotation-invariant texture histograms

#### CT - DenseNet121
- Pre-trained on ImageNet
- Extracts high-level anatomical features
- Global average pooling for feature vectors

### 3. Feature Fusion
- Concatenates features from all modalities
- Applies StandardScaler normalization
- PCA reduces dimensionality to 100 components
- Retains most discriminative features

### 4. Classification Network
- Multiple fully connected layers (512 → 256 → 128 neurons)
- Batch normalization after each layer
- Dropout (0.5) for regularization
- L2 regularization to prevent overfitting
- Softmax output for multi-class classification

## Results
The trained models generate:
- Accuracy curves showing training progression
- Loss curves for overfitting analysis
- Confusion matrices for error analysis
- ROC curves with AUC scores per class
- Learning rate comparison plots
- detailed classification reports

## Technical Details
- **Framework**: TensorFlow/Keras
- **Optimizer**: Adam with adaptive learning rates
- **Loss Function**: Sparse categorical crossentropy
- **Metrics**: Accuracy, Precision, Recall, F1-Score
- **Callbacks**: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

## Citation
If you use this code, please cite:
```
An Integrated DWT–LBP–DenseNet Framework for Multimodal Brain Image Analysis 
in Alzheimer's Disease Stage Classification
```

## License
This project is provided for educational and research purposes.

## Authors
Implementation based on research paper specifications for AD classification.

## Contact
For questions or issues, please refer to the project documentation.
