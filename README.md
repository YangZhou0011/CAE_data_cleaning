This repository contains Python scripts for processing image datasets, training a Convolutional Autoencoder (CAE) for anomaly detection, evaluating anomaly detection performance, and visualizing results. Each script serves a distinct purpose, as described below.

## Files and Descriptions

1. **Dataset Preparation (`16000clean5dirty.pkl` Generation)**  
   - **Purpose**: Preprocesses and organizes a dataset of clean and anomalous driving images into a single Pickle file for further use.  
   - **Description**: Loads images from six folders (`clean`, `foggy`, `greenmarker`, `plastic`, `raindrop`, `hitwall`) on a specified path, resizes them to 224x224 pixels, converts them to NumPy arrays, and stores them in a dictionary. The resulting dataset is saved as `16000clean5dirty.pkl`.  
   - **Key Libraries**: `os`, `numpy`, `PIL`, `pickle`.

2. **CAE Training and Evaluation**  
   - **Purpose**: Trains a Convolutional Autoencoder (CAE) to detect anomalies in driving images and evaluates its performance.  
   - **Description**: Loads the preprocessed dataset (`16000clean7dirty.pkl`), normalizes images, and trains a CAE model using PyTorch. The model uses a combined loss function (MSE + optional regularization term) to reconstruct images. It evaluates reconstruction quality (MSE, RC, smoothed RC) and anomaly detection performance (AUC) at specified epochs, saving results to a directory.  
   - **Key Libraries**: `torch`, `numpy`, `pickle`, `sklearn`, `scipy`, `matplotlib`.  
   - **Parameters**: Configurable hyperparameters include learning rate, batch size, number of epochs, and anomaly type (e.g., `plastic`).

3. **Anomaly Detection Post-Processing**  
   - **Purpose**: Analyzes trained model outputs to identify and filter anomalous images.  
   - **Description**: Loads RC values from a trained model’s output (`rc_records.npy`), calculates statistics (mean and standard deviation) for the top 50% of RC values, and filters out images below a threshold (mean - σ * STD). It then computes the percentage of remaining dirty data.  
   - **Key Libraries**: `numpy`.  
   - **Note**: The script currently has an error (`ValueError: too many values to unpack`), indicating a mismatch in data structure that needs fixing.

4. **Visualization of RC Values**  
   - **Purpose**: Generates plots comparing raw and smoothed Reconstruction Correlation (RC) values across different models.  
   - **Description**: Loads RC data from three model outputs (baseline GCL, MSE + λ, and MSE alone), creates two subplots: one for raw RC values and one for smoothed RC values. The plots are formatted for IEEE publication standards using LaTeX and saved as a high-resolution PNG.  
   - **Key Libraries**: `matplotlib`, `numpy`, `scipy`.  
   - **Output**: A figure (`rc_values_comparison3.png`) showing RC comparisons for the `plastic` anomaly type.

## Requirements
- Python 3.10+
- Libraries: `numpy`, `torch`, `PIL`, `pickle`, `sklearn`, `scipy`, `matplotlib`
- Optional: LaTeX installation for rendering plot text (visualization script).

## Usage
1. **Prepare Dataset**: Run the first script with the correct folder paths to generate `16000clean5dirty.pkl`.
2. **Train Model**: Execute the training script, adjusting paths and hyperparameters as needed.
3. **Evaluate Anomalies**: Use the post-processing script after fixing the unpacking error to analyze results.
4. **Visualize Results**: Run the visualization script with appropriate file paths to generate comparison plots.
