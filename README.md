# CIFAR-10 Image Classification with TensorFlow/Keras

This repository contains a deep learning project for classifying images from the CIFAR-10 dataset (10 classes) using a convolutional neural network (CNN) implemented in TensorFlow/Keras. The project includes scripts for training the model, saving and visualizing training history, and performing inference on custom images.

## Features
- Trains a CNN model on the CIFAR-10 dataset with data augmentation.
- Uses batch normalization, dropout, and learning rate reduction for robust training.
- Saves training history as a CSV file for analysis.
- Visualizes training/validation accuracy and loss.
- Supports inference on custom 32x32 RGB images.
- Includes pre-trained model weights for immediate use.

## Repository Contents
- `train_cifar10.py`: Script to load CIFAR-10, train the CNN, and save the model.
- `save_history.py`: Script to save training history as a CSV file.
- `load_history.py`: Script to load and display training history from CSV.
- `visualize_history.py`: Script to plot training/validation accuracy and loss.
- `predict_image.py`: Script to predict the class of a custom image.
- `model.h5`: Pre-trained model weights for the CNN (10 classes, 32x32 RGB input).

**Note**: The CIFAR-10 dataset is not included. You must download it from [Kaggle](https://www.kaggle.com/datasets/quanbk/cifar10/data) (see [Data Preparation](#data-preparation)).

## Prerequisites
- **Python**: 3.8 or higher
- **Libraries**:
  - `tensorflow`
  - `numpy`
  - `pandas`
  - `matplotlib`
- **Hardware**:
  - Optional: GPU for faster training (CPU works but is slower).

Install dependencies:
```bash
pip install tensorflow numpy pandas matplotlib
```

## Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/CIFAR10-Classification.git
   cd CIFAR10-Classification
   ```

2. **Create Data Directory**:
   - Create a folder named `data` to store the CIFAR-10 dataset:
     ```bash
     mkdir data
     ```

## Data Preparation
1. **Download CIFAR-10 Dataset**:
   - Download the CIFAR-10 dataset from [Kaggle](https://www.kaggle.com/datasets/quanbk/cifar10/data).
   - You’ll need a Kaggle account and the Kaggle API or manual download:
     - **Using Kaggle API**:
       ```bash
       pip install kaggle
       kaggle datasets download -d quanbk/cifar10 -p data
       unzip data/cifar10.zip -d data
       ```
     - **Manual Download**:
       - Visit the [Kaggle dataset page](https://www.kaggle.com/datasets/quanbk/cifar10/data).
       - Download the dataset and extract it to the `data` folder.
   - Ensure the `data` folder contains:
     ```
     data/
     ├── data_batch_1
     ├── data_batch_2
     ├── data_batch_3
     ├── data_batch_4
     ├── data_batch_5
     ├── test_batch
     ```

2. **Verify Dataset**:
   - Confirm the `data` folder has the batch files (`data_batch_1` to `data_batch_5` and `test_batch`).
   - The dataset includes 50,000 training images and 10,000 test images (32x32 RGB, 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

## Training the Model
1. **Run the Training Script**:
   - Execute `train_cifar10.py` to train the CNN:
     ```bash
     python train_cifar10.py
     ```
   - The script:
     - Loads CIFAR-10 training and test data from the `data` folder.
     - Applies data augmentation (rotation, shifts, flips).
     - Trains a CNN with batch normalization, dropout, and learning rate reduction.
     - Saves the trained model as `model.h5`.

2. **Training Details**:
   - **Model**: CNN with four convolutional layers, two max-pooling layers, and dense layers.
   - **Input**: 32x32 RGB images.
   - **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
   - **Optimizer**: Adam.
   - **Loss**: Categorical cross-entropy.
   - **Callbacks**: Reduce learning rate on plateau.
   - **Epochs**: 50.
   - Expected validation accuracy: ~75–85% with the provided configuration.

3. **Output**:
   - The trained model is saved as `model.h5`.
   - Training history (accuracy, loss) is printed.

## Saving and Analyzing Training History
1. **Save History**:
   - Run `save_history.py` to save training history as a CSV:
     ```bash
     python save_history.py
     ```
   - Outputs `cifar10_training_history.csv` with columns: `accuracy`, `val_accuracy`, `loss`, `val_loss`.

2. **Load History**:
   - Run `load_history.py` to display the history:
     ```bash
     python load_history.py
     ```
   - Prints the first few rows of the CSV.

3. **Visualize History**:
   - Run `visualize_history.py` to plot accuracy and loss:
     ```bash
     python visualize_history.py
     ```
   - Displays two plots: training/validation accuracy and loss over epochs.

## Inference on Custom Images
1. **Run the Prediction Script**:
   - Use `predict_image.py` to classify a custom 32x32 RGB image:
     ```bash
     python predict_image.py
     ```
   - Modify the script to point to your image file:
     ```python
     img = load_img('path/to/your/image.png', target_size=(32, 32))
     ```
   - The script:
     - Loads and preprocesses the image (resize to 32x32, normalize).
     - Predicts the class using `model.h5`.
     - Prints the predicted class (0–9, corresponding to CIFAR-10 classes).

2. **Using the Pre-Trained Model**:
   - The included `model.h5` can be used for inference without retraining.
   - Ensure `model.h5` is in the same directory as `predict_image.py`.

## Notes
- **Dataset Access**: If you prefer not to download CIFAR-10 manually, modify `train_cifar10.py` to use TensorFlow’s built-in loader:
  ```python
  from tensorflow.keras.datasets import cifar10
  (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()
  train_data = train_data.astype('float32') / 255.0
  test_data = test_data.astype('float32') / 255.0
  train_labels = to_categorical(train_labels, 10)
  test_labels = to_categorical(test_labels, 10)
  ```
- **Overfitting**: If validation accuracy is much lower than training accuracy, increase dropout rates or add more augmentation.
- **Performance**: Training on CPU is slow; use a GPU if available.
- **Custom Images**: Ensure test images are 32x32 RGB and relevant to CIFAR-10 classes for meaningful predictions. For example, use images of animals or vehicles matching the dataset classes.
- **History CSV**: Run `save_history.py` before `load_history.py` or `visualize_history.py`.
- **Kaggle Environment**: The code was developed in a Kaggle environment. If running locally, ensure the `data` folder matches the Kaggle dataset structure.

## Contributing
Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m "Add YourFeature"`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Built with [TensorFlow/Keras](https://www.tensorflow.org/).
- Uses the [CIFAR-10 dataset](https://www.kaggle.com/datasets/quanbk/cifar10/data) from Kaggle.
- Inspired by image classification tutorials.