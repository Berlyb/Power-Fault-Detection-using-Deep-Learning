# Power-Fault-Detection-using-Deep-Learning

Powerline Components and Faults Dataset

https://huggingface.co/datasets/docmhvr/powerline-components-and-faults

This dataset contains images of powerline components and faults, intended for use in deep learning applications. The images can be analyzed using Convolutional Neural Networks (CNNs) to classify powerline components (such as poles, transformers, and wires) and detect faults (like broken parts or damage). By training a model on this dataset, automated systems can assist in powerline maintenance and inspection by identifying and diagnosing issues quickly and efficiently.

Object Detection Training and Validation
This project focuses on training an object detection model using PyTorch and validating its performance on a small subset of a larger dataset. The model is trained on 10% of the available data to accommodate limited computational resources. This project uses a custom dataset and employs basic data loading, training, and validation techniques with a progress bar for real-time tracking.
Requirements
- Python 3.x
- PyTorch 1.x or later
- tqdm
- torchvision
- numpy
- random

To install the necessary libraries, use the following command:
```bash
pip install torch torchvision tqdm numpy
```
Project Structure
```bash
project/
├── train.py                # Main training script
├── validate.py             # Validation script
├── dataset/                # Custom dataset folder
│   ├── train_dataset/      # Training data
│   └── val_dataset/        # Validation data
├── README.md              # Project documentation
└── requirements.txt        # List of required Python packages
```
How to Use
1. **Prepare the Dataset**: Ensure that the dataset is structured with images and corresponding annotations for object detection.
2. **Set Up the Model**: The model is defined using a standard architecture, typically a pre-trained Faster R-CNN or another suitable model for object detection.
3. **Run the Training**: The training loop will run for the specified number of epochs. It utilizes a random 10% subset of the dataset for training.

To train the model, run:
```bash
python train.py
```
4. **Run the Validation**: After training, validate the model on a separate 10% subset of the dataset.

To run validation, execute:
```bash
python validate.py
```
Key Features
- **Subset Training**: The model trains on a random 10% subset of the dataset to manage computational load.
- **Progress Tracking**: Training and validation loops display a progress bar with loss values for each batch, providing real-time feedback.
- **Evaluation**: Separate validation logic is implemented to assess the model's performance on a validation set.
- **Custom Collate Function**: Handles variable-sized images during batch processing.
Training Hyperparameters
- **Batch Size**: 8
- **Epochs**: 5 (can be adjusted)
- **Learning Rate**: Adjustable within the training script

Future Improvements
- Increase the dataset size for training and validation.
- Optimize hyperparameters (learning rate, batch size, etc.).
- Implement data augmentation for improved model generalization.
- Fine-tune pre-trained models for better performance.

