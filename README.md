# 🐶 Dog Breed Classification using MobileNetV2 and TensorFlow 2.0

## 1. Problem Statement  
Given an image of a dog, identify its breed from one of 120 possible classes.

Imagine you're at a cafe, you see a dog, take a photo, and want to instantly know the breed—this project makes that possible using deep learning and transfer learning techniques.

---

## 2. Data  
We use the dataset from [Kaggle's Dog Breed Identification Challenge](https://www.kaggle.com/c/dog-breed-identification/data), which contains:

- **Training set**: ~10,000+ labeled dog images  
- **Test set**: ~10,000+ unlabeled dog images  
- **Number of Classes**: 120 different dog breeds

Each image is labeled with a breed name, and file names correspond to the `id` column in the CSV.

---

## 3. Evaluation  
Model predictions are evaluated using **log loss** on a CSV file submission with prediction probabilities for each dog breed per image.  
See the [Kaggle evaluation page](https://www.kaggle.com/c/dog-breed-identification/overview/evaluation) for more details.

---

## 4. Features and Approach  

- 📷 **Image Classification**: Classify unstructured image data into 120 dog breeds  
- 🧠 **Transfer Learning**: Uses a pre-trained **MobileNetV2** model saved in TensorFlow's `SavedModel` format  
- 🔄 **Custom Keras Layer**: Implements a custom `TFSMLayer` to wrap the MobileNetV2 `SavedModel` for feature extraction  
- ⚙️ **TensorFlow 2.x**: Built and trained using TensorFlow 2.x and Keras  
- 🐾 **120 Dog Breeds**: Multi-class classification with one-hot encoded labels  
- 📊 **Submission-ready**: Outputs predictions in a Kaggle-compatible CSV format  

---

## 5. Model Architecture

The classifier uses the following architecture:


Input Image (224x224x3)
→ TFSMLayer (wraps MobileNetV2 SavedModel)
→ Dense (softmax, output size = number of breeds)

- **Base Model:** MobileNetV2 (`SavedModel`)  
- **Final Layer:** `Dense(len(unique_breeds), activation='softmax')`

All predictions are probability distributions across the 120 classes.

---

## 6. Getting Started

### 🔧 Prerequisites

Make sure you have the following:

- Python 3.7+  
- TensorFlow 2.x  
- TensorFlow Hub  
- Pandas, NumPy, Matplotlib  
- Kaggle CLI (optional, to download the dataset)

### 📦 Installation

Install required packages:


pip install tensorflow tensorflow-hub pandas numpy matplotlib

## 7. Usage

### 🏗️ Building the Model

```python
from model import create_model

SAVED_MODEL_PATH = "models/mobilenet_v2"
OUTPUT_SHAPE = 120  # or len(unique_breeds)

model = create_model(
    input_shape=(224, 224, 3),
    output_shape=OUTPUT_SHAPE,
    saved_model_path=SAVED_MODEL_PATH
)
model.summary()
```
### 🧠 Training
```
model.fit(train_dataset, epochs=10, validation_data=val_dataset)
```

## 8. Results & Improvements

✅ Achieved competitive log loss using MobileNetV2 as a feature extractor.

🧪 Potential improvements:

* Fine-tuning the base model (unfreeze layers)
* Data augmentation and regularization
* Use ensemble models for better accuracy

---

## 9. References

* Kaggle Dog Breed Identification Challenge
* TensorFlow Hub
* MobileNetV2 paper
* Transfer Learning with TensorFlow
