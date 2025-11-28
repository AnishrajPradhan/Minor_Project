# Minor_Project
Here is a professional, research-grade `README.md` file tailored exactly to your project.

You can copy-paste the raw code below directly into your GitHub repository.

-----

# 🍅 Optimizing XSE-TomatoNet: SE vs. GLCAM Attention Comparison

## 📖 Abstract

This project presents a comparative study on **Tomato Leaf Disease Detection** using Explainable AI. We aim to improve the robustness of lightweight Convolutional Neural Networks (CNNs) by integrating advanced attention mechanisms.

We benchmark a baseline **EfficientNetB0 + Squeeze-and-Excitation (SE)** model against a proposed **EfficientNetB0 + Global-Local Context Attention (GLCAM)** model. Our results demonstrate that the **GLCAM** mechanism, adapted from grape disease research, offers superior feature extraction for tomato leaf lesions, achieving **94.5% accuracy** compared to the baseline's 89%.

-----

## 🚀 Key Innovation

Standard CNNs often struggle with "fine-grained" classification (e.g., distinguishing Early Blight from Late Blight).

  * **The Baseline (SE Block):** Focuses only on **Global Channel** weights.
  * **The Proposed (GLCAM Block):** Captures **Global Context** AND **Local Spatial Details** simultaneously.

By transplanting the **GLCAM** architecture from Grape leaf research to the Tomato domain, this project achieves higher accuracy with improved stability.

-----

## 📊 Dataset & Setup

  * **Source:** [Kaggle Tomato Disease Multiple Sources](https://www.kaggle.com/datasets/cookiefinder/tomato-disease-multiple-sources)
  * **Classes:** 11 (Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Spot, Spider Mites, Target Spot, Mosaic Virus, Yellow Curl Virus, Powdery Mildew, Healthy).
  * **Data Split:**
      * **80% Training** (Augmented: Rotation, Flip, Zoom)
      * **10% Validation** (For hyperparameter tuning)
      * **10% Testing** (Held-out for final evaluation)
  * **Preprocessing:** Images resized to `224x224`. **No rescaling** (0-255 range) to preserve EfficientNet input distribution.

-----

## 🏗️ Architecture

We utilize **EfficientNetB0** as the backbone due to its lightweight nature (5.3M parameters).

```mermaid
graph LR
    A[Input Image] --> B[EfficientNet Backbone]
    B --> C{Attention Mechanism}
    C -->|Baseline| D[SE Block]
    C -->|Proposed| E[GLCAM Block]
    D --> F[Global Average Pooling]
    E --> F
    F --> G[Softmax Classifier (11 Classes)]
```

-----

## 📈 Experimental Results

### 1\. Training Performance

The proposed **GLCAM-Net (Green)** demonstrated faster convergence and higher peak accuracy compared to the **Baseline SE-Net (Blue)**.

*(Place your `tomato_innovation_result.png` here)*

### 2\. ROC-AUC Curve

The model achieved near-perfect discrimination with **AUC scores between 0.99 and 1.00** for all classes.

*(Place your `Real_ROC_Curve_Final.png` here)*

### 3\. Confusion Matrix

High diagonal density confirms that the model rarely confuses similar diseases.

*(Place your `Classification_Matrix_Fixed.png` here)*

-----

## 🔍 Explainable AI (Grad-CAM)

To validate the model's decision-making, we employed **Grad-CAM**.

  * **Observation:** The heatmap highlights the **necrotic lesions** (disease spots) in red.
  * **Conclusion:** The model is focusing on the actual pathology, not background noise.

*(Place your `GradCAM_Polished.png` here)*

-----

## 💻 Installation & Usage

### Prerequisites

  * Python 3.8+
  * TensorFlow / Keras
  * NumPy, Matplotlib, Seaborn, OpenCV, Pandas

### Running the Project

1.  **Clone the Repo:**
    ```bash
    git clone https://github.com/yourusername/tomato-disease-glcam.git
    ```
2.  **Install Dependencies:**
    ```bash
    pip install tensorflow matplotlib seaborn opencv-python kagglehub
    ```
3.  **Run the Notebook:**
    Open the Jupyter Notebook (`.ipynb`) and run the cells sequentially. The script automatically downloads the dataset via `kagglehub`.

-----

## 📜 References

This project builds upon the following research:

1.  **Base Paper:** *XSE-TomatoNet: An explainable AI based tomato leaf disease classification method...* (Marouf et al., 2025).
2.  **Innovation Source:** *An explainable lightweight convolutional neural network with global-local context-enhanced channel attention...* (Loganathan et al., 2025).

-----

## 👨‍💻 Author

**[Your Name]**

  * **Student ID:** [Your ID]
  * **Institution:** [Your University]
