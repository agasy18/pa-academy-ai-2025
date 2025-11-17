## 🧭 Proposed Simplified Structure (with “zero-stress” learning flow)

### **Module 1: Deep Learning Fundamentals**

**Duration:** 1 weeks
**Goal:** Understand how a neural network *learns* and *predicts*.

**Topics (Simplified & Interactive):**

Lesson 1:
* From linear / logistic regression to neural networks (when regression fails)
* Neuron = linear model + activation (ReLU, sigmoid) on top
* Layers and depth as compositions that bend decision boundaries (circles / spirals)
* Forward pass intuition through a small 2D PyTorch model

**Lesson 1 Homework:**
* Recreate one of the Playground experiments in the notebook (circles or spirals), try at least two different architectures, 
* find the the most depth network with least units
* find the the least depth network with least units

Lesson 2:
* Activations and there properties and best use cases
* Loss functions and their properties and best use cases
* Oprimizers and their properties and best use cases
* MNIST dataset, training, evaluation, and inference in notebook, ...etc

**Lesson 2 Homework:**
* Train a model on Fashion MNIST dataset, evaluate the model
(if you have GPU or high-end CPU, CIFAR-10 is better)
* Find the best architecture, optimizer, learning rate, and epochs
* Add visualiztions as it done for Lesson 2 Pytorch MNIST notebook
* Visualize the training and validation loss and accuracy curves
* Infer the model on a new image shot by a camera with the best model

**Hands-on:**
👉 Generate a non-linear 2D dataset (circles or spirals) and train a small PyTorch network
👉 Visualize decision regions and relate them to the TensorFlow Playground experiments

---

### **Module 2: Computer Vision I — Seeing with CNNs**

**Duration:** 2 weeks
**Goal:** Understand how CNNs “see” patterns and edges.

**Topics:**

* Intuition of Convolution (interactive kernel visualization)
* Filters, Strides, Pooling — visual demos
* CNN Architectures (LeNet → VGG → ResNet — intuition only)
* Regularization (Dropout), Normalization (BatchNorm)
* Data Augmentation made fun (flip, rotate, color-jitter demos)

**Hands-on:**
👉 Train a CNN on MNIST or Fashion-MNIST
👉 Experiment with augmentation & observe accuracy changes

---

### **Module 3: Computer Vision II — Beyond Classification**

**Duration:** 2 weeks
**Goal:** Expand from image *classification* to *detection* and *segmentation*.

**Topics:**

* Receptive Fields — why deeper = more context
* Transfer Learning (reusing pre-trained CNNs)
* Object Detection & Segmentation (intro-level YOLO / UNet intuition)
* Evaluation Metrics (Precision, Recall, IoU — explained with visuals)

**Hands-on:**
👉 Use transfer learning (e.g. ResNet on a new dataset)
👉 Try a pretrained YOLOv5 or UNet model on custom images

---

### **Module 4: NLP — Teaching Machines to Read**

**Duration:** 2 weeks
**Goal:** Understand how AI processes and generates text.

**Topics:**

* From Words to Numbers (Tokenization, Embeddings, Word2Vec)
* Text Classification (sentiment analysis example)
* Sequence Models (LSTM — intuition, visualize time steps)
* Transformers & Attention (interactive demo)
* Pre-trained Models (BERT & GPT — how they understand context)

**Hands-on:**
👉 Build a simple sentiment classifier
👉 Use a pretrained transformer to answer questions or summarize text

---

### **Module 5: Multi-Modal AI — Seeing + Reading Together**

**Duration:** 2 weeks
**Goal:** Show how AI connects vision and language.

**Topics:**

* What is Multi-Modal Learning? (e.g., “AI that describes what it sees”)
* CLIP concept (match text & image embeddings)
* Image Captioning (encoder-decoder idea)
* Visual Question Answering — combine text + vision inputs

**Hands-on:**
👉 Use CLIP to find similar images from text
👉 Build a simple image captioning app (with pre-trained models)

---

### 🔁 Rhythm & Pedagogy

Each week:

* **Lesson 1 (2h):** Concept + interactive demo (with minimal math)
* **Lesson 2 (2h):** Guided coding + mini challenge

Every 2 modules → **mini project** (e.g. “Build your own image classifier” or “Train a chatbot”).

---

Would you like me to:

1. **Add estimated durations per module and per topic** (to fill the “TO BE ADDED” parts)?
2. Or first, refine the **content depth** (what stays vs what gets simplified/removed)?

We can do both, but let’s pick one to focus on next.

---

## 📽️ Slides (Reveal.js)

- Quick start: from repo root run `python3 -m http.server` and visit `http://localhost:8000/slides/`
- Open individual lessons from the index page:
  - `slides/lesson1.html` — Lesson 1: From Regression to Deep Learning (nonlinear 2D data, neurons, layers, activations, decision boundaries)
  - `slides/lesson2.html`
  - `slides/lesson3.html`
  - `slides/lesson4.html`

---

## 📒 Notebooks (Lesson 1)

- Demo: `notebooks/lesson1_demo.ipynb`
  - Mirrors Lesson 1 slides: generate 2D data (circles/spiral), build a small PyTorch model, and plot decision regions.
- Homework: `notebooks/lesson1_homework.ipynb`
  - Guided experiments on layers, activations, and learning-rate sensitivity with short reflection prompts.

Run locally:
- Option A (Jupyter): `jupyter notebook` and open the files in `notebooks/`
- Option B (Lab): `jupyter lab`
- If packages are missing, use `pip install torch scikit-learn matplotlib`
