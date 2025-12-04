## Picsart Academy Deep Learning 2025

[Slides are available here](https://agasy18.github.io/pa-academy-ai-2025)

### **Module 1: Deep Learning Fundamentals**

**Duration:** 1 weeks
**Goal:** Understand how a neural network *learns* and *predicts*.

***Lesson 1:***
* From linear / logistic regression to neural networks (when regression fails)
* Neuron = linear model + activation (ReLU, sigmoid) on top
* Layers and depth as compositions that bend decision boundaries (circles / spirals)
* Forward pass intuition through a small 2D PyTorch model

***Lesson 1 Homework:***
* Recreate one of the Playground experiments in the notebook spirals, try at least two different architectures, 
* find the the most depth network with least units
* find the the least depth network with least units

***Lesson 2:***
* Recap single neuron and activations; extend to common activation functions (ReLU, LeakyReLU, ELU, tanh, sigmoid, softmax, GELU) and their best use cases
* Loss functions for regression (MSE, MAE) and classification (binary and multiclass cross-entropy), including cross-entropy as model “surprise”
* Optimizers (SGD, SGD with momentum, Adam): update rules, intuition, and how they affect training dynamics
* MNIST dataset: training, evaluation, and inference in the `notebooks/lesson2_Pytorch_MNIST.ipynb` notebook

***Lesson 2 Homework:***
* Train a model on Fashion MNIST dataset, evaluate the model
(if you have GPU or high-end CPU, CIFAR-10 is better)
* Find the best architecture, optimizer, learning rate, and epochs
* Add visualiztions as it done for Lesson 2 Pytorch MNIST notebook
* Visualize the training and validation loss and accuracy curves
* Infer the model on a new image shot by a camera with the best model

---

### **Module 2: Computer Vision I — Seeing with CNNs**

**Duration:** 2 weeks


***Lesson 3:***
* Motivation for CNNs vs fully connected networks on image data
* Convolutions, receptive fields, and pooling (shapes, stride, padding)
* Regularization and normalization in CNNs (Dropout, BatchNorm) with PyTorch examples
* LeNet-style architecture and a simple CNN for MNIST (`notebooks/lesson3_Pytorch_MNIST_cnn.ipynb`)

Additional resources (used in Lesson 3 slides):
* [CNN Explainer](https://poloclub.github.io/cnn-explainer/)
* [Convolutional operation visualization](https://medium.com/data-science/gentle-dive-into-math-behind-convolutional-neural-networks-79a07dd44cf9)
* [LeNet visualization video](https://www.youtube.com/watch?v=UxIS_PoVoz8)
* [3D LeNet visualization](https://adamharley.com/nn_vis/cnn/3d.html)
* [FastAI convolutional neural network](https://github.com/fastai/course22p2/blob/master/nbs/07_convolutions.ipynb)


***Lesson 4:***
* Regularization with weight decay
* Data augmentation (flip, rotate, color-jitter, geometric warps) and its effect on overfitting
* Transfer learning with a pre-trained CNN backbone (e.g., xResNet18/ResNet) using data augmentation (`notebooks/lesson4_data_augmentation.ipynb`)
* CNN architectures: LeNet → AlexNet → VGG → Inception → ResNet → xResNet
* Depthwise separable convolutions and residual blocks in modern CNNs
* Image feature vectors, cosine similarity, and evaluation metrics (accuracy, precision, recall) for similarity search


***Lesson 4 Homework:***
* Build a small Gradio app that takes a custom input image (file upload) and returns the most similar images from a reference set.
* Use a pretrained CNN backbone (e.g., xResNet18) to extract a feature vector for each image.
* Compute cosine similarity between the query feature vector and all reference features, and display the top-k matches with similarity scores.
* Optionally, compare at least two different backbones (e.g., ResNet18 vs xResNet18) and report precision, recall, and accuracy on a small labeled evaluation set.
* Briefly document your design choices (feature extractor, reference dataset, normalization) and discuss limitations of this approach.

---

### **Module 3: Generative Models — Variational Autoencoders**

**Duration:** 2 weeks  
**Goal:** Learn how convolutional VAEs model images, structure a latent space, and expose it through interactive PCA controls.

***Lesson 5:***
* Convolutional VAE architecture for 64×64 RGB faces: encoder/decoder CNN blocks and latent heads (`projects/face_autoencoder/src/model.py`).
* Latent Gaussian variables, the reparameterization trick, and the \(\beta\)-ELBO loss (reconstruction term + KL divergence).
* Training utilities, inline reconstruction visualizations, and different reconstruction losses (MSE, log-MSE, perceptual).
* Collecting latent representations, running PCA, and saving components / variance / latent mean as artifacts.
* Using PCA components to control face generation via Gradio sliders in `projects/face_autoencoder/src/app.py`.
* Comparing VAEs vs regular autoencoders and understanding how KL regularization shapes a smooth latent space.

**Hands-on / Mini-project:**
👉 Run `projects/face_autoencoder/face_autoencoder_training.ipynb` to train the convolutional VAE on faces and export artifacts.  
👉 Launch the interactive app with `python -m projects.face_autoencoder.src.app` and explore how PCA sliders change facial attributes.  
👉 Optionally, modify \(\beta\), architecture depth, or reconstruction loss and compare reconstructions and latent PCA directions.

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

### 🏆 Competitions

**Lesson 1: The Minimalist Network Challenge**

**Goal:** Solve the "Spiral" classification problem (`notebooks/lesson1_demo.ipynb`) with the most efficient architectures possible while maintaining **> 83% validation accuracy**.

**Tasks:**
1. **Baseline:** Implement and train at least two different architectures that exceed 83% accuracy on the spiral dataset.
2. **The "Shallow" Challenge:** Design a network with exactly **2 hidden layers**. Find the configuration with the **fewest total parameters** that still passes the accuracy threshold.
   - *Report:* Hidden unit counts (e.g., 2 -> H1 -> H2 -> 1) and total parameter count.
3. **The "Deep" Challenge:** Design a network with exactly **10 hidden layers**. Find the configuration with the **fewest total parameters** that still passes the accuracy threshold.
   - *Report:* Hidden unit counts and total parameter count.

**Submission:** Submit your notebook with the architecture definitions, training plots, and a summary table of your results.

**Lesson 2: The Fashion MNIST Accuracy Challenge**

**Goal:** Achieve the highest possible test accuracy on the **Fashion MNIST** dataset.

**Rules:**
- **Open Leaderboard:** You may use **any** deep learning architecture (MLP, CNN, ResNet, ViT, etc.) and **any** training technique (data augmentation, learning rate scheduling, ensembles, etc.).
- **Constraint:** The model must be trained from scratch (no pre-trained weights from ImageNet, etc.).
- **Evaluation:** Accuracy on the standard Fashion MNIST test set (10,000 images).

**Submission:** Submit a notebook (or script) that:
1. Defines the model and training loop.
2. Trains the model (or loads your best checkpoint).
3. Evaluates and prints the final accuracy on the test set.
4. Includes a brief summary of your approach (what worked, what didn't).

---

## 📽️ Slides (Reveal.js)

- Quick start: from repo root run `python3 -m http.server` and visit `http://localhost:8000/slides/`
- Open individual lessons from the index page:
  - `slides/lesson1.html` — Lesson 1: From Regression to Deep Learning (nonlinear 2D data, neurons, layers, activations, decision boundaries)
  - `slides/lesson2.html` — Lesson 2: How Neural Networks Learn (activations, loss functions, optimizers, and MNIST training in PyTorch)
  - `slides/lesson3.html` — Lesson 3: Convolutional Neural Networks (convolutions, pooling, Dropout/BatchNorm, and a CNN for MNIST)
  - `slides/lesson4.html` — Lesson 4: Data Augmentation, Transfer Learning & CNN Architectures (data augmentation, intro transfer learning with a pre-trained CNN, LeNet → AlexNet → VGG → Inception → ResNet, and object tracking context)

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
