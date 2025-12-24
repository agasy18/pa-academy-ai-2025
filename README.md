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

**Duration:** 1 weeks  
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

* From Words to Numbers (tokenization, vocabularies, embeddings, CBOW & Skip-Gram / Word2Vec, and visualizing embeddings with PCA / TensorFlow Embedding Projector)
* Text Classification (sentiment analysis example)
* Sequence Models (LSTM — intuition, visualize time steps)
* Transformers & Attention (interactive demo)
* Pre-trained Models (BERT & GPT — how they understand context)

***Lesson 6:***
* From words to numbers: simple tokenization, building vocabularies, and mapping tokens to integer IDs.
* One-hot vs dense word embeddings; using `nn.Embedding` in PyTorch.
* Distributional semantics and Word2Vec intuition: CBOW and Skip-Gram.
* Visualizing word embeddings with PCA and the TensorFlow Embedding Projector (Word2Vec 10K).
* Simple embedding-based sentiment classifier with mean pooling over word vectors.

**Hands-on / Notebook:**
👉 Work through `notebooks/lesson6_word_embeddings.ipynb` to build tokenization and vocab, experiment with `nn.Embedding`, explore pretrained word vectors and “word math”, train tiny Skip-Gram/CBOW models, and visualize embeddings with PCA.

***Lesson 6 Homework:***
* Implement a small tokenizer and vocabulary builder for a text dataset (e.g., movie reviews).
* Train a simple sentiment classifier in PyTorch using `nn.Embedding` + mean pooling + linear layer.
* Inspect nearest neighbors in your learned embedding space (e.g., around “good”, “bad”, “great”, “terrible”) and briefly describe what you see.
* Optional: train a tiny Skip-Gram or CBOW model on your corpus and compare its nearest neighbors to those from pretrained GloVe/Word2Vec.

***Lesson 7:***
* Recap: From embeddings to sequences — why order matters.
* RNNs & LSTMs: sequential processing and its limitations.
* The Attention Mechanism: intuition as a "soft dictionary lookup" (Query, Key, Value).
* Self-Attention: attending to yourself — capturing long-range dependencies in one step.
* Multi-Head Attention: multiple heads for different relationship types.
* The Transformer Architecture: encoder-decoder blocks, positional encoding, LayerNorm + residuals.
* BERT (Bidirectional Encoder): Masked Language Modeling, understanding context from both sides.
* GPT (Autoregressive Decoder): next-token prediction, causal masking, text generation.
* BERT vs GPT: when to use which (understanding vs generation tasks).
* Using pre-trained models with Hugging Face Transformers (pipelines for sentiment, fill-mask, generation).

**Hands-on / Notebook:**
👉 Work through `notebooks/lesson7_transformers_attention.ipynb` to implement self-attention from scratch, visualize attention weights, and experiment with pre-trained BERT and GPT models using Hugging Face.

**Interactive Demos:**
* [exBERT](https://exbert.net/) — Visualize BERT attention heads.
* [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Best visual guide to Transformers.

***Lesson 7 Homework:***
* Use Hugging Face pipelines to perform sentiment analysis on a custom set of texts.
* Experiment with different prompts for GPT-2 text generation and analyze how prompt wording affects outputs.
* Fine-tune a small BERT model (e.g., DistilBERT) on a classification task using the Hugging Face Trainer.
* Visualize attention patterns from different layers and heads — describe what different heads seem to focus on.

---

### **Module 5: Multi-Modal AI — Seeing + Reading Together**

**Duration:** 2 weeks  
**Goal:** Show how AI connects vision and language.

***Lesson 8:***
* The vision-language challenge: connecting images and text (different modalities, different representations).
* Pre-CLIP era: CNN + RNN image captioning (Show and Tell, Show Attend and Tell).
* Visual attention mechanism deep dive: computing attention scores, softmax normalization, weighted context vectors.
* Soft attention vs hard attention: why soft attention won (differentiability, end-to-end training).
* The paradigm shift: from task-specific models to shared embedding spaces.
* CLIP architecture: dual encoders (image encoder + text encoder) projecting to a shared space.
* Contrastive learning: InfoNCE loss, batch-based training, cosine similarity, temperature scaling.
* Zero-shot image classification: classifying images using text prompts without task-specific training.
* Image-text retrieval: finding matching images/text via embedding similarity.
* Building on CLIP: modern image captioning (CLIP + LM), text-to-image generation (CLIP + Diffusion).
* CLIP in production: Stable Diffusion, DALL·E 2, and the modern multimodal ecosystem.

**Hands-on / Mini-project:**
👉 Use OpenAI CLIP to perform zero-shot classification on custom images.
👉 Build a text-to-image retrieval system using CLIP embeddings.
👉 Experiment with prompt engineering to improve zero-shot accuracy.
👉 Visualize CLIP embeddings with t-SNE/UMAP to see image-text clustering.

***Lesson 8 Homework:***
* Implement zero-shot classification with CLIP on a custom dataset (e.g., your own photos).
* Compare accuracy with different prompt templates ("a photo of a {}", "an image of a {}", etc.).
* Build a simple image search app: given a text query, return the most similar images from a collection.
* Optional: Fine-tune a CLIP-based captioning model or experiment with CLIP guidance for image generation.

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
  - `slides/lesson5.html` — Lesson 5: Variational Autoencoders & Latent PCA (convolutional VAE for faces, β-ELBO loss, KL divergence, latent PCA controls, and perceptual vs MSE reconstruction losses)
  - `slides/lesson6.html` — Lesson 6: NLP — From Words to Embeddings (tokenization, vocabularies, embeddings, Word2Vec/CBOW/Skip-gram intuition, and a simple sentiment classifier)
  - `slides/lesson7.html` — Lesson 7: Transformers, Attention & Pre-trained Models (self-attention, multi-head attention, Transformer architecture, BERT, GPT, and using pre-trained models)
  - `slides/lesson8.html` — Lesson 8: From Vision–Language Models to CLIP (CNN+RNN captioning, visual attention, CLIP architecture, contrastive learning, zero-shot classification, and text-to-image generation)

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
