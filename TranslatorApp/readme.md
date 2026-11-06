# 🇫🇷 English-to-French Neural Machine Translation (Seq2Seq + Bahdanau Attention)

An end-to-end Neural Machine Translation (NMT) system built from scratch in PyTorch, featuring a Gated Recurrent Unit (GRU) Encoder-Decoder architecture augmented with Bahdanau (additive) attention.

Designed and trained under local edge-computing constraints (NVIDIA GTX 1050 Ti, 4GB VRAM), this project explores the mechanics of recurrent sequence alignment, teacher forcing, and hyperparameter tuning for low-resource translation.

---

## 📌 Key Highlights & Technical Specifications

* Architecture: 1-layer GRU encoder + Bahdanau attention GRU decoder (hidden_size = 512, dropout = 0.1)
* Dataset: Tatoeba English-French parallel corpus (135,842 raw pairs -> 135,083 filtered pairs at MAX_LENGTH = 20)
* Vocabulary space: 12,890 English tokens | 21,143 French tokens
* Training regimen: randomized grid search across H in {128, 256, 512} and eta in {0.01, 0.005, 0.001} for 150,000 iterations (~1.1 epochs) using SGD
* Best validation loss: 2.5225 (modelV2 with attention) vs. 3.9734 (baseline model without attention)
* Hardware profile: edge-trained locally on an NVIDIA GeForce GTX 1050 Ti (CUDA)

---

## 🧠 Architectural Evolution

### 1. Baseline Seq2Seq (model/)

In the vanilla encoder-decoder architecture, the encoder compresses the whole source sequence into one fixed-size representation. This creates a bottleneck when translating longer sentences.

### 2. Bahdanau Attention Mechanism (modelV2/)

To resolve this bottleneck, the encoder stores outputs at every time step: H = (h_1, ..., h_Tx). At each decoding step t, the decoder computes a dynamic attention distribution alpha_t over the source sequence using the current embedded input e_t and previous decoder hidden state s_{t-1}:

alpha_t = softmax(W_a [e_t ; s_{t-1}] + b_a)

The resulting attention weights are multiplied against the encoder outputs to form a focused context vector c_t:

c_t = sum_{j=1}^{T_x} alpha_{t,j} * h_j

The context vector c_t and embedded token e_t are concatenated, projected through a linear combination layer with ReLU activation, and passed into the decoder GRU to predict the next token distribution over the French vocabulary.

---

## 🔍 Case Study: Visualizing Attention Alignment & Edge-Training Artifacts

The Streamlit interface (`app/app.py`) intercepts the `decoder_attention` tensor at inference time to render interactive heatmaps. These visualizations expose both successful syntactic alignments and classic recurrent failure modes.

### Case 1: Successful Syntactic Alignment ("I love coffee")

When translating high-frequency structures like "I love coffee", the model outputs a grammatically accurate translation: "j aime le cafe". It correctly inserts the required French definite article.

![Successful Translation and Attention Alignment - I love coffee](screenshots/i_love_coffee.png)

---

### Case 2: Attention Collapse & Hallucination Loop ("I have a dog")

When evaluating "I have a dog", the model outputs "j ai un chien chien". Inspecting the attention matrix reveals the cause of this hallucination:

1. At step 1, the decoder attends cleanly to token 1: "I".
2. At steps 2 and 3, the attention weights collapse onto the noun "dog" instead of the verb phrase "have" and article "a".
3. Because the single-head Bahdanau context vector saturates too early, the decoder loses positional tracking and repeats "chien" before reaching the end-of-sequence token.

This behavior illustrates the limits of single-vector recurrent attention and motivates the transition to multi-head self-attention and positional encoding in modern transformer architectures.

![Attention Collapse Case Study - I have a dog](screenshots/i_have_a_dog.png)

---

## 📂 Repository Directory Structure

```text
TranslatorApp/
├── app/
│   └── app.py                                             # Interactive Streamlit UI and attention visualizer
├── data/
│   └── eng-fra.txt                                        # English-French parallel sentence corpus
├── model/
│   ├── best_encoder.pth                                   # Baseline Seq2Seq GRU encoder weights
│   └── best_decoder.pth                                   # Baseline Seq2Seq GRU decoder weights
├── modelV2/
│   ├── best_encoder.pth                                   # Attention-augmented encoder weights (H=512, LR=0.005)
│   └── best_decoder.pth                                   # Bahdanau attention decoder weights (H=512, LR=0.005)
├── notebooks/
│   ├── TransUtilsPrototype.ipynb                          # Data preprocessing and vocabulary indexing prototype
│   ├── Machine_Translation_Seq2Seq_GRU.ipynb              # Baseline Seq2Seq training and grid search
│   └── Machine_translation_Seq2seq_with_Attention.ipynb   # Bahdanau attention training, metric tracking, and evaluation
├── screenshots/
│   ├── i_love_coffee.png                                  # Successful syntactic alignment example
│   └── i_have_a_dog.png                                   # Attention collapse and repetition analysis
├── src/
│   ├── models.py                                          # Modular PyTorch encoder and decoder definitions
│   └── translation_utils.py                               # Unicode normalization, language utilities, and tensor helpers
├── README.MD.md                                          # Legacy file name kept for compatibility
├── readme.md                                            # Recommended lowercase README
└── README.md                                            # Project README reference
```

---

## 🚀 Getting Started & Running Locally

### 1. Clone the repository

```bash
git clone https://github.com/ShivaUlavi/DeepLearning_with_PyTorch.git
cd DeepLearning_with_PyTorch/TranslatorApp
```

### 2. Install dependencies

Ensure you have Python 3.10+ and PyTorch installed. CUDA is recommended for GPU acceleration, although the app automatically falls back to CPU.

```bash
pip install torch streamlit pandas numpy matplotlib plotly
```

### 3. Launch the interactive web application

Run Streamlit from the `TranslatorApp` root directory (or directly inside `app/`):

```bash
streamlit run app/app.py
```

---

## 🔮 Future Engineering Roadmap

* Mini-batch padded sequences: refactor the batch-size-1 training loop to use packed sequences and improve GPU utilization
* Beam search decoding: replace greedy top-k(1) decoding with beam search to reduce repetitive token loops during inference
* Subword tokenization (BPE): replace whitespace and punctuation-based indexing with subword encoding to handle rare words and reduce OOV failures

---

This README was simplified to avoid LaTeX/KaTeX parsing issues while preserving the project documentation and technical details.
