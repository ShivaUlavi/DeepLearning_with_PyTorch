import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import streamlit as st
import torch

# ---------------------------------------------------------------------------
# Path & Working Directory Resolution
# Ensures relative paths inside src/translation_utils.py ('../data/eng-fra.txt')
# and modelV2 checkpoints resolve cleanly regardless of where streamlit is run.
# ---------------------------------------------------------------------------
CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURRENT_FILE_DIR).lower() == "app":
    ROOT_DIR = os.path.dirname(CURRENT_FILE_DIR)
    os.chdir(CURRENT_FILE_DIR)
else:
    ROOT_DIR = CURRENT_FILE_DIR
    app_subdir = os.path.join(ROOT_DIR, "app")
    if os.path.isdir(app_subdir):
        os.chdir(app_subdir)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.translation_utils import (
    SOS_token,
    EOS_token,
    MAX_LENGTH,
    device,
    normalizeString,
    prepareData,
    tensorFromSentence,
)
from src.models import EncoderRNN, AttnDecoderRNN

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Seq2Seq Neural Machine Translation",
    page_icon="🇫🇷",
    layout="wide",
)

st.title("🇫🇷 English-to-French Neural Machine Translation")
st.markdown(
    """
    **Architecture:** Sequence-to-Sequence (Seq2Seq) GRU with **Bahdanau Attention**  
    **Hardware Profile:** Edge-trained locally on an NVIDIA GTX 1050 Ti (4GB VRAM) across 150,000 iterations.
    """
)


# ---------------------------------------------------------------------------
# Cached System & Checkpoint Loader
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading vocabularies and trained Seq2Seq + Attention weights...")
def load_translation_system(hidden_size=512):
    input_lang, output_lang, pairs = prepareData("eng", "fra", reverse=False)

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, max_length=MAX_LENGTH).to(device)

    encoder_ckpt = os.path.join(ROOT_DIR, "modelV2", "best_encoder.pth")
    decoder_ckpt = os.path.join(ROOT_DIR, "modelV2", "best_decoder.pth")

    encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device, weights_only=True))
    decoder.load_state_dict(torch.load(decoder_ckpt, map_location=device, weights_only=True))

    encoder.eval()
    decoder.eval()

    return input_lang, output_lang, pairs, encoder, decoder


# ---------------------------------------------------------------------------
# Inference & Attention Extraction
# ---------------------------------------------------------------------------
def evaluate_with_attention(sentence, input_lang, output_lang, encoder, decoder, max_length=MAX_LENGTH):
    normalized = normalizeString(sentence)
    if not normalized:
        raise ValueError("Please enter a valid English sentence containing letters.")

    input_words = normalized.split(" ")
    if len(input_words) >= max_length:
        raise ValueError(
            f"Input sequence is {len(input_words)} tokens long. "
            f"Please keep sentences under {max_length} tokens (including punctuation)."
        )

    missing_words = [w for w in input_words if w not in input_lang.word2index]
    if missing_words:
        raise KeyError(", ".join(missing_words))

    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, normalized)
        input_length = input_tensor.size(0)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        encoder_hidden = encoder.init_hidden()

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei].unsqueeze(0), encoder_hidden
            )
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_attentions[di] = decoder_attention.data.cpu()

            _, topi = decoder_output.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.view(1, 1).detach()

        source_tokens = input_words + ["<EOS>"]
        attn_matrix = decoder_attentions[: len(decoded_words), : len(source_tokens)].numpy()

        return normalized, source_tokens, decoded_words, attn_matrix


def render_matplotlib_heatmap(source_tokens, decoded_words, attn_matrix, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(max(6, len(source_tokens) * 0.8), max(4, len(decoded_words) * 0.6)))
    cax = ax.matshow(attn_matrix, cmap=cmap)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(source_tokens)))
    ax.set_yticks(range(len(decoded_words)))
    ax.set_xticklabels(source_tokens, rotation=45, ha="left", fontsize=11)
    ax.set_yticklabels(decoded_words, fontsize=11)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlabel("Source English Tokens", fontsize=11, labelpad=10)
    ax.set_ylabel("Predicted French Tokens", fontsize=11, labelpad=10)

    for i in range(attn_matrix.shape[0]):
        for j in range(attn_matrix.shape[1]):
            val = attn_matrix[i, j]
            text_color = "black" if val > 0.5 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Load System & Sidebar Metadata
# ---------------------------------------------------------------------------
try:
    input_lang, output_lang, pairs, encoder, decoder = load_translation_system(hidden_size=512)
except Exception as e:
    st.error(f"Failed to load model checkpoints or dataset: {e}")
    st.stop()

with st.sidebar:
    st.header("⚙️ Model Specifications")
    st.markdown(
        f"""
        - **Encoder:** 1-Layer GRU (`hidden_size=512`)
        - **Decoder:** Bahdanau Attention GRU (`hidden_size=512`)
        - **Max Sequence Length:** `{MAX_LENGTH}` tokens
        - **English Vocabulary:** `{input_lang.n_words:,}` tokens
        - **French Vocabulary:** `{output_lang.n_words:,}` tokens
        - **Filtered Corpus:** `{len(pairs):,}` pairs
        - **Training Regimen:** Randomized Grid Search (150k iters, `lr=0.005`, SGD + Teacher Forcing `p=0.5`)
        - **Best Validation Loss:** `2.5225`
        - **Compute Device:** `{str(device).upper()}`
        """
    )
    st.divider()
    cmap_choice = st.selectbox(
        "Attention Heatmap Color Scale",
        options=["viridis", "magma", "plasma", "Blues", "YlGnBu"],
        index=0,
    )


# ---------------------------------------------------------------------------
# Main Interactive Translation Interface
# ---------------------------------------------------------------------------
preset_examples = [
    "Custom Input...",
    "just get back to work .",
    "i am cold .",
    "are you sure ?",
    "i am very hungry .",
    "she is very tall .",
    "we are ready .",
]

col_input, col_preset = st.columns([2, 1])

with col_preset:
    selected_preset = st.selectbox("Quick-Test Preset Sentences:", options=preset_examples, index=1)

default_text = "just get back to work ." if selected_preset == "Custom Input..." else selected_preset

with col_input:
    user_input = st.text_input(
        "Enter an English sentence (punctuation is automatically normalized):",
        value=default_text,
    )

translate_clicked = st.button("🚀 Translate & Visualize Attention", type="primary")

if translate_clicked or user_input:
    try:
        normalized_input, source_tokens, decoded_words, attn_matrix = evaluate_with_attention(
            user_input, input_lang, output_lang, encoder, decoder
        )

        clean_french = " ".join([w for w in decoded_words if w != "<EOS>"])

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.info(f"**Normalized English Input:**\n\n`{normalized_input}`")
        with res_col2:
            st.success(f"**Predicted French Translation:**\n\n`{clean_french}`")

        st.subheader("🔍 Interactive Bahdanau Attention Alignment")
        st.caption(
            "Each cell $(i, j)$ shows the softmax attention weight $\\alpha_{ij}$ assigned by the Decoder "
            "to English token $j$ while generating French token $i$."
        )

        # Build labeled DataFrame for inspection
        row_labels = [f"{idx + 1}: {w}" for idx, w in enumerate(decoded_words)]
        col_labels = [f"{idx + 1}: {w}" for idx, w in enumerate(source_tokens)]
        attn_df = pd.DataFrame(attn_matrix, index=row_labels, columns=col_labels)

        tab_interactive, tab_matplotlib, tab_inspector = st.tabs(
            ["📊 Interactive Heatmap", "🖼️ Publication Figure (Matplotlib)", "🎯 Token-by-Token Inspector"]
        )

        with tab_interactive:
            if HAS_PLOTLY:
                fig_plotly = px.imshow(
                    attn_matrix,
                    x=col_labels,
                    y=row_labels,
                    color_continuous_scale=cmap_choice,
                    text_auto=".2f",
                    aspect="auto",
                    labels=dict(
                        x="Source English Tokens",
                        y="Predicted French Tokens",
                        color="Attention Weight",
                    ),
                )
                fig_plotly.update_xaxes(side="top")
                fig_plotly.update_layout(height=max(420, len(decoded_words) * 55))
                st.plotly_chart(fig_plotly, use_container_width=True)
            else:
                st.markdown("**Hover & Sort Interactive Attention Matrix:**")
                st.dataframe(
                    attn_df.style.background_gradient(cmap=cmap_choice, axis=None).format("{:.3f}"),
                    use_container_width=True,
                )

        with tab_matplotlib:
            fig_mpl = render_matplotlib_heatmap(
                source_tokens, decoded_words, attn_matrix, cmap=cmap_choice
            )
            st.pyplot(fig_mpl, use_container_width=False)

        with tab_inspector:
            selected_step = st.selectbox(
                "Select a generated French token to inspect its source attention distribution:",
                options=list(range(len(decoded_words))),
                format_func=lambda idx: f"Step {idx + 1}: '{decoded_words[idx]}'",
            )
            step_weights = attn_matrix[selected_step]
            top_src_idx = int(np.argmax(step_weights))
            st.markdown(
                f"When generating **`{decoded_words[selected_step]}`**, the decoder focused most strongly on "
                f"English token **`{source_tokens[top_src_idx]}`** (weight = `{step_weights[top_src_idx]:.3f}`)."
            )
            step_chart_df = pd.DataFrame(
                {"English Token": col_labels, "Attention Weight": step_weights}
            ).set_index("English Token")
            st.bar_chart(step_chart_df)

    except KeyError as e:
        st.error(
            f"**Out-of-Vocabulary (OOV) Token Detected:** `{e}` is not present in the 12,890-word English training dictionary."
        )
    except ValueError as e:
        st.warning(str(e))
