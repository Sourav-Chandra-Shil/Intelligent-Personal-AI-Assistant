# 🤖 Zunayra — Personal AI Assistant

Zunayra is a locally-run personal AI assistant built on **Qwen3-4B-Instruct**, featuring persistent long-term memory, **LoRA fine-tuning** on a coding-instruction dataset, and a ChatGPT-style browser chat interface built with **Gradio**.

---

## ✨ Features

- 🧠 **Persistent memory** — remembers facts about the user (name, preferences, pets, projects, relationships, etc.) across conversations, stored in `memory.json`.
- 🎯 **Grounded responses** — the assistant only states facts it can actually verify from memory or conversation history, and admits uncertainty instead of fabricating answers.
- 💻 **Fine-tuned for coding help** — fine-tuned via LoRA on the [CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) dataset for improved programming assistance.
- 🔍 **Optional live web search** — integrates with the Tavily Search API for time-sensitive queries (news, prices, current events).
- 💬 **Two interfaces**:
  - A terminal-based chat loop (`zunayra_app.py`)
  - A ChatGPT-style browser UI built with Gradio (`zunayra_ui.py`)
- 🏷️ **Custom AI nickname** — users can rename the assistant, and it remembers the new name.
- 📝 **Automatic conversation summarization** — older conversation turns are summarized to keep long chats manageable.

---

## 🛠️ Tech Stack

| Component | Tool / Library |
|---|---|
| Base model | Qwen3-4B-Instruct-2507 |
| Fine-tuning | Hugging Face `transformers`, `peft`, `trl` |
| Training environment | Google Colab |
| Dataset | [CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) |
| Chat UI | Gradio |
| Web search (optional) | Tavily API |
| Checkpointing | Google Drive |

---

## 📂 Project Structure

```
ZunayraProject/
├── zunayra_app.py          # Terminal-based chat application
├── zunayra_ui.py            # Gradio browser-based chat UI
├── requirements.txt         # Python dependencies
├── config.json               # Model config (fine-tuned model)
├── model.safetensors         # Fine-tuned model weights
├── tokenizer.json            # Tokenizer files
├── tokenizer_config.json
├── generation_config.json
├── chat_template.jinja
├── memory.json                # Auto-generated: stores user memory
└── conversation_history.json  # Auto-generated: recent chat history
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.11 or 3.12 (recommended; avoid very new/unstable Python releases)
- At least 16GB RAM recommended
- (Windows only) [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### 2. Clone / Download

```bash
git clone <your-repo-url>
cd ZunayraProject
```

> **Note:** The fine-tuned model weights (`model.safetensors`, ~8GB) are not included directly in this repository due to size — see [Model Weights](#-model-weights) below.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Enable web search

```bash
export TAVILY_API_KEY="your_key_here"      # Mac/Linux
setx TAVILY_API_KEY "your_key_here"        # Windows
```

If not set, the assistant runs normally with web search disabled.

### 5. Run

**Browser chat UI (recommended):**
```bash
python zunayra_ui.py
```
This opens a ChatGPT-style interface automatically in your browser at `http://127.0.0.1:7860`.

**Terminal chat:**
```bash
python zunayra_app.py
```

---

## 🧪 Model Fine-Tuning Summary

- **Method:** LoRA (Low-Rank Adaptation) — `r=16`, `alpha=32`, applied to attention projection layers.
- **Dataset:** 5,000-example subset of CodeAlpaca-20k, trained incrementally in 20 chunks of 250 examples to fit free-tier compute limits.
- **Evaluation metric:** Evaluation loss / perplexity on a held-out 5% split (accuracy is not applicable, since this is a generation task, not classification).
- **Checkpointing:** Progress and LoRA weights saved to Google Drive after every chunk, allowing training to resume safely across sessions.
- **Deployment:** LoRA adapter merged into the base model (`merge_and_unload()`) to produce a single standalone fine-tuned model.

For full details, see [`PROJECT_REPORT.md`](./PROJECT_REPORT.md).

---

## 📦 Model Weights

Due to GitHub's file size limits, the fine-tuned model weights are hosted separately:

- 📥 **Download:** *[add your Google Drive / Hugging Face Hub link here]*

After downloading, place `model.safetensors` in the same folder as the other model files (`config.json`, `tokenizer.json`, etc.) before running the app.

---

## ⚠️ Known Limitations

- Performance depends heavily on available RAM/GPU — on machines with limited RAM, parts of the model may be offloaded to disk, slowing down responses.
- The lightweight UI version (`zunayra_ui.py`) prioritizes stability by making a single model call per message, which means automatic memory extraction from casual chat is limited compared to the terminal version.
- Fine-tuning covers general multi-language coding instructions (not limited to one programming language).

---

## 📄 License

*Add your license here (e.g., MIT).*

---

## 🙌 Acknowledgements

- Base model: [Qwen3-4B-Instruct](https://huggingface.co/Qwen)
- Dataset: [CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k) by Sahil Chaudhary
- Built with [Hugging Face Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl), and [Gradio](https://github.com/gradio-app/gradio)
