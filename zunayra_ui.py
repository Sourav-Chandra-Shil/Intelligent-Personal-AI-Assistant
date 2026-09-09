"""
Zunayra — Chat UI version (Gradio, ChatGPT/Gemini-style) — STABLE/LITE

Run this script to open a chat page in your browser:

    python zunayra_ui.py

The browser will open automatically at a local URL (e.g. http://127.0.0.1:7860)
with a ChatGPT-style chat interface.

REQUIRED (install once):
    pip install gradio transformers torch accelerate tavily-python

NOTE ON THIS VERSION:
This is a lightweight/stable version. On hardware where the model is
partially offloaded to disk (slow), making multiple model calls per
message (nickname detection, memory extraction, search classification)
can exhaust memory and crash the process. This version makes ONLY ONE
model call per user message — it still reads existing memory.json to
personalize answers, but does not run extra classifier calls to update
memory automatically. This trades some automatic features for speed
and stability.
"""

import json
import os
import re

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================================================================
# CONFIG
# =====================================================================

MODEL_PATH = "."  # folder containing your model files

MEMORY_FILE = "memory.json"

# Lower this further (e.g. 120) if responses are still too slow.
MAX_NEW_TOKENS = 150

# =====================================================================
# MODEL LOAD (runs once, before the UI starts)
# =====================================================================

print("Loading model, please wait...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
)
print("Model loaded successfully!")

# =====================================================================
# MEMORY (read-only in this lite version — no extra classifier calls)
# =====================================================================

DEFAULT_MEMORY_STRUCTURE = {
    "profile": {
        "name": None,
        "preferred_name": None,
        "birthday": None,
        "country": None,
        "hobbies": [],
        "skills": [],
        "siblings": [],
    },
    "relationships": {"friends": [], "family": []},
    "pets": [],
    "preferences": {
        "favorite_food": None,
        "favorite_player": None,
        "favorite_color": None,
        "favorite_game": [],
        "favorite_movie": [],
        "favorite_book": [],
    },
    "projects": [],
    "programming_languages": [],
    "interests": [],
    "episodic_memories": [],
    "conversation_summaries": [],
    "ai_profile": {"nickname": None},
}


def load_memory():
    memory = {}
    for key, value in DEFAULT_MEMORY_STRUCTURE.items():
        memory[key] = value.copy() if isinstance(value, (dict, list)) else value

    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                loaded_data = json.load(f)
            for category, default_values in DEFAULT_MEMORY_STRUCTURE.items():
                if category in loaded_data:
                    if isinstance(default_values, dict):
                        for field, default_value in default_values.items():
                            memory[category][field] = loaded_data[category].get(field, default_value)
                    elif isinstance(default_values, list):
                        memory[category] = loaded_data[category]
            return memory
        except json.JSONDecodeError:
            return memory
    return memory


def build_memory_text(memory_dict):
    lines = []

    def format_section(name, data):
        if isinstance(data, dict):
            entries = [(k, v) for k, v in data.items() if v not in (None, [], {})]
            if entries:
                lines.append(f"{name.replace('_', ' ').title()}:")
                for k, v in entries:
                    v_text = ", ".join(map(str, v)) if isinstance(v, list) else v
                    lines.append(f"  - {k.replace('_', ' ').title()}: {v_text}")
        elif isinstance(data, list) and data:
            lines.append(f"{name.replace('_', ' ').title()}: {', '.join(map(str, data))}")

    for category, content in memory_dict.items():
        if category == "ai_profile":
            continue
        format_section(category, content)

    return "\n".join(lines) if lines else "No specific facts remembered yet."


# =====================================================================
# CHAT FUNCTION (single model call, grounding rules included)
# =====================================================================


def _as_text(content):
    """Normalize Gradio message content (which can sometimes be a list) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(part) for part in content)
    return str(content)


def chat(user_message, history=None):
    memory_dict = load_memory()
    ai_nickname = memory_dict.get("ai_profile", {}).get("nickname")
    name_part = f" named {ai_nickname}" if ai_nickname else ""

    memory_text = build_memory_text(memory_dict)
    ai_identity_line = (
        f"Your own name (the AI's nickname, NOT the user's name) is: {ai_nickname}\n"
        if ai_nickname
        else ""
    )

    system_prompt = (
        f"You are a helpful personal AI assistant{name_part}, texting with a friend. "
        "Talk like a real person — casual, warm, a little brief. Not like a customer "
        "service bot, not like an encyclopedia.\n\n"
        "Rules for how you talk:\n"
        "- Always respond to what the user ACTUALLY wrote, never a memorized, "
        "example, or earlier response.\n"
        "- Short, plain sentences. No markdown headers, no bullet lists, no bold "
        "titles — unless the user asks for a breakdown or steps.\n"
        "- Never say 'As an AI assistant' or 'I don't have access to personal "
        "information' — just answer like a person would.\n"
        "- Don't repeat the user's question back before answering.\n"
        "- No emojis or exclamation marks unless the user's tone is upbeat.\n\n"
        "GROUNDING RULES (critical — never break these):\n"
        "- Only state a fact about the user (age, name, pets, preferences, etc.) if "
        "it actually appears in the conversation history or in the known facts "
        "given to you below. Never invent, assume, or guess a fact that isn't there.\n"
        "- If the user challenges you, corrects you, or says you're wrong or "
        "'making things up', do NOT get defensive and do NOT invent an explanation "
        "for yourself unless that explanation is actually visible in the conversation "
        "history. Look back at the real history: if you can point to the exact message "
        "where they told you the fact, say that plainly. If you genuinely can't find "
        "it, admit you may have gotten it wrong and ask them to clarify.\n"
        "- Never fabricate a justification, a memory, or a reason for your own past "
        "behavior. If unsure, say you're not sure, instead of a confident-sounding story.\n"
        "- When in doubt between sounding confident and being accurate, always "
        "choose being accurate.\n\n"
        "If the user is trying to give you (the AI) a nickname — e.g. 'I'll call you "
        "X' or 'your name is now X' — acknowledge it warmly in your reply (e.g. "
        "'Got it, I'm X now!'). Do not confuse this with the user stating a fact "
        "about themselves (like their own age or name).\n\n"
        f"{ai_identity_line}Known facts about the user (this is about the user, NOT about you):\n"
        f"{memory_text}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        for m in history:
            messages.append({"role": m["role"], "content": _as_text(m["content"])})
    messages.append({"role": "user", "content": user_message})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=0.3,
            top_p=0.9,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return response.strip()


# =====================================================================
# GRADIO CHAT UI
# =====================================================================


def user_submit(user_message, chat_history):
    """Runs instantly on Enter — shows the user's message right away."""
    if not user_message.strip():
        return "", chat_history
    chat_history = chat_history + [{"role": "user", "content": user_message}]
    return "", chat_history


def bot_respond(chat_history):
    """Runs after user_submit. Makes exactly ONE model call for stability."""
    if not chat_history or chat_history[-1]["role"] != "user":
        return chat_history

    user_message = _as_text(chat_history[-1]["content"])
    model_history = chat_history[:-1]

    try:
        response = chat(user_message, history=model_history)
    except Exception as e:
        response = f"Sorry, something went wrong while generating a response: {e}"

    chat_history = chat_history + [{"role": "assistant", "content": response}]
    return chat_history


with gr.Blocks(title="Zunayra") as demo:
    gr.Markdown("# 🤖 Zunayra — Personal AI Assistant")

    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="Type your message here...", show_label=False)
    clear = gr.Button("Clear Chat")

    msg.submit(user_submit, [msg, chatbot], [msg, chatbot]).then(bot_respond, chatbot, chatbot)
    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), inbrowser=True)
