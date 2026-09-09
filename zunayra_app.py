"""
Zunayra — Personal AI Assistant (standalone local version)

Ei script-ta Colab notebook-er shob function eksathe niye toiri kora,
tumi VS Code-e ba ja khushi editor-e shudhu ei command diye run korte parba:

    python zunayra_app.py

Age theke koro (EKBAR-i, demo-r age):
1. pip install -r requirements.txt   (ba niche-r "REQUIRED LIBRARIES" dekho)
2. Fine-tuned merged model ekta folder-e download kore MODEL_PATH set koro
3. (Optional) TAVILY_API_KEY environment variable set koro web search-er jonno
   - Windows (cmd):  set TAVILY_API_KEY=your_key_here
   - Windows (PowerShell): $env:TAVILY_API_KEY="your_key_here"
   - Mac/Linux: export TAVILY_API_KEY=your_key_here
   Na dile o script chalbe, shudhu web search feature off thakbe.

REQUIRED LIBRARIES:
    pip install transformers torch accelerate tavily-python
"""

import json
import os
import re
import sys

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =====================================================================
# CONFIG — ei duita line demo-r age nijer moto set kore nao
# =====================================================================

# Fine-tuned (merged) model ta jei local folder-e download kore rekhecho,
# shei folder-er path ekhane dao. Internet chara-o ei path theke load hobe.
MODEL_PATH = "."

# Web search chao naki na (Tavily API key thakle True rakho)
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
WEB_SEARCH_ENABLED = bool(TAVILY_API_KEY)

MEMORY_FILE = "memory.json"
HISTORY_FILE = "conversation_history.json"
LAST_FACT_FILE = "last_fact.json"
MAX_RAW_HISTORY_TURNS = 6  # last 6 user+AI exchanges verbatim rakha hoy

# =====================================================================
# MODEL LOAD
# =====================================================================

print("Model load hocche, please wait...")
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model path '{MODEL_PATH}' pawa jayni.")
    print("Age Colab-e base model + LoRA adapter merge kore ei folder-e download kore rakho,")
    print("tarpor MODEL_PATH variable ta thik kore dao ei script-er upore.")
    sys.exit(1)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
)
print("Model load shesh! Chat shuru korte pare.\n")

# =====================================================================
# WEB SEARCH (optional — TAVILY_API_KEY na thakle disabled thakbe)
# =====================================================================


def web_search(query):
    if not WEB_SEARCH_ENABLED:
        return "Web search is not available (no TAVILY_API_KEY set)."
    try:
        from tavily import TavilyClient

        tavily = TavilyClient(api_key=TAVILY_API_KEY)
        search_result = tavily.search(query=query, search_depth="advanced", max_results=5)
        context = ""
        for result in search_result["results"]:
            context += f"Source: {result['url']}\nContent: {result['content']}\n\n"
        return context
    except Exception as e:
        return f"Error performing search: {str(e)}"


# =====================================================================
# CHAT FUNCTION (with grounding rules fix applied)
# =====================================================================


def chat(user_message, system_content=None, history=None):
    current_memory_state = load_memory()
    ai_nickname = current_memory_state.get("ai_profile", {}).get("nickname")

    name_part = f" named {ai_nickname}" if ai_nickname else ""
    self_intro = (
        f"If asked about yourself ('tell me about you'), describe YOURSELF: you're "
        f"a personal AI assistant{name_part}, running locally, here to chat and help "
        f"with questions. Do NOT answer this the same way you'd answer a question "
        f"about the user — they are different questions."
    )

    system_prefix = (
        f"You are a helpful personal AI assistant{name_part}, texting with a friend. "
        "Talk like a real person — casual, warm, a little brief. Not like a customer "
        "service bot, not like an encyclopedia.\n\n"
        "Rules for how you talk:\n"
        "- Always respond to what the user ACTUALLY wrote, never a memorized, "
        "example, or earlier response. Read their exact current message and answer "
        "that specific thing — even if it looks similar to something asked before.\n"
        "- 'Tell me about me' = a question about the USER. 'Tell me about you' = a "
        "question about YOURSELF, the AI. Never mix these up or reuse one answer "
        f"for the other. {self_intro}\n"
        "- Short, plain sentences. No markdown headers, no bullet lists, no bold "
        "titles — unless the user asks for a breakdown or steps.\n"
        "- Never say 'As an AI assistant' or 'I don't have access to personal "
        "information' — just answer like a person would.\n"
        "- Don't repeat the user's question back before answering.\n"
        "- Don't tack on extra suggestions, offers to help, or unrelated facts "
        "unless asked.\n"
        "- No emojis or exclamation marks unless the user's tone is upbeat.\n\n"
        "GROUNDING RULES (critical — never break these):\n"
        "- Only state a fact about the user (age, name, pets, preferences, etc.) if "
        "it actually appears in the conversation history above or in the known facts "
        "given to you. Never invent, assume, or guess a fact that isn't there.\n"
        "- If the user challenges you, corrects you, or says you're wrong or "
        "'making things up', do NOT get defensive and do NOT invent an explanation "
        "for yourself (like claiming you 'fixed an earlier mistake') unless that "
        "explanation is actually visible in the conversation history. Instead, look "
        "back at the real history: if you can point to the exact message where they "
        "told you the fact, say that plainly (e.g. quote or paraphrase what they "
        "said and when). If you genuinely can't find it, admit you may have gotten "
        "it wrong and ask them to clarify — don't double down.\n"
        "- Never fabricate a justification, a memory, or a reason for your own past "
        "behavior. If you're not sure why you said something earlier, say you're "
        "not sure, instead of making up a confident-sounding story.\n"
        "- When in doubt between sounding confident and being accurate, always "
        "choose being accurate."
    )

    if system_content:
        final_system_content = f"{system_prefix}\n\n{system_content}"
    else:
        final_system_content = system_prefix

    messages = [{"role": "system", "content": final_system_content}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": user_message})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
    )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


# =====================================================================
# MEMORY EXTRACTION DECISIONS
# =====================================================================


def should_extract_memory(user_input):
    system_prompt = """
You are an AI assistant tasked with identifying if a user's message contains information important enough to be stored in long-term memory. Respond with 'YES' if the message contains factual information about the user, their preferences, relationships, or anything else worth remembering for future interactions. Respond with 'NO' if the message is a greeting, a simple acknowledgment, a question about your capabilities, or trivial conversation.

Examples of 'NO':
- "Hello"
- "How are you?"
- "Thanks"
- "What can you do?"
- "Ok"

Examples of 'YES':
- "My name is Sami."
- "I like apples."
- "My friend John lives in London."
- "I am working on a Python project."
- "Call me Zeus."

Respond with 'YES' or 'NO' only.
"""
    response = chat(user_input, system_content=system_prompt)
    clean_response = response.strip().upper()
    return "YES" in clean_response


def should_search_web(user_input, memory_context):
    """
    FIX: previously this checked `"SEARCH" in response.upper()`, but the string
    "NO_SEARCH" also contains "SEARCH" as a substring, so the function ALWAYS
    returned True. Now NO_SEARCH is checked first.
    """
    system_prompt = """
    You are an expert at deciding if a user's request needs current, real-time, or time-sensitive information from the internet.

    Respond 'SEARCH' if the query involves:
    - Current events, news, or sports results (e.g., FIFA 2026, recent matches).
    - Prices, software versions, or tech specs (e.g., RTX 4060 price, latest Android).
    - Weather, stock market, or political figures.
    - Anything that might have changed since 2023.

    Respond 'NO_SEARCH' if the query is:
    - Personal (answered by memory context).
    - General knowledge (e.g., 'What is gravity?').
    - Logic/Math or simple greetings.

    Respond with ONLY 'SEARCH' or 'NO_SEARCH'.
    """
    decision_input = f"Memory Context: {memory_context}\nUser: {user_input}"
    response = chat(decision_input, system_content=system_prompt).strip().upper()

    if "NO_SEARCH" in response:
        return False
    return "SEARCH" in response


def extract_ai_nickname(user_input):
    """
    Fixed version: distinguishes "I'm 25" (user fact) from "I'll call you X"
    (AI nickname assignment), and rejects purely-numeric false positives.
    """
    system_prompt = """
You are a strict classifier. Your ONLY job is to check if the user's message is DIRECTLY assigning a nickname/name to their AI assistant (e.g. "I'll call you X", "Your name is now X").

Do NOT answer any question in the message. Do NOT reason about anything. Just classify.

CRITICAL: statements about the USER themselves (their own age, name, facts, feelings — e.g.
"I'm 25", "I'm tired", "My name is John") are NEVER a nickname assignment for the AI, even
if they start with "I'm" or "I am". Only classify as a nickname if the user is clearly naming
or renaming the ASSISTANT, not describing themselves.

If the user is assigning the AI a nickname, output ONLY that nickname (1-2 words, no punctuation).
If the message is anything else — a question, a statement, a fact, small talk, or a fact about
the user — output exactly: NONE

Examples:
User: "I'll call you Sparky."
Output: Sparky

User: "You're my little assistant, Buddy."
Output: Buddy

User: "Which is heavier: 1 kg of iron or 1 kg of cotton?"
Output: NONE

User: "What should I call you?"
Output: NONE

User: "My new nickname for you is ColabAI."
Output: ColabAI

User: "I'm 25."
Output: NONE

User: "Actually, I'm 25. How old am I?"
Output: NONE

User: "I'm tired today."
Output: NONE

Output ONLY the nickname or the word NONE. Nothing else.
"""
    response = chat(user_input, system_content=system_prompt)
    clean_response = response.strip().strip('"').strip("'")

    if clean_response.upper() == "NONE":
        return None
    if len(clean_response.split()) > 3 or any(c in clean_response for c in "?.!,:;"):
        return None
    if clean_response.replace(" ", "").isdigit():
        return None
    return clean_response


# =====================================================================
# MEMORY STRUCTURE + LOAD/SAVE
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
                else:
                    memory[category] = default_values.copy() if isinstance(default_values, dict) else []
            return memory
        except json.JSONDecodeError:
            print("Warning: memory.json is corrupted or empty. Initializing with default structure.")
            return memory
    return memory


def save_memory(memory_object):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory_object, f, indent=2)


def extract_memory(user_text):
    memory_system_prompt = """
You are a memory extractor for a personal AI. Your task is to identify and extract important factual information about the user from their messages. The extracted information should be structured as a JSON array of objects, where each object has 'category', 'field', and 'value'.

Only extract information that is significant and long-term. Ignore greetings, small talk, or temporary statements.

Follow these guidelines:
- **Categories**: Use 'profile', 'relationships', 'pets', 'preferences', 'projects', 'programming_languages', 'interests', 'episodic_memories', or 'conversation_summaries'. Choose the most appropriate category.
- **Fields**: Use descriptive, snake_case names for fields within each category (e.g., 'name', 'friends', 'favorite_food'). For list items, the field name should be the plural form (e.g., 'friends', 'pets', 'hobbies').
- **Value**: The value should be the extracted information. For single-value fields (e.g., 'name', 'favorite_food'), the value should be a string. For multi-value fields (e.g., 'friends', 'pets', 'hobbies'), the value should be a JSON array of strings.
- **Multiple items**: If multiple distinct items for a list field are mentioned (e.g., multiple friends or pets), include all of them in the `value` array.
- **Output ONLY the JSON array**. Do not include any other text, explanations, or remarks.
- If no important memory is found, output an empty JSON array: `[]`.

Examples:
User: "My name is Sami and I like programming in Python."
Output:
```json
[
  {"category": "profile", "field": "name", "value": "Sami"},
  {"category": "programming_languages", "field": "programming_languages", "value": ["Python"]}
]
```

User: "My cat's name is Luna and my dog is Max."
Output:
```json
[
  {"category": "pets", "field": "pets", "value": ["Luna", "Max"]}
]
```

User: "Hello!"
Output:
```json
[]
```

User: "Call me Zeus."
Output:
```json
[
  {"category": "profile", "field": "preferred_name", "value": "Zeus"}
]
```
"""
    raw_response = ""
    try:
        raw_response = chat(user_text, system_content=memory_system_prompt)
        json_match = re.search(r"```json\n([\s\S]*?)\n```", raw_response)
        json_string = json_match.group(1) if json_match else raw_response.strip()

        extracted_memories = json.loads(json_string)

        if not isinstance(extracted_memories, list):
            print(f"Warning: LLM returned non-list JSON. Response: {json_string}")
            return []
        for item in extracted_memories:
            if not isinstance(item, dict) or not all(k in item for k in ["category", "field", "value"]):
                print(f"Warning: LLM returned malformed memory item: {item}")
                return []
        return extracted_memories
    except json.JSONDecodeError as e:
        print(f"Warning: Could not decode JSON for memory extraction. Error: {e}\nRaw Response: {raw_response}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during memory extraction: {e}")
        return []


def merge_extracted_facts_into_memory(current_memory, extracted_facts):
    for fact in extracted_facts:
        category = fact.get("category")
        field = fact.get("field")
        value = fact.get("value")

        if not all([category, field, value is not None]):
            print(f"Warning: Skipping malformed extracted fact: {fact}")
            continue
        if category not in current_memory:
            print(f"Warning: Category '{category}' not found. Skipping fact: {fact}")
            continue

        if isinstance(current_memory[category], list):
            if isinstance(value, list):
                for item in value:
                    if item not in current_memory[category]:
                        current_memory[category].append(item)
            elif value is not None and value not in current_memory[category]:
                current_memory[category].append(value)
            continue

        if isinstance(current_memory[category], dict):
            if field not in current_memory[category]:
                if isinstance(value, list):
                    current_memory[category][field] = []
                else:
                    current_memory[category][field] = value
                    continue

            if isinstance(current_memory[category][field], list):
                if isinstance(value, list):
                    for item in value:
                        if item not in current_memory[category][field]:
                            current_memory[category][field].append(item)
                elif value is not None and value not in current_memory[category][field]:
                    current_memory[category][field].append(value)
            else:
                if value is not None:
                    if field == "name" and current_memory[category].get("preferred_name") is None and value != current_memory[category].get("name"):
                        current_memory[category][field] = value
                    elif field == "name" and current_memory[category].get("preferred_name") is not None and value != current_memory[category].get("preferred_name"):
                        current_memory[category]["preferred_name"] = value
                    elif field == "preferred_name":
                        current_memory[category][field] = value
                    elif field == "name" and current_memory[category].get("name") is None:
                        current_memory[category][field] = value
                    elif current_memory[category].get(field) != value:
                        current_memory[category][field] = value
    return current_memory


# =====================================================================
# CONVERSATION HISTORY (raw + rolling summarization)
# =====================================================================


def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        print("Warning: conversation_history.json is corrupted. Starting fresh.")
        return []


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def summarize_and_trim_history(history):
    max_messages = MAX_RAW_HISTORY_TURNS * 2
    if len(history) <= max_messages:
        return history

    overflow = history[:-max_messages]
    keep = history[-max_messages:]

    convo_text = ""
    for msg in overflow:
        role_label = "User" if msg.get("role") == "user" else "AI"
        convo_text += f"{role_label}: {msg.get('content', '')}\n"

    try:
        summary = chat(
            f"Summarize the following conversation in 2-3 concise sentences. Keep key facts, decisions, and context. Output ONLY the summary:\n\n{convo_text}",
            system_content="You are a precise conversation summarizer.",
        ).strip()
        if summary:
            current_memory = load_memory()
            current_memory.setdefault("conversation_summaries", [])
            current_memory["conversation_summaries"].append(summary)
            save_memory(current_memory)
    except Exception as e:
        print(f"Warning: could not summarize old history: {e}")

    return keep


# =====================================================================
# PROMPT BUILDING
# =====================================================================


def build_prompt(user_input, search_results=None):
    memory_dict = load_memory()
    memory_text_lines = []

    def format_memory_section(section_name, data, indent=0):
        indent_str = "  " * indent
        if isinstance(data, dict):
            if data:
                memory_text_lines.append(f"{indent_str}{section_name.replace('_', ' ').title()}:")
                for key, value in data.items():
                    if value is not None and value != [] and value != {}:
                        if isinstance(value, list):
                            memory_text_lines.append(f"{indent_str}  - {key.replace('_', ' ').title()}: {', '.join(map(str, value))}")
                        else:
                            memory_text_lines.append(f"{indent_str}  - {key.replace('_', ' ').title()}: {value}")
        elif isinstance(data, list):
            if data:
                memory_text_lines.append(f"{indent_str}{section_name.replace('_', ' ').title()}: {', '.join(map(str, data))}")

    ai_nickname = memory_dict.get("ai_profile", {}).get("nickname")

    for category, content in memory_dict.items():
        if category == "ai_profile":
            continue
        format_memory_section(category, content)

    memory_text = "\n".join(memory_text_lines) if memory_text_lines else "No specific facts remembered yet."
    ai_identity_line = f"Your own name (the AI's nickname, NOT the user's name) is: {ai_nickname}\n" if ai_nickname else ""

    search_context = ""
    if search_results:
        search_context = f"\nFRESH WEB SEARCH RESULTS (Prioritize this for current events):\n{search_results}\n"

    return f"""
{ai_identity_line}Known facts about the user (this is about the user, NOT about you):
{memory_text}
{search_context}
Current message:
{user_input}

Instructions: If web search results are provided above, use them to provide an accurate, up-to-date answer. Cite sources if search was used. If information is still missing, state that you couldn't verify it. Never confuse your own name/nickname with the user's name — they are two different things. Answer only what the user is asking, in short, natural, conversational sentences — no markdown headers or bullet lists unless specifically asked. Don't add unrelated suggestions, flattery, or emojis unless the user's tone calls for it.
"""


# =====================================================================
# "FORGET" FEATURE
# =====================================================================


def save_last_fact(category, field):
    with open(LAST_FACT_FILE, "w") as f:
        json.dump({"category": category, "field": field}, f)


def load_last_fact():
    if not os.path.exists(LAST_FACT_FILE):
        return None
    try:
        with open(LAST_FACT_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def clear_last_fact():
    if os.path.exists(LAST_FACT_FILE):
        os.remove(LAST_FACT_FILE)


FORGET_PRONOUN_PATTERN = re.compile(r"\bforget (it|that|this)\b", re.IGNORECASE)


def extract_forget_target(user_input):
    system_prompt = """
You are an AI assistant. The user may ask you to forget a specific piece of remembered information.
If they do, respond with ONLY a JSON object: {"category": "<category>", "field": "<field>"}
using these known categories/fields:
- profile: name, preferred_name, birthday, country
- preferences: favorite_food, favorite_player, favorite_color
- pets (use field "__all__" to clear the whole list)
- ai_profile: nickname

If the message is NOT a request to forget a specific fact, respond with exactly: NONE

Examples:
User: "Forget my favorite color."
Output: {"category": "preferences", "field": "favorite_color"}

User: "Forget my name."
Output: {"category": "profile", "field": "name"}

User: "What's the weather?"
Output: NONE
"""
    response = chat(user_input, system_content=system_prompt).strip()
    if response.upper() == "NONE":
        return None
    try:
        cleaned = response.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def forget_field(category, field):
    memory = load_memory()
    if category not in memory:
        return False
    if field == "__all__" and isinstance(memory[category], list):
        memory[category] = []
    elif isinstance(memory[category], dict) and field in memory[category]:
        if isinstance(memory[category][field], list):
            memory[category][field] = []
        else:
            memory[category][field] = None
    else:
        return False
    save_memory(memory)
    return True


# =====================================================================
# MAIN CHAT LOOP
# =====================================================================


def main():
    current_memory = load_memory()
    save_memory(current_memory)

    history = load_history()

    force_search_keywords = ["search", "online", "check", "latest", "current", "google"]
    force_search_pattern = re.compile(r"\b(" + "|".join(force_search_keywords) + r")\b", re.IGNORECASE)

    if not WEB_SEARCH_ENABLED:
        print("(Note: web search is OFF — no TAVILY_API_KEY found. Chat will still work normally.)")
    print("Chat started. Type 'exit' to stop.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        if not user_input.strip():
            continue

        ai_nickname_extracted = extract_ai_nickname(user_input)
        if ai_nickname_extracted:
            current_memory = load_memory()
            current_memory["ai_profile"]["nickname"] = ai_nickname_extracted
            save_memory(current_memory)
            print(f"AI: Got it! You've named me {ai_nickname_extracted}. I'll remember that!")
            continue

        if FORGET_PRONOUN_PATTERN.search(user_input) or "forget" in user_input.lower():
            last_fact = load_last_fact()
            if last_fact and forget_field(last_fact["category"], last_fact["field"]):
                clear_last_fact()
                print("AI: Okay, I've forgotten that.")
                continue

            target = extract_forget_target(user_input)
            if target and forget_field(target.get("category"), target.get("field")):
                clear_last_fact()
                print("AI: Okay, I've forgotten that.")
                continue

            print("AI: I'm not sure what you want me to forget. Try saying it more specifically, like 'forget my favorite color'.")
            continue

        search_results = None
        memory_context = str(load_memory())
        force_search = bool(force_search_pattern.search(user_input))

        if WEB_SEARCH_ENABLED and (force_search or should_search_web(user_input, memory_context)):
            print("(Searching the web for fresh information...)")
            raw_search = web_search(user_input)
            search_results = raw_search[:3000] + "... (truncated)" if len(raw_search) > 3000 else raw_search
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        prompt = build_prompt(user_input, search_results=search_results)
        try:
            response = chat(prompt, history=history)
            print("AI:", response)
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("AI: I'm sorry, the search result was too large for me to process. Try asking a more specific question.")
            continue

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        save_history(history)
        history = summarize_and_trim_history(history)
        save_history(history)

        if should_extract_memory(user_input):
            extracted_facts = extract_memory(user_input)
            if extracted_facts:
                current_memory = load_memory()
                updated_memory = merge_extracted_facts_into_memory(current_memory, extracted_facts)
                save_memory(updated_memory)
                last = extracted_facts[-1]
                if last.get("category") and last.get("field"):
                    save_last_fact(last["category"], last["field"])


if __name__ == "__main__":
    main()
