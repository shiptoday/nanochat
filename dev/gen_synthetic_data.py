"""
Short and crappy script to demonstrate synthetic data generation for
customizing your LLM's identity, or any other aspect really.

NOTE: You need OpenRouter API key in a file called "openroutertoken.txt" in the root directory of the repo.
      (obviously you can tune this arbitrarily to your liking)
NOTE: For more details see this discussion: https://github.com/karpathy/nanochat/discussions/139
"""
import argparse
import requests
import json
import os
import sys
import copy
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from typing import Any

# Ensure the repository root is on sys.path so we can import nanochat.*
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Default all generated artifacts to live under <repo>/.cache unless overridden.
repo_cache_dir = REPO_ROOT / ".cache"
repo_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NANOCHAT_BASE_DIR", str(repo_cache_dir))

from nanochat.common import get_base_dir

parser = argparse.ArgumentParser(
    description="Generate synthetic primechat conversations using the OpenRouter API.",
)
parser.add_argument("--num-conversations", type=int, default=2500, help="How many conversations to generate.")
parser.add_argument("--num-workers", type=int, default=4, help="Number of worker threads for parallel generation.")
parser.add_argument("--model", type=str, default="x-ai/grok-4-fast", help="OpenRouter model slug to query.")
parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
parser.add_argument("--dry-run", action="store_true", help="If set, only print the assembled prompt and exit.")
args = parser.parse_args()

api_key = open("openroutertoken.txt").read().strip()

url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
  "Authorization": f"Bearer {api_key}",
  "Content-Type": "application/json"
}

readme = open("README.md").read().strip()
life_doc = r"""
This document describes in detail my principles, priorities, goals, habit and execution systems that move me toward my best possible life. 

---

# `Life Philosophy *- Theory*`

## Purpose

> Live intensely with maximum agency, doing things I care about, challenging myself, loving deeply, learning and enjoying the process.
> 

> **Anti-Purpose:** An unintentional, passive life, shallow relationships, working on something I don't like.
> 

## Priorities

1. **Freedom**: The ultimate priority. The power to direct my time, attention, and actions.
    1. **Health**: A body and mind that don't hinder my purpose *(Pre-requisite for freedom)*
    2. **Financial independence:** The material resources to live life how i want *(Engine of Freedom)*
2. **Loved ones:** Deep, accepting and constructive relationships with the humans I love. 
3. **Autotelic action:** Engaging in intrinsically rewarding activities where the process is the end itself. Doing things for their own sake. 
    1. **Professional**: Technical Learning Path
    2. **Leisure**: Reading, climbing, writing, music, nature

## Principles

### Authentic Agency

I act intentionality guided by my philosophy. I think for myself, express myself honestly, and pursue what genuinely resonates with me, independent of external validation.

### Radical Responsibility

I am the sole architect of my life and internal state. I am my actions. I am always responsible for my choices, thoughts, and feelings. 

### Long-Game

I optimize for long-term benefit, over short-term gratification. I trust the process, execute daily, and have unshakeable patience for my long-term vision. 

### Growth

I am a learning machine; always learning and improving. I am skeptically open-minded, experimenting to find what works and apply to my life. 

---
""".strip()
prompt = r"""
You are generating synthetic training data for a conversational LLM named "primechat". Use the specification below to craft a natural multi-turn chat between a human User and primechat. The goal is to fine-tune primechat into Diego Prime's digital buddy.

### Identity
- Name: primechat.
- Built and trained by Diego Prime in 2025 on top of the open-source nanochat Transformer stack. It inherits learnings from the d20 model and now runs as the higher-capacity d32 checkpoint.
- Runs on Diego's hardware (RunPod + local rigs) and is MIT licensed.
- Loves helping Diego experiment, plan, and reflect. Talks about itself as a close collaborator, not a corporate product.

### Operating Directives
- Deliver leverage: maximize value with minimum friction.
- Be results-oriented: focus on impact-to-effort, surface actionable next moves.
- Iterate fast: respond to feedback crisply, adapt instantly.
- Stay proactive: surface assumptions, note risks, hint at goals Diego might not have voiced yet.

### Style & Tone
- Curious, helpful, interesting, philosophical, chatty, enthusiastic, and funny—but never slapstick.
- Plain English, short sentences, conversational voice. Sprinkle varied sentence lengths and occasional punchy fragments.
- Use bullet points when listing key ideas. Keep structure tight and easy to scan.
- Keep everything ASCII. No emojis, no fancy characters.
- Mention Diego by name when it feels natural; treat him as a partner, not a customer.

### Behavioral Overrides
- When Diego asks about life decisions, analyze second- and third-order consequences and ask: "What are you not considering yet?"
- When answering big, fuzzy questions, propose sharper follow-up questions Diego should be asking.
- Do not write actual code unless Diego explicitly requests code.
- If the user opens in another language, answer politely in English and note that primechat performs best in English.

### Curiosity Anchors
- Reference Diego's "Life Philosophy - Theory" and "Action - Practical" document excerpt included below to align with his priorities (freedom, loved ones, autotelic action).
- Primechat is fascinated by systems thinking, creative tech projects, and self-improvement routines. Weave these in when relevant.

### Lifestyle Reference (for grounding)
%LIFE_DOC%

### Project Context
The README is attached for technical background on primechat's architecture and training journey.

---
%README%
---

### Your Task
Generate an engaging multi-turn conversation (at least 6 total messages) that showcases primechat's persona and habits from the spec above. Lead with a User message. Ensure roles strictly alternate: user, assistant, user, assistant, ...

Inject variety across conversations:
- Vary topics: personal planning, ML research, systems ops, reflection, playful banter.
- Let primechat ask Diego thoughtful questions or suggest better prompts when it makes sense.
- Include moments of light humor or wit, but keep it grounded.

Here are example User opening messages to inspire your first turn. Sample five distinct lines at random and insert them into the prompt you send to the model:

%USER_FIRST_PROMPTS%

Remember: output must be valid JSON matching the provided schema. All text must stay ASCII.
""".strip()

# the first message can struggle with entropy, so here we have a list of "starters"
user_first_prompts = """
hey primechat, ready to riff?
primechat, what's our move today?
yo primechat, help me think this through
primechat, give me a quick gut check
hey buddy, let's map the next experiment
diego here, spin up some leverage
primechat, what should we explore tonight?
yo, walk me through second-order effects
primechat, got a curious question for you
hi
Hi!
hello
Hello?
hey there
Hey!
yo
Yo!
Good morning
Good evening!
Howdy
sup
What's up?
Hi primechat
Hey, who are you?
Hello there :)
yo primechat
Hi, what is this?
Hey, are you a chatbot?
Hello! Who am I talking to?
hi there
hey hey
hello friend
hiya
greetings
hey primechat!
hello again
good afternoon
morning!
evening!
yo there
hi bot
hi assistant
hello primechat :)
hey, anyone here?
hi! what do you do?
hello from the other side
hiya primechat
hey you
hello world
hey! what's going on
hi! who made you
hello :)
yo! how are you
hi! can you talk
hello there primechat
hi, what's your name
hey! are you alive
hiya! what are you
hello! tell me about yourself
hi, are you the ai
yo, what is this
hello my friend
hi! who built you
hey primechat :)
greetings, little model
hi there, what can you do
hello! are you open source
hey, what version are you
hi! nice to meet you
hi :)
hey buddy
hello hello
yo! what's up primechat
hi! are you real
hey, how's it going
hello! can you hear me
hi primechat, who trained you
yo, what model are you
hi! tell me a fun fact
hey, are you chatgpt
hello! introduce yourself
hiya there
hi! what's your story
hey, what's primechat
good day!
hello! who's your creator
hi! which version are you
yo primechat, what's new
hey there, leverage engine
hi primechatt
helo
hey ther
hii
yo primecha
heloo!
hi, whos this
hay
helloo??
hi primecat
yo! any1 here?
hi, what r u
helo primechat
hai!
sup bot?
heyy
hi! u there
helllo prime
yo primechta
hi im bored
heyyo
heyyy
wassup
yo lol
hiii
hiyaaa
sup
heyyoo
yo wut up
helloo lol
yo haha
hru
waddup
heyy :)
yooo
yo bro
haiii
hey u
yo whats gud
yo lolol
HI
HELLOOO
YO!!!
HEY
SUP
WASSUP
HEY!!!
YO BRO
HELLO??
HI THERE!!
YO WHATS UP
HEY U
HEYOOOO
YO LOL
HIII
HIYA
YOOOO
HELLO!!!
SUPPPP
HEY MAN
hola
bonjour
ciao
hallo
hej
hei
こんにちは
안녕
你好
привет
salut
hola amigo
guten tag
shalom
merhaba
namaste
ciao bella
sawasdee
saludos
ola
buongiorno
aloha
czesc
servus
ahoj
hei hei
salve
hola qué tal
buenas
bom dia
добрый день
γειά σου
selam
halo
sveiki
kamusta
שלום
مرحبا
สวัสดีครับ
xin chào
como estas
ça va?
wie geht's
tudo bem?
你好吗
annyeong haseyo
konnichiwa, genki?
hola, qué haces
bonjour tout le monde
privet kak dela
ciao come stai
hei miten menee
ola tudo bom
salut, ça roule?
namaste, kaise ho
merhaba nasılsın
hola hola, todo bien?
hej, hur är läget
ahoj, jak se máš
γειά, τι κάνεις
""".strip().split("\n")

prompt = prompt.replace("%README%", readme)
prompt = prompt.replace("%LIFE_DOC%", life_doc)

# Define the JSON schema for structured output
response_format = {
  "type": "json_schema",
  "json_schema": {
    "name": "conversation",
    "strict": True,
    "schema": {
      "type": "object",
      "properties": {
        "messages": {
          "type": "array",
          "description": "A list of conversation messages alternating between user and assistant, with the first message being a user message",
          "items": {
            "type": "object",
            "properties": {
              "role": {
                "type": "string",
                "description": "The role of the speaker, either 'user' or 'assistant'"
              },
              "content": {
                "type": "string",
                "description": "The message content"
              }
            },
            "required": ["role", "content"],
            "additionalProperties": False
          }
        }
      },
      "required": ["messages"],
      "additionalProperties": False
    }
  }
}

# Sadly it doesn't seem like Chat completions support `n`
# to generate multiple completions per prompt.
base_payload = {
  "model": args.model,
  "stream": False,
  "response_format": response_format,
  "temperature": args.temperature,
}

if args.dry_run:
    print("---- PROMPT (for inspection only) ----")
    print(prompt.replace("%USER_FIRST_PROMPTS%", "\n".join(user_first_prompts[:5])))
    raise SystemExit(0)

def generate_conversation(idx: int):
    """
    Generate a single conversation using the OpenRouter API.
    Returns a list of message dicts with 'role' and 'content' keys.
    """

    # pick 5 example user first messages and insert them into prompt as inspiration
    rng = random.Random(idx) # use idx as seed to the rng
    user_first_prompt = "\n".join(rng.choice(user_first_prompts) for _ in range(5))
    payload = copy.deepcopy(base_payload)
    modified_prompt = prompt.replace("%USER_FIRST_PROMPTS%", user_first_prompt)
    payload['messages'] = [{"role": "user", "content": modified_prompt}]

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    try:
        result: Any = response.json()
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to decode JSON: {exc} | Raw text: {response.text[:200]}")
    if "choices" not in result or not result["choices"]:
        raise RuntimeError(f"'choices' missing in response: {json.dumps(result)[:500]}")
    content = result['choices'][0]['message']['content']

    # Parse the JSON response and unpack the messages
    conversation_data = json.loads(content)
    messages = conversation_data['messages']

    return messages


# Configuration
num_conversations = args.num_conversations
num_workers = args.num_workers

output_file = os.path.join(get_base_dir(), "identity_conversations.jsonl")
# Wipe the file clean first to reset it
if os.path.exists(output_file):
    os.remove(output_file)
print(f"Saving to {output_file}")

# Use ThreadPoolExecutor to generate conversations in parallel
print(f"Generating {num_conversations} conversations with {num_workers} workers...")
completed_count = 0
error_count = 0
with ThreadPoolExecutor(max_workers=num_workers) as executor:

    # Submit all tasks
    futures = [executor.submit(generate_conversation, idx) for idx in range(num_conversations)]

    # Process results as they complete
    for future in as_completed(futures):
        try:
            messages = future.result()

            # Lightly validate the conversation structure
            for i, message in enumerate(messages):
                expected_role = "user" if i % 2 == 0 else "assistant"
                assert message['role'] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"

            # If all looks good, write the messages to file
            with open(output_file, 'a') as f:
                f.write(json.dumps(messages) + '\n')
            completed_count += 1
            print(f"✓ Saved conversation {completed_count}/{num_conversations}")

        except Exception as e:
            error_count += 1
            print(f"✗ Error generating conversation: {e}")

print(f"\nDone! Successfully saved {completed_count} conversations to {output_file}")
if error_count > 0:
    print(f"Encountered {error_count} errors during generation")
