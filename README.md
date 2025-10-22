# primechat
![nanochat logo](dev/nanochat.png)

fork of karpathy's nanochat. i'm using it to learn more about llm's arquitecture/training. This is my playground to experiment and learn.  
561M parameters, trained on 11b tokens, developed by andrej karpathy, pre-trained and finetuned by me (Diego Prime)


# Pretraining Details
- I trained my own model with the defaults ran by bash speedrun.sh
- Used runpod.io for training, details below.
- Training took 3hr and cost was $85
- Technical Specs
  * **Parameters:** 560,988,160
  * **Vocab size:** 65,536
  * **Layers:** 20
  * **Model dimension:** 1280
  * **Total batch size:** 524,288
  * **Target tokens:** 11.22B

# Setup
* **Cloud:** RunPod (GPU training)
* **Pod Specs:**
  * 8× H200 SXM (1128 GB VRAM total)
  * 192 vCPU / 2008 GB RAM
  * 80 GB disk
  * Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
  * Cost: ≈ $28.72 / hr
* **Local Machine:** macOS + Cursor via SSH

# To-do
- 32d model
- tokenizer playground? https://gpt-tokenizer.dev/
- visualize embeddings/tokens?
- flashattention3
Add LoRA adapters for fine-tuning on small domains. Distill your 500M model into a 100M student to observe retention and degradation curves.
Implement an auto-chat loop (model chatting with itself using alternating prompts). Compare entropy and diversity across generations.
- Remove Benchmark-Focused Training
- Decide post-training resources that I'll add to pipeline:
  - https://huggingface.co/datasets/google/Synthetic-Persona-Chat
  - https://huggingface.co/datasets/ConvLab/dailydialog
  - https://huggingface.co/datasets/lmsys/lmsys-chat-1m
  - https://huggingface.co/datasets/argilla/magpie-ultra-v1.0

- Custumize the ui: 
  - model picker (multiple models)
- stats for nerds: t/s, ttft, total tokens 
- copy message output
-


  Go-To Commands
  - python -m venv .venv && source .venv/bin/activate — create/enter your virtual environment.
  - pip install -e . — install nanochat and deps in editable mode (run inside the venv).
  - python dev/gen_synthetic_data.py num_conversations=100 model=google/gemini-2.5-flash —
  - python -m scripts.chat_sft source=mid model_tag=d32 run=my_sft — launch the SFT fine-tune
    (swap args as needed).
  - python -m scripts.chat_web --source sft --model-tag my_sft --port 8000 — serve the chat
    UI/API on a given checkpoint.


# Miscellaneous
[https://github.com/karpathy/nanochat/discussions/1](https://github.com/karpathy/nanochat/discussions/1)

[https://deepwiki.com/karpathy/nanochat](https://deepwiki.com/karpathy/nanochat)

[https://huggingface.co/nanochat-students](https://huggingface.co/nanochat-students)


## License
MIT
-
