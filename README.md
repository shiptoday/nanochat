# primechat
![nanochat logo](dev/nanochat.png)
fork of karpathy's nanochat. i'm using it to learn more about llm's arquitecture/training. This is my playground to experiment and learn.  
561M parameters, trained on 11b tokens, developed by andrej karpathy, pre-trained and finetuned by me (Diego Prime)

# Models
- d32 trained by karpathy: 
  - weights: https://huggingface.co/karpathy/nanochat-d32
  - details: https://github.com/karpathy/nanochat/discussions/8
- d20-base trained by me: 
  - weights: https://huggingface.co/shiptoday101/nanochat-d20-base/tree/main
  - details: 
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

# GPU setup
* **Cloud:** RunPod (GPU training)
* **Pod Specs:**
  * 8× H200 SXM (1128 GB VRAM total)
  * 192 vCPU / 2008 GB RAM
  * 80 GB disk
  * Image: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
  * Cost: ≈ $28.72 / hr
* **Local Machine:** macOS + Cursor via SSH

# To-do (ideas/experiments)
    - [ ]  tokenizer playground
    - [ ]  Nanochat visualizations
    - [ ]  Nanochat vibe testing
    - [ ]  Nanochat finetuning
        - [ ]  auto-chat loop: model chatting with itself using alternating prompts). Compare entropy and diversity across generations.
    - [ ]  UI modifications:
        - [ ]  copy messags
        - [ ]  model picker
        - [ ]  stats for nerds: t/s, ttft, total tokens
        - [ ]  run inference multiple times at the same time to compare outputs
- [ ]  synthetic data to finetune 32d model

# Miscellaneous
[https://github.com/karpathy/nanochat/discussions/1](https://github.com/karpathy/nanochat/discussions/1)

[https://deepwiki.com/karpathy/nanochat](https://deepwiki.com/karpathy/nanochat)

[https://huggingface.co/nanochat-students](https://huggingface.co/nanochat-students)
