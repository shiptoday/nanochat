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

## Importing External Models

1. **Pick the storage root.** Checkpoints and tokenizer assets live under `NANOCHAT_BASE_DIR` (defaults to `~/.cache/nanochat`). Export another path before running anything if you want a dedicated tree:
   ```bash
   export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat_d20
   ```
   Make sure the directory exists.

2. **Download tokenizer assets.** Every model expects `tokenizer/tokenizer.pkl` and `tokenizer/token_bytes.pt` inside that base dir. Grab them from the Hugging Face repo (e.g. `karpathy/nanochat-d32`) and drop them in:
   ```
   $NANOCHAT_BASE_DIR/
     tokenizer/
       tokenizer.pkl
       token_bytes.pt
   ```

3. **Place each model in its own folder.** Under the relevant checkpoint family (`base_checkpoints/`, `mid_checkpoints/`, `chatsft_checkpoints/`, or `chatrl_checkpoints/`), create a new subdirectory matching your model tag. Keep the filename pattern `model_<step>.pt`, `meta_<step>.json`, and optionally `optim_<step>.pt`.
   ```
   $NANOCHAT_BASE_DIR/base_checkpoints/diego-d20-base/
     model_021400.pt
     meta_021400.json
     optim_021400.pt   # optional
   ```

4. **Download from Hugging Face.** Use `hf download` (or `huggingface-cli download`) to pull the blobs. Example:
   ```bash
   hf download shiptoday101/nanochat-d20-base \
     --repo-type model \
     --local-dir $NANOCHAT_BASE_DIR/base_checkpoints/diego-d20-base \
     model_021400.pt meta_021400.json optim_021400.pt
   ```
   Add `--token <hf_token>` if the repo is private.

5. **Load the model.** The existing helpers discover the newest checkpoint automatically, or you can be explicit:
   ```python
   from nanochat.checkpoint_manager import load_model
   model, tokenizer, meta = load_model(
       "base", device="cuda", phase="eval",
       model_tag="diego-d20-base", step=21400,
   )
   ```
   No custom code is required—just keep the directory layout intact.

To maintain multiple experiments, repeat the structure with different `model_tag` folders or point `NANOCHAT_BASE_DIR` to a new location before launching your scripts.

### Automated download helper

On new machines (e.g. RunPod instances) you can populate the cache in one step:

```bash
export NANOCHAT_BASE_DIR=/workspace/cache/nanochat  # pick a persistent volume
uv pip install -e .
python -m scripts.download_assets
```

By default this downloads:

- Tokenizer assets (`tokenizer.pkl`, `token_bytes.pt`) from `karpathy/nanochat-d32`
- The `diego-d20-base` checkpoint (`model_021400.pt`, `meta_021400.json`, `optim_021400.pt`) from `shiptoday101/nanochat-d20-base`

Use `--skip-tokenizer` or `--skip-diego-base` to toggle defaults, and add more bundles via `--extra repo_id::subdir::pattern1,pattern2`. Pass `--token` if the Hugging Face repo is private.

# Miscellaneous
[https://github.com/karpathy/nanochat/discussions/1](https://github.com/karpathy/nanochat/discussions/1)

[https://deepwiki.com/karpathy/nanochat](https://deepwiki.com/karpathy/nanochat)

[https://huggingface.co/nanochat-students](https://huggingface.co/nanochat-students)


## License
MIT
-
