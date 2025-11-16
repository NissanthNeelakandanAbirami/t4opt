# 🚀 Quick Guide: Push to GitHub & Use in Colab

## Step 1: Push to GitHub

### Option A: Use the Script (Easiest)

```bash
cd /Users/nissanthneelakandanabirami/Downloads/t4opt
./push_to_github.sh
```

Follow the prompts to enter your GitHub repository URL.

### Option B: Manual Steps

```bash
cd /Users/nissanthneelakandanabirami/Downloads/t4opt

# Add all files
git add .

# Commit
git commit -m "Initial commit: T4-OPT - Lightweight Agentic LLM Training & Optimization System"

# Add your GitHub repository (replace with your URL)
git remote add origin https://github.com/YOUR_USERNAME/t4opt.git

# Rename branch to main
git branch -M main

# Push
git push -u origin main
```

**Note**: If you haven't created the GitHub repository yet:
1. Go to https://github.com/new
2. Create a new repository (name it `t4opt` or `t4-opt`)
3. **Don't** initialize with README (we already have one)
4. Copy the repository URL and use it in the commands above

## Step 2: Use in Colab

### Quick Setup Cell for Colab

Copy this into the first cell of your Colab notebook:

```python
# Clone T4-OPT from GitHub (replace YOUR_USERNAME with your GitHub username)
!git clone https://github.com/YOUR_USERNAME/t4opt.git /content/t4opt

# Install dependencies
!pip install -q transformers accelerate bitsandbytes peft datasets torch sentencepiece protobuf psutil tqdm numpy

# Add to Python path
import sys
sys.path.append('/content/t4opt')

# Verify installation
from utils.colab_tools import ColabTools
ColabTools.print_system_info()
ColabTools.check_gpu()
```

### Then Use the Notebooks

1. **Option A**: Copy cells from `/content/t4opt/notebooks/1_setup_environment.ipynb` etc.
2. **Option B**: Upload the notebook files to Colab and run them

## Step 3: Start Training!

After setup, you can start training:

```python
from training.qlora import QLoRATrainer, QLoRAConfig
from training.dataset import DatasetManager

# Configure
config = QLoRAConfig(
    model_name="microsoft/phi-2",
    output_dir="./checkpoints",
    max_seq_length=1024,
    micro_batch_size=1,
    gradient_accumulation_steps=16,
    num_epochs=1  # Start with 1 epoch for testing
)

# Load dataset
dataset_manager = DatasetManager()
dataset_info = dataset_manager.load_dataset("alpaca", max_samples=500)

# Train
trainer = QLoRATrainer(config=config)
model, tokenizer = trainer.load_model()
tokenized = dataset_manager.tokenize_dataset(
    dataset_info['dataset'], tokenizer, max_length=1024
)
trainer.train(tokenized)
```

## 📝 Important Notes

1. **Replace `YOUR_USERNAME`** with your actual GitHub username in all URLs
2. **Enable GPU** in Colab: Runtime → Change runtime type → GPU (T4)
3. **First run**: Use small dataset (500 samples) and 1 epoch to test
4. **Save checkpoints**: They're saved automatically to `./checkpoints`

## 🔗 Helpful Links

- Full documentation: [README.md](README.md)
- Detailed Colab guide: [COLAB_SETUP.md](COLAB_SETUP.md)
- GitHub setup details: [SETUP_GITHUB.md](SETUP_GITHUB.md)
- Quick start examples: [QUICKSTART.md](QUICKSTART.md)

## ✅ Checklist

- [ ] Created GitHub repository
- [ ] Pushed code to GitHub
- [ ] Updated clone URL in Colab notebooks
- [ ] Enabled GPU in Colab
- [ ] Ran setup cell successfully
- [ ] Started first training run

That's it! You're ready to train LLMs on T4 GPUs! 🎉

