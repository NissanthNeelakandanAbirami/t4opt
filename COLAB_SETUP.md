# Using T4-OPT in Google Colab

Quick guide to use T4-OPT from GitHub in Colab.

## Method 1: Clone from GitHub (Recommended)

### Step 1: Create GitHub Repository

1. Push T4-OPT to GitHub (see `SETUP_GITHUB.md`)
2. Get your repository URL: `https://github.com/YOUR_USERNAME/t4opt`

### Step 2: Open Colab

1. Go to https://colab.research.google.com/
2. Create a new notebook
3. Enable GPU: Runtime → Change runtime type → GPU (T4)

### Step 3: Clone and Setup

Add this cell at the beginning:

```python
# Clone T4-OPT from GitHub
!git clone https://github.com/YOUR_USERNAME/t4opt.git /content/t4opt

# Install dependencies
!pip install -q transformers accelerate bitsandbytes peft datasets torch sentencepiece protobuf psutil tqdm numpy

# Add to Python path
import sys
sys.path.append('/content/t4opt')

# Verify installation
from utils.colab_tools import ColabTools
ColabTools.print_system_info()
```

### Step 4: Use the Notebooks

You can either:
- **Option A**: Copy cells from the notebooks in `/content/t4opt/notebooks/`
- **Option B**: Upload the notebook files to Colab and run them

## Method 2: Upload Notebooks Directly

1. Download the notebooks from `t4opt/notebooks/`
2. Upload to Colab: File → Upload notebook
3. Update the first cell to clone from GitHub:

```python
!git clone https://github.com/YOUR_USERNAME/t4opt.git /content/t4opt
sys.path.append('/content/t4opt')
```

## Method 3: Install as Package

```python
# Clone repository
!git clone https://github.com/YOUR_USERNAME/t4opt.git /content/t4opt

# Install as package
!cd /content/t4opt && pip install -e .

# Now you can import directly
from training.qlora import QLoRATrainer
from agents.planner import PlannerAgent
```

## Quick Start Example

```python
# 1. Clone and setup
!git clone https://github.com/YOUR_USERNAME/t4opt.git /content/t4opt
!pip install -q transformers accelerate bitsandbytes peft datasets torch sentencepiece
import sys
sys.path.append('/content/t4opt')

# 2. Check GPU
from utils.colab_tools import ColabTools
ColabTools.check_gpu()

# 3. Train a model
from training.qlora import QLoRATrainer, QLoRAConfig
from training.dataset import DatasetManager

config = QLoRAConfig(
    model_name="microsoft/phi-2",
    output_dir="./checkpoints",
    max_seq_length=1024,
    micro_batch_size=1,
    gradient_accumulation_steps=16,
    num_epochs=1
)

dataset_manager = DatasetManager()
dataset_info = dataset_manager.load_dataset("alpaca", max_samples=500)

trainer = QLoRATrainer(config=config)
model, tokenizer = trainer.load_model()
tokenized = dataset_manager.tokenize_dataset(
    dataset_info['dataset'], tokenizer, max_length=1024
)
trainer.train(tokenized)
```

## Troubleshooting

### Import Errors
- Make sure path is correct: `/content/t4opt`
- Restart runtime after cloning
- Check that all `__init__.py` files exist

### GPU Not Available
- Runtime → Change runtime type → GPU
- Wait a few seconds for GPU to initialize
- Check with `ColabTools.check_gpu()`

### Out of Memory
- Reduce `max_samples` in dataset
- Reduce `max_seq_length` to 512
- Increase `gradient_accumulation_steps`

## Tips

1. **Save to Drive**: Mount Google Drive to save checkpoints
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   # Then use: output_dir="/content/drive/MyDrive/t4opt_checkpoints"
   ```

2. **Resume Training**: Checkpoints are saved automatically, you can resume from them

3. **Monitor Progress**: Use `MemoryManager.print_memory_summary()` regularly

4. **Share Notebooks**: You can share your Colab notebooks with others

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [QUICKSTART.md](QUICKSTART.md) for more examples
- Explore the agent system in the notebooks

Happy training! 🚀

