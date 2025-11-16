# T4-OPT Quick Start Guide

Get started with T4-OPT in 5 minutes!

## 🚀 Colab Quick Start

### Step 1: Open Colab and Setup

1. Open Google Colab: https://colab.research.google.com/
2. Enable GPU: Runtime → Change runtime type → GPU (T4)
3. Clone/Upload T4-OPT:

```python
# Option A: If you have the code in a repo
!git clone <your-repo-url> /content/t4opt

# Option B: Upload the t4opt folder manually
# Then unzip if needed
```

### Step 2: Run Setup Notebook

Open `notebooks/1_setup_environment.ipynb` and run all cells.

This will:
- ✅ Check GPU availability
- ✅ Install dependencies
- ✅ Setup environment
- ✅ Verify installation

### Step 3: Train Your First Model

Open `notebooks/2_train_llm_t4.ipynb` and run all cells.

**Recommended settings for first run:**
- Model: `microsoft/phi-2` (2.7B parameters)
- Dataset: `alpaca` with 500-1000 samples
- Epochs: 1-2 (for testing)
- Expected time: ~30-60 minutes

### Step 4: Evaluate

Open `notebooks/4_eval_model.ipynb` to evaluate your trained model.

### Step 5: (Optional) Quantize

Open `notebooks/3_quantize_model.ipynb` to quantize your model for deployment.

## 📝 Minimal Example

```python
# 1. Setup
import sys
sys.path.append('/content/t4opt')
from utils.colab_tools import ColabTools
ColabTools.setup_environment()

# 2. Train
from training.qlora import QLoRATrainer, QLoRAConfig
from training.dataset import DatasetManager

config = QLoRAConfig(
    model_name="microsoft/phi-2",
    output_dir="./checkpoints",
    max_seq_length=1024,
    micro_batch_size=1,
    gradient_accumulation_steps=16,
    num_epochs=1  # Start with 1 epoch
)

dataset_manager = DatasetManager()
dataset_info = dataset_manager.load_dataset("alpaca", max_samples=500)

trainer = QLoRATrainer(config=config)
model, tokenizer = trainer.load_model()
tokenized = dataset_manager.tokenize_dataset(
    dataset_info['dataset'], tokenizer, max_length=1024
)
trainer.train(tokenized)

# 3. Test
from transformers import pipeline
pipe = pipeline("text-generation", model="./checkpoints", tokenizer=tokenizer)
print(pipe("What is machine learning?")[0]['generated_text'])
```

## ⚡ Tips for Success

1. **Start Small**: Use 500 samples and 1 epoch for first run
2. **Monitor Memory**: Check memory usage regularly
3. **Save Often**: Checkpoints are saved automatically
4. **Be Patient**: T4 training is slow (~2 it/s) but works!

## 🐛 Common Issues

### Out of Memory
- Reduce `max_samples` to 300-500
- Reduce `max_seq_length` to 512
- Increase `gradient_accumulation_steps` to 32

### Slow Training
- Normal! T4 is slow but free
- Use smaller models (Phi-2, Gemma-2B)
- Reduce epochs for testing

### Import Errors
- Restart Colab runtime
- Re-run setup notebook
- Check Python path includes `/content/t4opt`

## 📚 Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the agent system in `notebooks/5_agents_inference.ipynb`
- Try different models and datasets
- Experiment with quantization

## 🎯 Recommended Workflow

1. **First Run**: Use default settings, 500 samples, 1 epoch
2. **Second Run**: Increase to 1000 samples, 2-3 epochs
3. **Production**: Full dataset, 3+ epochs, then quantize

Happy training! 🚀

