# T4-OPT: Lightweight Agentic LLM Training & Optimization System

**Fully T4-Compatible | Pure Python | Works 100% in Free Colab**

T4-OPT is an advanced, production-ready system for training, quantizing, and evaluating large language models on T4 GPUs (16GB VRAM). It features a multi-agent orchestration system and is designed to work entirely within Google Colab's free tier.

## 🌟 Features

- ✅ **QLoRA 4-bit Training** - Efficient fine-tuning with minimal VRAM usage
- ✅ **Multi-Agent System** - Planner, Trainer, Optimizer, Evaluator, and Recovery agents
- ✅ **Model Quantization** - INT8, AWQ, and NF4 quantization support
- ✅ **Comprehensive Evaluation** - Perplexity, benchmarks, and speed tests
- ✅ **T4-Optimized** - Memory-efficient configurations for 16GB VRAM
- ✅ **Pure Python** - No backend or frontend required
- ✅ **Colab-Ready** - Works out of the box in Google Colab

## 📂 Project Structure

```
t4opt/
├── agents/              # Multi-agent system
│   ├── base.py         # Base agent class
│   ├── planner.py      # Task planning agent
│   ├── trainer.py      # Training agent
│   ├── optimizer.py    # Quantization agent
│   ├── evaluator.py    # Evaluation agent
│   └── recovery.py     # Recovery agent
│
├── training/           # Training modules
│   ├── qlora.py       # QLoRA trainer
│   ├── dataset.py     # Dataset management
│   └── trainer_utils.py  # Training utilities
│
├── quant/              # Quantization modules
│   ├── merge_lora.py  # LoRA merging
│   ├── quant_int8.py  # INT8 quantization
│   └── quant_awq.py   # AWQ quantization
│
├── eval/               # Evaluation modules
│   ├── perplexity.py  # Perplexity evaluation
│   ├── benchmarks.py  # Benchmark suite
│   └── speed_test.py  # Speed/latency tests
│
├── utils/              # Utilities
│   ├── memory.py      # Memory management
│   ├── logger.py      # Logging
│   ├── config.py      # Configuration
│   └── colab_tools.py # Colab helpers
│
└── notebooks/          # Jupyter notebooks
    ├── 1_setup_environment.ipynb
    ├── 2_train_llm_t4.ipynb
    ├── 3_quantize_model.ipynb
    ├── 4_eval_model.ipynb
    └── 5_agents_inference.ipynb
```

## 🚀 Quick Start

### 1. Setup Environment (Colab)

```python
# In Colab notebook
!git clone <your-repo-url> /content/t4opt
%cd /content/t4opt

from utils.colab_tools import ColabTools
ColabTools.setup_environment()
```

### 2. Train a Model

```python
from training.qlora import QLoRATrainer, QLoRAConfig
from training.dataset import DatasetManager

# Configure training
config = QLoRAConfig(
    model_name="microsoft/phi-2",
    output_dir="./checkpoints",
    max_seq_length=1024,
    micro_batch_size=1,
    gradient_accumulation_steps=16,
    num_epochs=3
)

# Load dataset
dataset_manager = DatasetManager()
dataset_info = dataset_manager.load_dataset("alpaca", max_samples=1000)

# Train
trainer = QLoRATrainer(config=config)
model, tokenizer = trainer.load_model()
tokenized_dataset = dataset_manager.tokenize_dataset(
    dataset_info['dataset'], tokenizer, max_length=1024
)
trainer.train(tokenized_dataset)
```

### 3. Quantize Model

```python
from quant.merge_lora import merge_lora_weights
from quant.quant_int8 import quantize_to_int8

# Merge LoRA
merge_lora_weights(
    base_model_path="microsoft/phi-2",
    lora_path="./checkpoints",
    output_path="./merged_model"
)

# Quantize
quantize_to_int8("./merged_model", {"output_path": "./quantized_model"})
```

### 4. Evaluate Model

```python
from eval.perplexity import PerplexityEvaluator
from eval.benchmarks import BenchmarkRunner

# Perplexity
evaluator = PerplexityEvaluator(model_path="./checkpoints")
result = evaluator.evaluate()
print(f"Perplexity: {result['perplexity']:.4f}")

# Benchmarks
runner = BenchmarkRunner(model_path="./checkpoints")
results = runner.run(benchmarks=["mmlu", "generation"])
```

### 5. Use Multi-Agent System

```python
from agents.planner import PlannerAgent
from agents.trainer import TrainingAgent

# Plan task
planner = PlannerAgent()
plan = planner.execute("Train a model on alpaca dataset")

# Execute with training agent
trainer = TrainingAgent()
result = trainer.execute("run_training", context={
    "model_name": "microsoft/phi-2",
    "dataset_name": "alpaca"
})
```

## 🎯 Recommended Models for T4

| Model | Size | Trainable on T4? | Notes |
|-------|------|------------------|-------|
| TinyLlama-1.1B | 1.1B | ✅✅✅ | Easiest, perfect for training |
| Phi-2 | 2.7B | ✅✅ | Good fit with QLoRA |
| Gemma-2B | 2B | ✅✅ | Fast and high quality |
| Llama-3-3B | 3B | ⚠️ | Works but slow |
| Mistral 7B | 7B | ❌ | Too large for T4 training |

**Recommended**: Phi-2 or Gemma-2B for best results.

## ⚙️ T4 Training Configuration

Key settings for T4 (16GB VRAM):

- **QLoRA 4-bit**: `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`
- **Gradient Checkpointing**: `use_gradient_checkpointing=True`
- **Sequence Length**: `max_seq_length=1024`
- **Micro Batch Size**: `micro_batch_size=1-2`
- **Gradient Accumulation**: `gradient_accumulation_steps=16-32`
- **Mixed Precision**: `fp16=True`
- **Optimizer**: `optim="paged_adamw_8bit"`

Expected performance:
- Training speed: ~2 it/s
- Finetuning time: ~45-90 mins per epoch
- Model fits well in VRAM

## 📊 Evaluation Metrics

T4-OPT includes comprehensive evaluation:

- **Perplexity**: Language modeling quality
- **MMLU**: Multi-task language understanding (subset)
- **Generation Quality**: Text generation benchmarks
- **Speed Tests**: Latency and throughput measurements
- **Toxicity Tests**: Basic safety evaluation

## 🔧 Installation

### Local Installation

```bash
pip install -r requirements.txt
```

### Colab Installation

The notebooks handle installation automatically, or run:

```python
from utils.colab_tools import ColabTools
ColabTools.install_dependencies()
```

## 📝 Usage Examples

See the Jupyter notebooks for detailed examples:

1. **1_setup_environment.ipynb** - Environment setup
2. **2_train_llm_t4.ipynb** - Training workflow
3. **3_quantize_model.ipynb** - Quantization workflow
4. **4_eval_model.ipynb** - Evaluation workflow
5. **5_agents_inference.ipynb** - Multi-agent system

## 🧩 Agent System

T4-OPT features a multi-agent orchestration system:

- **PlannerAgent**: Breaks down high-level tasks into actionable steps
- **TrainingAgent**: Handles QLoRA fine-tuning
- **OptimizeAgent**: Manages quantization and optimization
- **EvalAgent**: Runs evaluation benchmarks
- **RecoveryAgent**: Handles training failures and checkpoint recovery

## 💡 Tips for T4 Training

1. **Start Small**: Use 500-1000 samples for initial testing
2. **Monitor Memory**: Use `MemoryManager.print_memory_summary()` regularly
3. **Save Frequently**: Set `save_steps=500` to avoid losing progress
4. **Use Gradient Accumulation**: Compensate for small batch sizes
5. **Clear Cache**: Call `MemoryManager.clear_cache()` between operations

## 🐛 Troubleshooting

### Out of Memory (OOM)

- Reduce `max_seq_length` to 512
- Increase `gradient_accumulation_steps`
- Reduce `max_samples` in dataset
- Use `MemoryManager.clear_cache()`

### Slow Training

- Normal for T4: expect ~2 it/s
- Reduce `num_epochs` for testing
- Use smaller models (Phi-2, Gemma-2B)

### Import Errors

- Ensure all dependencies are installed
- Check Python path includes t4opt directory
- Restart Colab runtime if needed

## 📄 License

This project is provided as-is for educational and research purposes.

## 🙏 Acknowledgments

Built with:
- [Transformers](https://github.com/huggingface/transformers)
- [PEFT](https://github.com/huggingface/peft)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- [Accelerate](https://github.com/huggingface/accelerate)

## 📧 Contact

For questions or issues, please open an issue on the repository.

---

**Built for T4 GPUs | Optimized for Research | Production-Ready Architecture**

