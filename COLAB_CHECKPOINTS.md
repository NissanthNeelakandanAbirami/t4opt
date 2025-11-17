# Saving and Loading Checkpoints in Colab

## ⚠️ Important: Colab Sessions are Temporary

**Files in `/content/` are LOST when your Colab session ends!**

To keep your checkpoints:
1. **Save to Google Drive** (recommended)
2. **Download checkpoints** before session ends
3. **Re-upload** to new session

## Method 1: Save to Google Drive (Best)

### During Training

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Update output directory to save in Drive
config = QLoRAConfig(
    model_name="microsoft/phi-2",
    output_dir="/content/drive/MyDrive/t4opt_checkpoints/phi-2-qlora",  # Save to Drive!
    max_seq_length=1024,
    micro_batch_size=1,
    gradient_accumulation_steps=16,
    num_epochs=3
)

# Train - checkpoints will be saved to Drive
trainer = QLoRATrainer(config=config)
# ... rest of training code
```

### After Session Crashes

```python
# Mount Drive again
from google.colab import drive
drive.mount('/content/drive')

# Check if checkpoints exist
import os
checkpoint_dir = "/content/drive/MyDrive/t4opt_checkpoints/phi-2-qlora"
if os.path.exists(checkpoint_dir):
    print("✅ Checkpoints found!")
    # List checkpoints
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint")]
    print(f"Found {len(checkpoints)} checkpoints: {checkpoints}")
else:
    print("❌ No checkpoints found")
```

### Resume Training from Checkpoint

```python
from training.qlora import QLoRATrainer, QLoRAConfig

# Use the same config with Drive path
config = QLoRAConfig(
    model_name="microsoft/phi-2",
    output_dir="/content/drive/MyDrive/t4opt_checkpoints/phi-2-qlora",
    # ... other config
)

trainer = QLoRATrainer(config=config)
model, tokenizer = trainer.load_model()

# Load dataset
from training.dataset import DatasetManager
dataset_manager = DatasetManager()
dataset_info = dataset_manager.load_dataset("alpaca", max_samples=1000)
tokenized = dataset_manager.tokenize_dataset(
    dataset_info['dataset'], tokenizer, max_length=1024
)

# Resume training - Trainer will automatically resume from latest checkpoint
trainer.train(tokenized)
```

## Method 2: Download Checkpoints Before Session Ends

```python
# After training, download checkpoints
from google.colab import files
import shutil

# Create zip of checkpoints
shutil.make_archive('checkpoints', 'zip', './checkpoints/phi-2-qlora')

# Download
files.download('checkpoints.zip')
```

### Upload in New Session

```python
# Upload the zip file in new Colab session
from google.colab import files
uploaded = files.upload()

# Extract
import zipfile
with zipfile.ZipFile('checkpoints.zip', 'r') as zip_ref:
    zip_ref.extractall('./checkpoints/')
```

## Method 3: Check if Checkpoints Exist

```python
import os

# Check local checkpoints (will be lost on session end)
local_checkpoint = "./checkpoints/phi-2-qlora"
if os.path.exists(local_checkpoint):
    print("✅ Local checkpoints found (temporary)")
    checkpoints = [d for d in os.listdir(local_checkpoint) if "checkpoint" in d]
    print(f"Checkpoints: {checkpoints}")
else:
    print("❌ No local checkpoints")

# Check Drive checkpoints (persistent)
drive_checkpoint = "/content/drive/MyDrive/t4opt_checkpoints/phi-2-qlora"
if os.path.exists(drive_checkpoint):
    print("✅ Drive checkpoints found (persistent)")
    checkpoints = [d for d in os.listdir(drive_checkpoint) if "checkpoint" in d]
    print(f"Checkpoints: {checkpoints}")
else:
    print("❌ No Drive checkpoints")
```

## Using Checkpoints for Quantization

### If Checkpoints are in Drive

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Use Drive path for quantization
from quant.merge_lora import merge_lora_weights

base_model_path = "microsoft/phi-2"
lora_path = "/content/drive/MyDrive/t4opt_checkpoints/phi-2-qlora"  # From Drive!
merged_output = "/content/drive/MyDrive/t4opt_checkpoints/phi-2-merged"

merge_result = merge_lora_weights(
    base_model_path=base_model_path,
    lora_path=lora_path,
    output_path=merged_output
)
```

### If Checkpoints are Local (and session didn't crash)

```python
# Use local path
lora_path = "./checkpoints/phi-2-qlora"
```

## Quick Recovery Script

```python
def check_and_recover_checkpoints():
    """Check for checkpoints and provide recovery options."""
    import os
    
    # Check Drive first (persistent)
    drive_path = "/content/drive/MyDrive/t4opt_checkpoints"
    if os.path.exists(drive_path):
        print("✅ Found checkpoints in Google Drive!")
        return drive_path
    
    # Check local (temporary)
    local_path = "./checkpoints"
    if os.path.exists(local_path):
        print("⚠️ Found local checkpoints (will be lost when session ends)")
        print("💡 Tip: Copy to Drive to save them!")
        return local_path
    
    print("❌ No checkpoints found")
    return None

# Use it
checkpoint_path = check_and_recover_checkpoints()
if checkpoint_path:
    print(f"Checkpoint path: {checkpoint_path}")
```

## Best Practices

1. **Always save to Drive** for important training runs
2. **Check checkpoint frequency**: `save_steps=500` means checkpoints every 500 steps
3. **Monitor training**: Checkpoints are saved automatically during training
4. **Verify before closing**: List checkpoints before ending session
5. **Use descriptive paths**: Include model name and date in path

## Example: Complete Workflow with Drive

```python
# 1. Setup
from google.colab import drive
drive.mount('/content/drive')

# 2. Train with Drive path
from training.qlora import QLoRATrainer, QLoRAConfig
from training.dataset import DatasetManager

config = QLoRAConfig(
    model_name="microsoft/phi-2",
    output_dir="/content/drive/MyDrive/t4opt_checkpoints/phi-2-qlora",
    save_steps=500,  # Save every 500 steps
    num_epochs=3
)

# ... training code ...

# 3. After session crash, resume:
# Just re-run the training code - it will resume from latest checkpoint automatically!

# 4. Quantize using Drive checkpoints
from quant.merge_lora import merge_lora_weights
merge_lora_weights(
    base_model_path="microsoft/phi-2",
    lora_path="/content/drive/MyDrive/t4opt_checkpoints/phi-2-qlora",
    output_path="/content/drive/MyDrive/t4opt_checkpoints/phi-2-merged"
)
```

## Troubleshooting

### "No checkpoints found"
- Check if training actually saved (look for `save_steps` in config)
- Verify the output directory path
- Check if session crashed before first checkpoint save

### "Cannot resume training"
- Make sure you're using the same `output_dir` path
- Check that checkpoint files exist (not just directory)
- Verify model config matches

### "Drive not mounted"
- Run `drive.mount('/content/drive')` first
- Authorize access when prompted

