# Setting Up T4-OPT on GitHub

Follow these steps to push T4-OPT to GitHub and use it in Colab.

## Step 1: Initialize Git Repository

```bash
cd /Users/nissanthneelakandanabirami/Downloads/t4opt

# Initialize git
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: T4-OPT - Lightweight Agentic LLM Training & Optimization System"
```

## Step 2: Create GitHub Repository

1. Go to https://github.com/new
2. Create a new repository (e.g., `t4opt` or `t4-opt`)
3. **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. Copy the repository URL (e.g., `https://github.com/yourusername/t4opt.git`)

## Step 3: Push to GitHub

```bash
# Add remote (replace with your GitHub URL)
git remote add origin https://github.com/yourusername/t4opt.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 4: Use in Colab

### Option A: Clone in Colab Notebook

Add this cell at the beginning of your Colab notebook:

```python
# Clone T4-OPT from GitHub
!git clone https://github.com/yourusername/t4opt.git /content/t4opt

# Add to Python path
import sys
sys.path.append('/content/t4opt')
```

### Option B: Install as Package

```python
# Clone and install
!git clone https://github.com/yourusername/t4opt.git /content/t4opt
!cd /content/t4opt && pip install -e .
```

## Step 5: Update Notebooks

The notebooks already have the path setup. Just update the clone URL in the first cell:

```python
# In notebook first cell
!git clone https://github.com/yourusername/t4opt.git /content/t4opt
sys.path.append('/content/t4opt')
```

## Quick Colab Setup

Here's a complete setup cell for Colab:

```python
# Install dependencies and clone T4-OPT
!pip install -q transformers accelerate bitsandbytes peft datasets torch sentencepiece protobuf psutil tqdm numpy

# Clone your repository
!git clone https://github.com/yourusername/t4opt.git /content/t4opt

# Add to path
import sys
sys.path.append('/content/t4opt')

# Verify installation
from utils.colab_tools import ColabTools
ColabTools.print_system_info()
```

## Troubleshooting

### If you get authentication errors:
- Use HTTPS with personal access token, or
- Use SSH: `git@github.com:yourusername/t4opt.git`

### If files are too large:
- Check `.gitignore` excludes model files
- Use Git LFS for large files if needed

### If Colab can't find modules:
- Make sure path is correct: `/content/t4opt`
- Restart Colab runtime after cloning
- Check that `__init__.py` files exist in all modules

## Next Steps

1. Push your code to GitHub
2. Update the clone URL in notebooks
3. Share the repository link
4. Start training in Colab!

