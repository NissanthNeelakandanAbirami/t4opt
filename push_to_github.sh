#!/bin/bash
# Quick script to push T4-OPT to GitHub

echo "🚀 T4-OPT GitHub Push Script"
echo "=============================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
fi

# Add all files
echo "Adding files to git..."
git add .

# Check if there are changes
if git diff --staged --quiet; then
    echo "No changes to commit."
else
    # Commit
    echo "Creating commit..."
    git commit -m "Initial commit: T4-OPT - Lightweight Agentic LLM Training & Optimization System"
fi

# Check if remote exists
if git remote | grep -q "origin"; then
    echo ""
    echo "Remote 'origin' already exists."
    echo "Current remote URL:"
    git remote get-url origin
    echo ""
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter your GitHub repository URL: " repo_url
        git remote set-url origin "$repo_url"
    fi
else
    echo ""
    read -p "Enter your GitHub repository URL (e.g., https://github.com/username/t4opt.git): " repo_url
    git remote add origin "$repo_url"
fi

# Rename branch to main
git branch -M main

# Push
echo ""
echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ Done! Your code is now on GitHub."
echo ""
echo "To use in Colab, add this to your notebook:"
echo "!git clone $repo_url /content/t4opt"

