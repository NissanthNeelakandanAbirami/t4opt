import os
from typing import Optional, List, Dict, Any


def find_checkpoints(checkpoint_dir: str) -> List[str]:
    """
    Find all checkpoint directories.
    
    Args:
        checkpoint_dir: Base checkpoint directory
        
    Returns:
        List of checkpoint directory paths
    """
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoints = []
    for item in os.listdir(checkpoint_dir):
        item_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(item_path) and "checkpoint" in item.lower():
            checkpoints.append(item_path)
    
    # Sort by checkpoint number
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0)
    return checkpoints


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Get the latest checkpoint path.
    
    Args:
        checkpoint_dir: Base checkpoint directory
        
    Returns:
        Path to latest checkpoint or None
    """
    checkpoints = find_checkpoints(checkpoint_dir)
    return checkpoints[-1] if checkpoints else None


def check_checkpoint_exists(checkpoint_dir: str) -> Dict[str, Any]:
    """
    Check if checkpoints exist and return info.
    
    Args:
        checkpoint_dir: Checkpoint directory to check
        
    Returns:
        Dictionary with checkpoint information
    """
    result = {
        "exists": False,
        "path": checkpoint_dir,
        "checkpoints": [],
        "latest_checkpoint": None,
        "files": []
    }
    
    if not os.path.exists(checkpoint_dir):
        return result
    
    result["exists"] = True

    checkpoints = find_checkpoints(checkpoint_dir)
    result["checkpoints"] = checkpoints
    
    if checkpoints:
        result["latest_checkpoint"] = checkpoints[-1]
        latest_path = checkpoints[-1]
        if os.path.exists(latest_path):
            result["files"] = os.listdir(latest_path)
    
    if os.path.exists(checkpoint_dir):
        files = os.listdir(checkpoint_dir)
        checkpoint_files = [f for f in files if any(
            ext in f for ext in [".pt", ".safetensors", ".bin", "adapter_config.json", "adapter_model.bin"]
        )]
        if checkpoint_files:
            result["files"] = files
    
    return result


def print_checkpoint_info(checkpoint_dir: str):
    """Print checkpoint information."""
    info = check_checkpoint_exists(checkpoint_dir)
    
    
    if info['exists']:
        if info['checkpoints']:
            print(f"\nFound {len(info['checkpoints'])} checkpoint(s):")
            for cp in info['checkpoints']:
                print(f"  - {os.path.basename(cp)}")
            print(f"\nLatest: {os.path.basename(info['latest_checkpoint'])}")
        else:
            print("\n Directory exists but no checkpoint subdirectories found")
            if info['files']:
                print(f"Found {len(info['files'])} file(s) in directory")
                print("Files:")
                for f in info['files'][:10]: 
                    print(f"  - {f}")
                if len(info['files']) > 10:
                    print(f"  ... and {len(info['files']) - 10} more")
    else:
        print(f"   /content/drive/MyDrive/t4opt_checkpoints/...")
    


def check_drive_checkpoints(drive_base: str = "/content/drive/MyDrive/t4opt_checkpoints") -> Dict[str, Any]:
    """
    Check for checkpoints in Google Drive.
    
    Args:
        drive_base: Base directory in Google Drive
        
    Returns:
        Dictionary with checkpoint information
    """
    result = {
        "drive_mounted": os.path.exists("/content/drive"),
        "base_exists": False,
        "checkpoints": {}
    }
    
    if not result["drive_mounted"]:
        return result
    
    if os.path.exists(drive_base):
        result["base_exists"] = True
        
        for item in os.listdir(drive_base):
            item_path = os.path.join(drive_base, item)
            if os.path.isdir(item_path):
                cp_info = check_checkpoint_exists(item_path)
                if cp_info["exists"]:
                    result["checkpoints"][item] = cp_info
    
    return result

