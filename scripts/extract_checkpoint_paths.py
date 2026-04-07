#!/usr/bin/env python3
"""
Extract checkpoint paths from .out files in logs/move4d_exps/
Creates a JSON file mapping model names to checkpoint paths.
"""

import os
import re
import json
from pathlib import Path

def extract_checkpoint_paths(logs_dir="logs/move4d_exps"):
    """Extract checkpoint paths from .out files."""
    checkpoint_map = {}
    
    logs_path = Path(logs_dir)
    out_files = sorted(logs_path.glob("move4d_*.out"))
    
    for out_file in out_files:
        # Extract model name from filename (e.g., move4d_mlp.out -> mlp)
        model_name = out_file.stem.replace("move4d_", "")
        
        # Read file and search for checkpoint path
        try:
            with open(out_file, 'r') as f:
                content = f.read()
                
            # Search for "Found best checkpoint: <path>"
            match = re.search(r'Found best checkpoint:\s+(\S+\.ckpt)', content)
            if match:
                ckpt_path = match.group(1)
                # Convert relative path to absolute
                if ckpt_path.startswith('./'):
                    ckpt_path = ckpt_path[2:]
                abs_path = os.path.abspath(ckpt_path)
                
                # Verify checkpoint exists
                if os.path.exists(abs_path):
                    checkpoint_map[model_name] = abs_path
                    print(f"✓ {model_name:25s} -> {ckpt_path}")
                else:
                    print(f"✗ {model_name:25s} -> {ckpt_path} (NOT FOUND)")
            else:
                print(f"✗ {model_name:25s} -> No checkpoint path found in .out file")
        except Exception as e:
            print(f"✗ {model_name:25s} -> Error: {e}")
    
    return checkpoint_map


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract checkpoint paths from .out files")
    parser.add_argument("--logs-dir", default="logs/move4d_exps", 
                        help="Directory containing .out files")
    parser.add_argument("--output", default="move4d_checkpoint_paths.json",
                        help="Output JSON file")
    args = parser.parse_args()
    
    print(f"Scanning {args.logs_dir} for checkpoint paths...\n")
    checkpoint_map = extract_checkpoint_paths(args.logs_dir)
    
    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(checkpoint_map, f, indent=2)
    
    print(f"\n✓ Saved {len(checkpoint_map)} checkpoint paths to {args.output}")
    print(f"  Models found: {', '.join(sorted(checkpoint_map.keys()))}")
