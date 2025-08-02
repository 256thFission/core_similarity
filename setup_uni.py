#!/usr/bin/env python3
"""
Setup script for UNI2-h model authentication and download
"""

import os
import sys
from pathlib import Path
from rich.console import Console
from rich.prompt import Prompt

console = Console()

def setup_uni_model():
    """Setup UNI2-h model with authentication"""
    
    console.print("[bold blue]🔧 UNI2-h Model Setup[/bold blue]")
    console.print("=" * 50)
    
    # Check if huggingface_hub is available
    try:
        from huggingface_hub import login, whoami
    except ImportError:
        console.print("[red]❌ huggingface_hub not installed[/red]")
        console.print("Install with: pip install -r requirements.txt")
        sys.exit(1)
    
    # Check if already authenticated
    try:
        user_info = whoami()
        console.print(f"[green]✅ Already authenticated as: {user_info['name']}[/green]")
    except:
        console.print("[yellow]⚠️  Not authenticated with Hugging Face[/yellow]")
        
        console.print("\n[bold]Setup Instructions:[/bold]")
        console.print("1. Go to https://huggingface.co/settings/tokens")
        console.print("2. Create a new token with 'Read' permissions")
        console.print("3. Request access to UNI2-h model at: https://huggingface.co/MahmoodLab/UNI2-h")
        console.print("4. Enter your token below")
        
        token = Prompt.ask("\n[bold]Enter your Hugging Face token", password=True)
        
        try:
            login(token=token)
            console.print("[green]✅ Authentication successful![/green]")
        except Exception as e:
            console.print(f"[red]❌ Authentication failed: {e}[/red]")
            sys.exit(1)
    
    # Test model access
    console.print("\n[blue]📦 Testing model access...[/blue]")
    
    try:
        import timm
        import torch
        
        # UNI2-h configuration from official docs
        timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        
        console.print("[blue]⬇️  Downloading UNI2-h model (this may take a while)...[/blue]")
        
        # This will download the model to huggingface cache
        model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        
        console.print("[green]✅ UNI2-h model downloaded successfully![/green]")
        console.print(f"[blue]📊 Model embed dimension: {model.embed_dim}[/blue]")
        
        # Test inference
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
        console.print("[green]✅ Image transforms configured[/green]")
        
    except Exception as e:
        console.print(f"[red]❌ Model setup failed: {e}[/red]")
        console.print("\nPossible issues:")
        console.print("- You may not have access to the UNI2-h model")
        console.print("- Check your internet connection")
        console.print("- Ensure you have sufficient disk space")
        sys.exit(1)
    
    console.print("\n[bold green]🎉 Setup Complete![/bold green]")
    console.print("UNI2-h model is ready for use.")
    console.print("\nYou can now run:")
    console.print("  python main.py match-tissues <set_a_dir> <set_b_dir>")

if __name__ == "__main__":
    setup_uni_model()