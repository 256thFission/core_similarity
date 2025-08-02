#!/usr/bin/env python3
"""
CLI interface for tissue microarray core similarity matching
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.core import TissueMatchingPipeline, BasicImageProcessor, ImagePathUtils

app = typer.Typer(help="Tissue Microarray Core Similarity Matching CLI")
console = Console()


@app.command()
def match_tissues(
    set_a_dir: Path = typer.Argument(..., help="Set A directory (grayscale images)", exists=True, file_okay=False, dir_okay=True),
    set_b_dir: Path = typer.Argument(..., help="Set B directory (H&E images with calibration dots)", exists=True, file_okay=False, dir_okay=True),
    output_file: Optional[Path] = typer.Option("matches.json", "--output", "-o", help="Output file for matched pairs"),
    model: str = typer.Option("uni2h", "--model", "-m", help="Feature extraction model (uni2h, biomedclip, conch, virchow2, kimianet)"),
    flip_set_b: bool = typer.Option(False, "--flip-set-b", help="Flip Set B images horizontally (left-right) before processing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """
    Match tissue microarray images between Set A (clean grayscale) and Set B (H&E with dots).
    """
    
    console.print("[bold blue]🔬 Starting Tissue Microarray Matching Pipeline[/bold blue]")
    console.print(f"[blue]Using model: {model}[/blue]")
    
    # Initialize pipeline
    pipeline = TissueMatchingPipeline(model=model)
    
    # Get image counts (only _image.png files)
    filename_filter = "_image.png"
    set_a_count = ImagePathUtils.count_images_in_directory(set_a_dir, filename_filter)
    set_b_count = ImagePathUtils.count_images_in_directory(set_b_dir, filename_filter)
    
    console.print(f"[green]Found {set_a_count} images in Set A (clean grayscale)[/green]")
    console.print(f"[green]Found {set_b_count} images in Set B (H&E with dots)[/green]")
    
    if set_a_count == 0 or set_b_count == 0:
        console.print("[red]No images found. They must have a ***_image.png file name. [/red]")
        raise typer.Exit(1)
    
    # Progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Processing...", total=None)
        
        def progress_callback(message: str):
            progress.update(task, description=message)
            if verbose:
                console.print(f"[cyan]{message}[/cyan]")
        
        try:
            # Run the complete pipeline
            matched_pairs, summary = pipeline.match_tissues(
                set_a_dir=set_a_dir,
                set_b_dir=set_b_dir,
                output_file=str(output_file),
                progress_callback=progress_callback,
                filename_filter=filename_filter,
                flip_set_b=flip_set_b
            )
            
            # Display results
            console.print(f"[bold green] Matching complete[/bold green]")
            console.print(f"[green]Results saved to {output_file}[/green]")
            console.print(f"[green]Tuple format saved to {output_file.with_suffix('.py')}[/green]")
            console.print(f"[blue]Successfully matched {summary['successful_matches']} pairs[/blue]")
            console.print(f"[blue]Average cosine distance: {summary['average_distance']:.4f}[/blue]")
            
            if verbose:
                console.print(f"[dim]Distance range: [{summary['min_distance']:.4f}, {summary['max_distance']:.4f}][/dim]")
                console.print(f"[dim]Distance std dev: {summary['std_distance']:.4f}[/dim]")
                
                # Show first few matches
                console.print("\n[bold]First 5 matches:[/bold]")
                for i, pair in enumerate(matched_pairs[:5]):
                    console.print(f"  {pair['set_a']} → {pair['set_b']} (distance: {pair['cosine_distance']:.4f})")
            
        except Exception as e:
            console.print(f"[red]❌ Pipeline failed: {e}[/red]")
            raise typer.Exit(1)


@app.command()
def process_directories(
    dir1: Path = typer.Argument(..., help="First directory containing images", exists=True, file_okay=False, dir_okay=True),
    dir2: Path = typer.Argument(..., help="Second directory containing images", exists=True, file_okay=False, dir_okay=True),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for processed images"),
    median_filter_size: int = typer.Option(3, "--median-filter", "-m", help="Median filter size for noise reduction"),
    downsample_factor: float = typer.Option(8.0, "--downsample", "-d", help="Downsample factor for images"),
    normalize: bool = typer.Option(True, "--normalize/--no-normalize", help="Apply intensity normalization"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
):
    """
    Process images from two directories using basic preprocessing (no UNI2-h features).
    
    This command applies basic preprocessing steps like grayscale conversion,
    downsampling, median filtering, and normalization.
    """
    
    console.print("[bold blue]🔧 Processing Images with Basic Preprocessing[/bold blue]")
    
    # Initialize basic processor
    processor = BasicImageProcessor(
        downsample_factor=downsample_factor,
        median_filter_size=median_filter_size
    )
    
    # Prepare directories and output
    directories = [dir1, dir2]
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[blue]Output directory: {output_dir}[/blue]")
    
    # Progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("Processing images...", total=None)
        
        def progress_callback(current: int, total: int, filename: str):
            progress.update(task, description=f"Processing {filename} ({current}/{total})")
            if verbose:
                console.print(f"[cyan]Processing: {filename}[/cyan]")
        
        try:
            # Process images
            stats = processor.process_images_from_directories(
                directories=directories,
                output_dir=output_dir,
                normalize=normalize,
                progress_callback=progress_callback
            )
            
            # Display results
            console.print(f"[bold green]✅ Processing complete![/bold green]")
            console.print(f"[green]Processed: {stats['processed']} images[/green]")
            console.print(f"[yellow]Failed: {stats['failed']} images[/yellow]")
            console.print(f"[blue]Total: {stats['total']} images[/blue]")
            
        except Exception as e:
            console.print(f"[red]❌ Processing failed: {e}[/red]")
            raise typer.Exit(1)


@app.command()
def info(
    directory: Path = typer.Argument(..., help="Directory to analyze", exists=True, file_okay=False, dir_okay=True)
):
    """
    Display information about images in a directory.
    """
    from src.core import ImageLoader
    
    console.print(f"[green]Directory: {directory}[/green]")
    
    # Get image files
    images = ImagePathUtils.get_images_from_directory(directory)
    console.print(f"[green]Found {len(images)} images[/green]")
    
    if images:
        # Sample first image for detailed info
        loader = ImageLoader()
        sample_image = loader.load_image(str(images[0]))
        
        if sample_image is not None:
            console.print(f"[blue]Sample image shape: {sample_image.shape}[/blue]")
            console.print(f"[blue]Sample image dtype: {sample_image.dtype}[/blue]")
        
        # List first 10 images  
        console.print("\n[bold]Image files:[/bold]")
        for image_path in images[:10]:
            console.print(f"  - {image_path.name}")
        
        if len(images) > 10:
            console.print(f"  ... and {len(images) - 10} more")
    else:
        console.print("[yellow]No images found in directory[/yellow]")


if __name__ == "__main__":
    app()