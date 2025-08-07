import numpy as np
import click
import base64
from io import BytesIO
from PIL import Image
import os
from typing import List, Optional, Tuple
import math
import boto3
import pandas as pd
from data.grasp_dataset import load_np_s3
from viz.grasp_viz import draw_top_grasp_point

def create_interactive_image_grid_html(images: List[np.ndarray],
                                      names: List[str],
                                      save_dir: str,
                                      images_per_row: int = 6,
                                      max_images_per_picture: int = 24,
                                      filename_prefix: str = "image_grid",
                                      target_image_size: Tuple[int, int] = (200, 200),
                                      spacing: int = 20,
                                      background_color: str = "#ffffff",
                                      text_color: str = "#000000",
                                      font_size: int = 14,
                                      show_borders: bool = True,
                                      border_color: str = "#cccccc",
                                      include_copy_buttons: bool = True) -> List[str]:
    """
    Create interactive HTML files with copy-able image names.
    
    Args:
        images: List of numpy arrays representing images
        names: List of image names/labels
        images_per_row: Number of images per row
        max_images_per_picture: Maximum images per HTML file
        save_dir: Directory to save HTML files
        filename_prefix: Prefix for HTML files
        target_image_size: Target size for each image
        spacing: Spacing between images in pixels
        background_color: Background color
        text_color: Text color for names
        font_size: Font size for names
        show_borders: Whether to show borders around images
        border_color: Color of borders
        include_copy_buttons: Whether to include copy buttons
    
    Returns:
        List of saved HTML file paths
    """
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Validate inputs
    if len(images) != len(names):
        raise ValueError("Number of images must match number of names")
    
    saved_files = []
    total_images = len(images)
    
    # Calculate number of HTML files needed
    num_files = math.ceil(total_images / max_images_per_picture)
    
    for file_idx in range(num_files):
        # Calculate start and end indices for this file
        start_idx = file_idx * max_images_per_picture
        end_idx = min(start_idx + max_images_per_picture, total_images)
        
        # Get images and names for this file
        file_images = images[start_idx:end_idx]
        file_names = names[start_idx:end_idx]
        num_images_in_file = len(file_images)
        
        # Calculate grid dimensions
        num_rows = math.ceil(num_images_in_file / images_per_row)
        
        # Create HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Grid {file_idx + 1}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: {background_color};
        }}
        .grid-container {{
            display: grid;
            grid-template-columns: repeat({images_per_row}, 1fr);
            gap: {spacing}px;
            max-width: {images_per_row * (target_image_size[0] + spacing)}px;
            margin: 0 auto;
        }}
        .image-item {{
            text-align: center;
            padding: 10px;
            border: {2 if show_borders else 0}px solid {border_color};
            border-radius: 5px;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .image-item img {{
            max-width: {target_image_size[0]}px;
            max-height: {target_image_size[1]}px;
            width: auto;
            height: auto;
            display: block;
            margin: 0 auto;
        }}
        .image-name {{
            margin-top: 8px;
            font-size: {font_size}px;
            color: {text_color};
            word-break: break-all;
            cursor: text;
            user-select: text;
            -webkit-user-select: text;
            -moz-user-select: text;
            -ms-user-select: text;
            padding: 5px;
            border-radius: 3px;
            transition: background-color 0.2s;
        }}
        .image-name:hover {{
            background-color: #f0f0f0;
        }}
        .image-name:focus {{
            background-color: #e8f4fd;
            outline: 2px solid #0078d4;
        }}
        .copy-button {{
            margin-top: 5px;
            padding: 3px 8px;
            background-color: #0078d4;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
        }}
        .copy-button:hover {{
            background-color: #106ebe;
        }}
        .copy-button:active {{
            background-color: #005a9e;
        }}
        .header {{
            text-align: center;
            margin-bottom: 20px;
            color: #333;
        }}
        .info {{
            text-align: center;
            margin-bottom: 20px;
            color: #666;
            font-size: 14px;
        }}
        .controls {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .select-all-btn {{
            padding: 8px 16px;
            background-color: #28a745;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin: 0 5px;
        }}
        .select-all-btn:hover {{
            background-color: #218838;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Image Grid {file_idx + 1}</h1>
        <div class="info">
            Images {start_idx + 1} to {end_idx} of {total_images} total
        </div>
        <div class="controls">
            <button class="select-all-btn" onclick="selectAllNames()">Select All Names</button>
            <button class="select-all-btn" onclick="copyAllNames()">Copy All Names</button>
        </div>
    </div>
    
    <div class="grid-container">
"""
        
        # Add images and names to HTML
        for i, (img_array, name) in enumerate(zip(file_images, file_names)):
            # Convert numpy array to PIL Image
            if img_array.dtype == np.uint8:
                if len(img_array.shape) == 3:
                    pil_img = Image.fromarray(img_array)
                else:
                    pil_img = Image.fromarray(img_array, mode='L').convert('RGB')
            else:
                # Normalize to 0-255 range
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)
                
                if len(img_array.shape) == 3:
                    pil_img = Image.fromarray(img_array)
                else:
                    pil_img = Image.fromarray(img_array, mode='L').convert('RGB')
            
            # Resize image to target size (maintaining aspect ratio)
            pil_img.thumbnail(target_image_size, Image.Resampling.LANCZOS)
            
            # Convert image to base64 for embedding in HTML
            buffer = BytesIO()
            pil_img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Create HTML for this image item
            html_content += f"""
        <div class="image-item">
            <img src="data:image/png;base64,{img_base64}" alt="{name}">
            <div class="image-name" contenteditable="true" spellcheck="false">{name}</div>
"""
            
            if include_copy_buttons:
                html_content += f"""
            <button class="copy-button" onclick="copyText(this.previousElementSibling)">Copy</button>
"""
            
            html_content += """
        </div>
"""
        
        # Close grid container and add JavaScript
        html_content += f"""
    </div>
    
    <script>
        function copyText(element) {{
            const text = element.textContent;
            navigator.clipboard.writeText(text).then(function() {{
                // Visual feedback
                const button = element.nextElementSibling;
                const originalText = button.textContent;
                button.textContent = 'Copied!';
                button.style.backgroundColor = '#28a745';
                setTimeout(() => {{
                    button.textContent = originalText;
                    button.style.backgroundColor = '#0078d4';
                }}, 1000);
            }}).catch(function(err) {{
                console.error('Could not copy text: ', err);
                // Fallback: select text
                const range = document.createRange();
                range.selectNodeContents(element);
                const selection = window.getSelection();
                selection.removeAllRanges();
                selection.addRange(range);
            }});
        }}
        
        function selectAllNames() {{
            const names = document.querySelectorAll('.image-name');
            const selection = window.getSelection();
            selection.removeAllRanges();
            
            names.forEach(name => {{
                const range = document.createRange();
                range.selectNodeContents(name);
                selection.addRange(range);
            }});
        }}
        
        function copyAllNames() {{
            const names = document.querySelectorAll('.image-name');
            const allText = Array.from(names).map(name => name.textContent).join('\\n');
            
            navigator.clipboard.writeText(allText).then(function() {{
                alert('All names copied to clipboard!');
            }}).catch(function(err) {{
                console.error('Could not copy text: ', err);
                // Fallback: select all and let user copy manually
                selectAllNames();
            }});
        }}
        
        // Add keyboard shortcuts
        document.addEventListener('keydown', function(e) {{
            if (e.ctrlKey && e.key === 'a') {{
                e.preventDefault();
                selectAllNames();
            }}
            if (e.ctrlKey && e.key === 'c') {{
                // Let the default copy behavior work
            }}
        }});
        
        // Make names clickable to select
        document.addEventListener('click', function(e) {{
            if (e.target.classList.contains('image-name')) {{
                e.target.focus();
                const range = document.createRange();
                range.selectNodeContents(e.target);
                const selection = window.getSelection();
                selection.removeAllRanges();
                selection.addRange(range);
            }}
        }});
    </script>
</body>
</html>
"""
        
        # Save HTML file
        filename = f"{filename_prefix}_{file_idx:03d}.html"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        saved_files.append(filepath)
        print(f"Created HTML file {file_idx + 1}/{num_files}: {num_images_in_file} images")
    
    print(f"\nCreated {len(saved_files)} HTML file(s) with {total_images} total images")
    print(f"Files saved to: {save_dir}")
    print("\nFeatures:")
    print("- Click on any name to select it")
    print("- Use Ctrl+C to copy selected text")
    print("- Use 'Copy' buttons for individual names")
    print("- Use 'Copy All Names' to copy all names at once")
    print("- Use 'Select All Names' to select all names")
    print("- Use Ctrl+A to select all names")
    
    return saved_files

@click.command()
@click.option('--data_dir', type=str, required=True)
@click.option('--save_dir', type=str, required=True)
@click.option('--filename_prefix', type=str, required=True)
def main(data_dir, save_dir, filename_prefix):

    df = pd.read_parquet(data_dir)
    s3_client = boto3.client('s3')
    img_list = []
    name_list = []
    for i in range(len(df)):
        datum = df.iloc[i]
        image = load_np_s3(datum['obj_rgb'], s3_client)
        session_id = datum['session_id']
        img_list.append(image)
        name_list.append(session_id)

    saved_files = create_interactive_image_grid_html(img_list, name_list, save_dir, filename_prefix=filename_prefix)


if __name__ == '__main__':
    main()
    




