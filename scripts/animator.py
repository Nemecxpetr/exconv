"""Creates a gif animation from a sequence of images loaded from a directory.
"""
import os
from pathlib import Path
import imageio.v3 as iio
import click    

@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_path", type=click.Path(dir_okay=False))
def create_gif(input_dir, output_path):
    """Creates a gif animation from a sequence of images in INPUT_DIR and saves it to OUTPUT_PATH.

    INPUT_DIR : directory containing input images
    OUTPUT_PATH: output gif path
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    # Collect image file paths and sort them
    image_files = sorted(input_dir.glob("*"))
    
    # Read images
    images = [iio.imread(str(img_file)) for img_file in image_files]

    # Save as gif
    iio.imwrite(str(output_path), images, format="GIF", loop=0, duration=0.1)

    print(f"GIF animation saved to {output_path}")

if __name__ == "__main__":
    create_gif()