import os
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
from diffusers import DPMSolverMultistepScheduler

# Create sim2real diffusion pipeline
try:
    pipe # Check if pipeline already exists
except NameError:
    pipe = AutoPipelineForImage2Image.from_pretrained(
        "../training/model_autodrive_large_offroad",
        scheduler = DPMSolverMultistepScheduler.from_pretrained("../training/model_autodrive_large_offroad", subfolder="scheduler"),
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter-plus_sd15.bin"),
    pipe.set_ip_adapter_scale(0.9)

# Get the current directory where the script is located
script_dir = os.getcwd()

# Folder name to load/store images
image_folder = os.path.join(script_dir, "input_autodrive_large_offroad") # Load input frames from here
style_folder = os.path.join(script_dir, "style_autodrive_large_offroad") # Load style frames from here
output_folder = os.path.join(script_dir, "output_autodrive_large_offroad") # Store output results here
os.makedirs(output_folder, exist_ok=True) # Create the directory if it doesn't exist

# List all image files in the input folder
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
style_files = [f for f in os.listdir(style_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")#.manual_seed(22)

# Load images, perform style transfer, and save outputs
for i in range(len(image_files)):
    # Load image
    print(f"Loading {image_files[i]}...")
    image_path = os.path.join(image_folder, image_files[i]) # Get image path
    style_path = os.path.join(style_folder, style_files[0]) # Get style path
    image = load_image(image_path) # Load image
    style = load_image(style_path) # Load style

    # Run inference
    print(f"Running inference...")
    output = pipe(
        prompt="<autodrive_large_offroad> simulation",
        negative_prompt="patches, tiling, extra objects, distorted, discontinuous, ugly, blurry, low resolution, artifacts",
        image=image, # Input image
        ip_adapter_image=style, # Style image
        generator=generator, # Pytorch generator
        strength=0.4, # Adherence to input image (higher values = more freedom to change, lower values = more faithful)
        num_inference_steps=10, # Denoising steps (e.g., 5 effectively gets converted to 3)
        guidance_scale=5.0 # Adherence to prompt (higher values = more prompt adherence, lower values = more creativity)
    ).images[0]

    # Display or save the result
    print(f"Saving result...")
    # output.show()
    output.save(os.path.join(output_folder, image_files[i]))