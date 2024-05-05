import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np

# Load the MiDaS model from Torch Hub
model = torch.hub.load('intel-isl/MiDaS', 'MiDaS')
model.eval()

# Define the transform for input image preparation
transform = T.Compose([
    T.Resize(384),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to prepare the image
def prepare_image(image_path):
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return input_tensor, img

# Function to add haze based on depth map
def add_haze(img, depth_map, intensity, light_intensity):
    img_np = np.array(img)

    # Resize depth map to match image dimensions
    depth_map_resized = cv2.resize(depth_map, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Normalize depth map to range [0, 1] and invert the values
    depth_map_normalized = cv2.normalize(depth_map_resized, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    depth_map_normalized = 1 - depth_map_normalized
    
    # Calculate haze transmission map
    transmission = 1 - depth_map_normalized * intensity
    A = np.max(img_np) * light_intensity  # Estimate atmospheric light

    # Apply haze effect
    hazy_img = img_np * transmission[..., np.newaxis] + A * (1 - transmission[..., np.newaxis])
    hazy_img = np.clip(hazy_img, 0, 255).astype(np.uint8)
    return hazy_img

# Main function to run the process
def process_image(image_path):
    input_tensor, original_image = prepare_image(image_path)
    
    # Predict depth map using the MiDaS model
    with torch.no_grad():
        depth_map = model(input_tensor).squeeze().cpu().numpy()

    # Normalize and convert depth map for saving
    depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_image = np.uint8(depth_map_normalized)
    
    # Add haze effect
    hazy_image = add_haze(original_image, depth_map, intensity=0.8, light_intensity=0.8)
    
    # Save the images without changing the extension
    hazy_image_path = image_path[:-4] + '_hazy' + image_path[-4:]
    depth_map_path = image_path[:-4] + '_depth_map' + image_path[-4:]

    cv2.imwrite(hazy_image_path, cv2.cvtColor(hazy_image, cv2.COLOR_RGB2BGR))  # Save hazy image
    cv2.imwrite(depth_map_path, depth_image)  # Save depth map image

# Process an image
# process_image('j.png')



