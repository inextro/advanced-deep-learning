import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils import load_model, register_hook


def main(image_paths, target_layer_num, save_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    v1_encoder = load_model('convnext_v1_atto').to(device).eval()
    v2_encoder = load_model('convnext_v2_atto').to(device).eval()

    transform = _get_transform()
    for image_path in tqdm(image_paths):
        img_name = os.path.splitext(os.path.basename(image_path))[0]

        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device) # (1, 3, H, W)

        v1_activations = {}
        v2_activations = {}

        v1_hooks = register_hook(
            model=v1_encoder, model_name='convnext_v1_atto', activation_dict=v1_activations
        )
        v2_hooks = register_hook(
            model=v2_encoder, model_name='convnext_v2_atto', activation_dict=v2_activations
        )

        with torch.no_grad():
            v1_encoder(img_tensor)
            v2_encoder(img_tensor)

        v1_act = v1_activations[target_layer_num].squeeze(0) # (H, W, C)
        v1_act = v1_act.permute(2, 0, 1).cpu() # (H, W, C) -> (C, H, W)

        v2_act = v2_activations[target_layer_num].squeeze(0) # (H, W, C)
        v2_act = v2_act.permute(2, 0, 1).cpu() # (H, W, C) -> (C, H, W)

        np.random.seed(42)
        num_channels = v1_act.shape[0]
        channel_indices = np.random.choice(num_channels, 64, replace=False)

        v1_grid = _create_feature_grid(
            activation=v1_act, channel_indices=channel_indices
        )
        v2_grid = _create_feature_grid(
            activation=v2_act, channel_indices=channel_indices
        )

        _, axes = plt.subplots(1, 3, figsize=(18, 6), gridspec_kw={'wspace': 0.05})

        original_img = _unnormalize_image(img_tensor.squeeze(0))
        axes[0].imshow(original_img)
        axes[0].axis('off')

        axes[1].imshow(v1_grid, cmap='viridis')
        axes[1].axis('off')

        axes[2].imshow(v2_grid, cmap='viridis')
        axes[2].axis('off')

        save_path = os.path.join(save_dir, f'feature_activation_{img_name}.png')

        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        for h in v1_hooks + v2_hooks: h.remove()
        
def _get_transform():
    return transforms.Compose([
        transforms.Resize(256*4), 
        transforms.CenterCrop(256*4), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def _unnormalize_image(image_tensor):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = img * std + mean

    return np.clip(img, 0, 1)
    
def _create_feature_grid(activation, channel_indices, grid_size=8, upsample_size=56, padding=5):
    act_upsampled = torch.nn.functional.interpolate(
        activation.unsqueeze(0), size=(upsample_size, upsample_size), mode='bilinear', align_corners=False
    )
    activation = act_upsampled.squeeze(0).numpy()

    h, w = activation.shape[1], activation.shape[2]
    grid_h = grid_size * h + (grid_size - 1) * padding
    grid_w = grid_size * w + (grid_size - 1) * padding

    grid_img = np.zeros((grid_h, grid_w))

    for idx, ch_idx in enumerate(channel_indices):
        row = idx // grid_size
        col = idx % grid_size

        ch_img = activation[ch_idx]
        if ch_img.max() - ch_img.min() > 1e-6:
            ch_img = (ch_img - ch_img.min()) / (ch_img.max() - ch_img.min())
        
        else:
            ch_img = np.zeros_like(ch_img)

        start_h = row * (h + padding)
        start_w = col * (w + padding)

        grid_img[start_h:start_h + h, start_w:start_w + w] = ch_img

    return grid_img


if __name__ == '__main__':
    IMAGE_PATHS = [
        './assets/examples/fountain.jpg', 
        './assets/examples/goldfish.jpg', 
        './assets/examples/fox.jpg', 
        './assets/examples/dog.jpg'
    ]
    SAVE_DIR = './assets/results/'

    main(
        image_paths=IMAGE_PATHS, target_layer_num=10, save_dir=SAVE_DIR
    )