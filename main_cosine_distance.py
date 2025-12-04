import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from matplotlib import pyplot as plt
from models import fcmae_convnext_v1_atto, fcmae_convnext_v2_atto


def _load_model(model_name):
    if model_name == 'convnext_v1_atto':
        model = fcmae_convnext_v1_atto()
        ckpt = torch.load(
            os.path.join('./ckpt', f'{model_name + '_799.pth'}'),
            weights_only=False, map_location='cpu'
        )
    
    elif model_name == 'convnext_v2_atto':
        model = fcmae_convnext_v2_atto()
        ckpt = torch.load(
            os.path.join('./ckpt', f'{model_name + '_799.pth'}'), 
            weights_only=False, map_location='cpu'
        )

    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)

    return model.encoder


def _get_dataloader(data_dir, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(data_dir, transform=transform)
    targets = np.array(val_dataset.targets)
    
    selected_indices = []
    for class_idx in range(1000):
        indices = np.where(targets == class_idx)[0]
        if len(indices) > 0:
            selected_indices.append(indices[0])

    val_subset = Subset(val_dataset, selected_indices)

    return DataLoader(
        val_subset, batch_size=batch_size, shuffle=False
    )


def _register_hook(model, model_name, activation_dict):
    hooks = []

    def get_hook(name):
        def hook(model, input, output):
            activation_dict[name] = output.detach()
        return hook

    if model_name == 'convnext_v1_atto':
        idx = 0
        for stage in model.stages:
            for block in stage:
                h = block.gelu.register_forward_hook(get_hook(idx))
                hooks.append(h)
                idx += 1
    
    elif model_name == 'convnext_v2_atto':
        idx = 0
        for stage in model.stages:
            for block in stage:
                h = block.grn.register_forward_hook(get_hook(idx))
                hooks.append(h)
                idx += 1

    return hooks

def _compute_cosine_distance(activation_dict):
    layer_distances = {}

    for layer_idx, act in activation_dict.items():
        n, h, w, c = act.shape # (N, H, W, C)

        act = act.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        feature_map = act.reshape(n, c, -1) # (N, C, H*W)
        flat_feature_map = feature_map.permute(1, 0, 2).reshape(c, -1) # (C, N, H*W) -> (C, N*H*W)
        
        norm_feature_map = F.normalize(flat_feature_map, p=2, dim=1)
        similarity_matrix = torch.mm(norm_feature_map, norm_feature_map.T)
        
        cosine_distance = 1. - similarity_matrix
        layer_distances[layer_idx] = cosine_distance.mean().item() / 2.

    return layer_distances


def main(
    model_names, data_dir, 
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    avg_cosine_distances={}

    for model_name in model_names:
        model = _load_model(model_name).to(device).eval()

        data_loader = _get_dataloader(data_dir)

        activations = {}
        hooks = _register_hook(
            model=model, model_name=model_name, activation_dict=activations
        )

        total_distances = None
        num_batches = 0

        with torch.no_grad():
            for imgs, _ in tqdm(data_loader):
                imgs = imgs.to(device)
                model(imgs)

                cosine_distances = _compute_cosine_distance(activations)

                if total_distances is None:
                    total_distances = {k: 0. for k in cosine_distances.keys()}
                
                for k, v in cosine_distances.items():
                    total_distances[k] += v

                num_batches += 1
                activations.clear()
        
        for h in hooks:
            h.remove()

        sorted_layers = sorted(total_distances.keys())
        avg_cosine_distance = [total_distances[k] / num_batches for k in sorted_layers]
        avg_cosine_distances[model_name] = avg_cosine_distance

    plt.figure(figsize=(8, 5))
    
    styles = {
        # 'vit_tiny': {'color': 'green', 'marker': 'square', 'label': 'ViT MAE'}, 
        'convnext_v1_atto': {'color': 'red', 'marker': '^', 'label': 'ConvNeXt V1 Atto FCMAE'}, 
        'convnext_v2_atto': {'color': 'blue', 'marker': 'D', 'label': 'ConvNeXt V2 Atto FCMAE'}
    }
    
    for name, avg_cosine_distance in avg_cosine_distances.items():
        x_axis = np.linspace(0, 1, len(avg_cosine_distance))
        style = styles.get(name, {})
        plt.plot(x_axis, avg_cosine_distance, linewidth=2, markersize=8, **style)
    
    plt.xlabel('Normalized Layer Index')
    plt.ylabel('Feature Cosine Distance')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig('./assets/results/feature_cosine_distance.png')


if __name__ == '__main__':
    main(
        model_names=['convnext_v1_atto', 'convnext_v2_atto'], 
        data_dir='./data/valid/'
    )