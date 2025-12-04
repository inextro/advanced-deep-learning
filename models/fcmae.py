import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from .pretrain_convnext_v1 import V1Block, convnext_v1_atto
from .pretrain_convnext_v2 import convnext_v2_atto


class FCMAE(nn.Module):
    def __init__(
        self, 
        encoder, 
        img_size=224, 
        in_chans=3, 
        depth=[2, 2, 6, 2], # Atto default
        dims=[40, 80, 160, 320], # Atto default
        decoder_depth=1, 
        decoder_embed_dim=320, 
        patch_size=32, 
        mask_ratio=0.6, 
        norm_pix_loss=False
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = dims[-1]

        self.img_size = img_size
        self.in_chans = in_chans
        self.decoder_depth = decoder_depth
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (img_size // patch_size) ** 2
        self.norm_pix_loss = norm_pix_loss

        # encoder
        self.encoder = encoder

        # decoder
        self.proj = nn.Conv2d(
            in_channels=dims[-1], 
            out_channels=decoder_embed_dim, 
            kernel_size=1

        )
        decoder = [V1Block(dim=decoder_embed_dim) for _ in range(decoder_depth)]
        self.decoder = nn.Sequential(*decoder)

        # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))

        # pred
        self.pred = nn.Conv2d(
            in_channels=decoder_embed_dim, 
            out_channels=patch_size ** 2 * in_chans, 
            kernel_size=1
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w = m.weight.data
            trunc_normal_(w.view([w.shape[0], -1]))
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)
        if hasattr(self, 'mask_token'):
            torch.nn.init.normal_(self.mask_token, std=0.02)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 *3))

        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        
        return imgs

    def gen_random_mask(self, x, mask_ratio):
        N = x.shape[0]
        L = (x.shape[2] // self.patch_size) ** 2
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.randn(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return mask

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2

        p = int(mask.shape[1] ** 0.5)
        return mask.reshape(-1, p, p). \
            repeat_interleave(scale, dim=1). \
                repeat_interleave(scale, dim=2)

    def forward_encoder(self, imgs, mask_ratio):
        # generate random masks
        mask = self.gen_random_mask(imgs, mask_ratio)

        mask_upsampled = self.upsample_mask(mask, self.patch_size) # (N, H, W)
        mask_upsampled = mask_upsampled.unsqueeze(1).type_as(imgs) # (N, 1, H, W)

        imgs_masked = imgs * (1.0 - mask_upsampled)

        # encoding
        x = self.encoder(imgs_masked)

        return x, mask

    def forward_decoder(self, x, mask):
        x = self.proj(x)

        # append mask token
        n, c, h, w = x.shape
        mask = mask.reshape(-1, h, w).unsqueeze(1).type_as(x)
        mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        x = x * (1.0 - mask) + mask_token * mask

        # decoding
        x = self.decoder(x)

        # pred
        pred = self.pred(x)

        return pred

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: (N, 3, H, W)
        pred: (N, L, p*p*3)
        mask: (N, L), 0 is keep, 1 is remove
        """
        if len(pred.shape) == 4:
            n, c, _, _ = pred.shape
            pred = pred.reshape(n, c, -1)
            pred = torch.einsum('ncl->nlc', pred)

        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** 0.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1) # (N, L), mean loss per batch
        loss = (loss * mask).sum() / mask.sum() # mean loss on removed patches

        return loss

    def forward(self, imgs, labels=None, mask_ratio=0.6):
        x, mask = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(x, mask)
        loss = self.forward_loss(imgs, pred, mask)

        return loss, pred, mask


def fcmae_convnext_v1_atto(**kwargs):
    encoder = convnext_v1_atto()
    model = FCMAE(
        encoder=encoder, 
        depth=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs
    )
    return model

def fcmae_convnext_v2_atto(**kwargs):
    encoder = convnext_v2_atto()
    model = FCMAE(
        encoder=encoder, 
        depth=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs
    )
    return model