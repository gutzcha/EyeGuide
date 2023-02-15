import torch
import torch.nn as nn
from patch_embed import PatchEmbed
from timm.models.vision_transformer import Block
from functools import partial
import numpy as np

class MaskedAutoencoderTDCNN(nn.Module):
    """ Masked Autoencoder with VisionTransformer and TDCNN backbone
      """

    def __init__(self, img_size=np.array([64, 478]), in_chans=2,
                 embed_dim=32, depth=6, num_heads=8,
                 decoder_embed_dim=64, decoder_depth=8, decoder_num_heads=8,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, patch_channels=128,
                 patch_filter_width=None):
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed(img_size, in_chans, embed_dim,
                                      channels=patch_channels, filter_widths=patch_filter_width)
        self.num_frames, self.num_landmarks = img_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_landmarks + 1, embed_dim), requires_grad=True)


        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_landmarks + 1, decoder_embed_dim), requires_grad=True)


        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, img_size[0] * in_chans, bias=True)

        self.initialize_weights()

    def initialize_weights(self):

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.patch_embed.tdcnn_model.weight.data

        w = self.pos_embed.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        w = self.decoder_pos_embed.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, n_frames, n_landmarks, n_dims) (N, 64, 478, 2)
        x: (N, n_landmarks, embed_dim)

        I am expecting the images to already be ready for patch embedding with correct size
        so no need to reshape
        """
        # p = self.patch_embed.patch_size[0]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        #
        # h = w = imgs.shape[2] // p
        # x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        # x = torch.einsum('nchpwq->nhwpqc', x)
        imgs = imgs.reshape(shape=(imgs.shape[0], self.num_landmarks,self.in_chans*self.num_frames))
        return imgs

    def unpatchify(self, x):
        """
        x: (N, img_size[0]*img_size[1] * in_chans)
        imgs: (N, n_frames, n_landmarks, n_dims)
        """
        # p = self.patch_embed.patch_size[0]
        # h = w = int(x.shape[1] ** .5)
        # assert h * w == x.shape[1]

        imgs = x.reshape(shape=(x.shape[0], self.img_size[0], self.img_size[1], self.in_chans))
        # x = torch.einsum('nhwpqc->nchpwq', x)
        # imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        # cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x


    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

def mae_vit_tdcnn(**kwargs):
    model = MaskedAutoencoderTDCNN(

        img_size=np.array([64, 478]), in_chans=2,
        embed_dim=32, depth=6, num_heads=8,
        decoder_embed_dim=64, decoder_depth=8, decoder_num_heads=8,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=False, patch_channels=128,
        patch_filter_width=None,**kwargs)
    return model

if __name__ == '__main__':
    n_frames = 32 * 2
    n_landmarks = 478
    dims = 2
    test_input = torch.rand(1, n_frames, n_landmarks, dims)
    filter_widths = [3, 3, 3]
    dropout = 0.25
    channels = 64
    out_dim = 32
    out_frames = 1

    ####
    img_size = np.array([n_frames, n_landmarks])
    in_chans = dims
    embed_dim = out_dim
    model = mae_vit_tdcnn()
    loss, pred, mask = model(test_input)
    print(test_input.shape)
    print(loss)
    pred = model.unpatchify(pred)
    print(pred.shape)
