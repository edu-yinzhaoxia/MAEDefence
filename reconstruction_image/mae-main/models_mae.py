# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
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
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        import numpy as np
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        # noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        #
        noise = torch.arange(0, 196, 1)
        # noise2 = torch.arange(195, 0, -2)
        # ids_shuffle = torch.cat((noise2, noise1), 0)
        # ids_shuffle = ids_shuffle.unsqueeze(0)
        # sort noise for each sample

        ids_shuffle = torch.argsort(noise, dim=0)  # ascend: small is keep, large is remove
        ids_shuffle, id = torch.sort(ids_shuffle)
        ids_shuffle = ids_shuffle.reshape(14, 14)

        if mask_ratio == 0.25:
            ids1 = []
            ids2 = []
            ids3 = []
            ids4 = []
            for i in range(0, 14, 2):
                for j in range(0, 14, 2):
                    ids1.append(int(ids_shuffle[i][j+1]))
                    ids1.append(int(ids_shuffle[i + 1][j]))
                    ids1.append(int(ids_shuffle[i + 1][j + 1]))

                    ids2.append(int(ids_shuffle[i][j]))
                    ids2.append(int(ids_shuffle[i + 1][j]))
                    ids2.append(int(ids_shuffle[i + 1][j + 1]))

                    ids3.append(int(ids_shuffle[i][j]))
                    ids3.append(int(ids_shuffle[i][j+1]))
                    ids3.append(int(ids_shuffle[i + 1][j + 1]))

                    ids4.append(int(ids_shuffle[i][j]))
                    ids4.append(int(ids_shuffle[i][j+1]))
                    ids4.append(int(ids_shuffle[i + 1][j]))

            for i in range(0, 14, 2):
                for j in range(0, 14, 2):
                    ids1.append(int(ids_shuffle[i][j]))

                    ids2.append(int(ids_shuffle[i][j+1]))

                    ids3.append(int(ids_shuffle[i+1][j]))

                    ids4.append(int(ids_shuffle[i + 1][j+1]))
            ids1 = torch.Tensor(ids1)
            ids2 = torch.Tensor(ids2)
            ids3 = torch.Tensor(ids3)
            ids4 = torch.Tensor(ids4)
            ids1 = ids1.long()
            ids2 = ids2.long()
            ids3 = ids3.long()
            ids4 = ids4.long()

            ids_shuffle1 = ids1.expand(N, -1)
            ids_shuffle2 = ids2.expand(N, -1)
            ids_shuffle3 = ids3.expand(N, -1)
            ids_shuffle4 = ids4.expand(N, -1)

            ids_restore1 = torch.argsort(ids_shuffle1, dim=1)
            ids_restore2 = torch.argsort(ids_shuffle2, dim=1)
            ids_restore3 = torch.argsort(ids_shuffle3, dim=1)
            ids_restore4 = torch.argsort(ids_shuffle4, dim=1)

            # keep the first subset
            ids_keep1 = ids_shuffle1[:, :len_keep]
            x_masked1 = torch.gather(x, dim=1, index=ids_keep1.unsqueeze(-1).repeat(1, 1, D))

            ids_keep2 = ids_shuffle2[:, :len_keep]
            x_masked2 = torch.gather(x, dim=1, index=ids_keep2.unsqueeze(-1).repeat(1, 1, D))

            ids_keep3 = ids_shuffle3[:, :len_keep]
            x_masked3 = torch.gather(x, dim=1, index=ids_keep3.unsqueeze(-1).repeat(1, 1, D))

            ids_keep4 = ids_shuffle4[:, :len_keep]
            x_masked4 = torch.gather(x, dim=1, index=ids_keep4.unsqueeze(-1).repeat(1, 1, D))

            # generate the binary mask: 0 is keep, 1 is remove
            mask1 = torch.ones([N, L], device=x.device)
            mask1[:, :len_keep] = 0

            mask2 = torch.ones([N, L], device=x.device)
            mask2[:, :len_keep] = 0

            mask3 = torch.ones([N, L], device=x.device)
            mask3[:, :len_keep] = 0

            mask4 = torch.ones([N, L], device=x.device)
            mask4[:, :len_keep] = 0
            # unshuffle to get the binary mask
            mask1 = torch.gather(mask1, dim=1, index=ids_restore1)

            mask2 = torch.gather(mask2, dim=1, index=ids_restore2)

            mask3 = torch.gather(mask3, dim=1, index=ids_restore3)

            mask4 = torch.gather(mask4, dim=1, index=ids_restore4)

            return x_masked1, mask1, x_masked2, mask2, x_masked3, mask3, x_masked4, mask4, ids_restore1, ids_restore2, ids_restore3, ids_restore4


        elif mask_ratio == 0.50:
            ids1 = []
            ids2 = []
            for i in range(0, 14, 2):
                for j in range(0, 14, 2):
                    ids1.append(int(ids_shuffle[i][j]))
                    ids1.append(int(ids_shuffle[i+1][j+1]))

                    ids2.append(int(ids_shuffle[i][j+1]))
                    ids2.append(int(ids_shuffle[i + 1][j]))
            for i in range(0, 14, 2):
                for j in range(0, 14, 2):
                    ids1.append(int(ids_shuffle[i][j+1]))
                    ids1.append(int(ids_shuffle[i + 1][j]))

                    ids2.append(int(ids_shuffle[i][j]))
                    ids2.append(int(ids_shuffle[i+1][j+1]))
            ids1 = torch.Tensor(ids1)
            ids2 = torch.Tensor(ids2)

            ids1 = ids1.long()
            ids2 = ids2.long()


            ids_shuffle1 = ids1.expand(N, -1)
            ids_shuffle2 = ids2.expand(N, -1)


            ids_restore1 = torch.argsort(ids_shuffle1, dim=1)
            ids_restore2 = torch.argsort(ids_shuffle2, dim=1)


            # keep the first subset
            ids_keep1 = ids_shuffle1[:, :len_keep]
            x_masked1 = torch.gather(x, dim=1, index=ids_keep1.unsqueeze(-1).repeat(1, 1, D))

            ids_keep2 = ids_shuffle2[:, :len_keep]
            x_masked2 = torch.gather(x, dim=1, index=ids_keep2.unsqueeze(-1).repeat(1, 1, D))


            # generate the binary mask: 0 is keep, 1 is remove
            mask1 = torch.ones([N, L], device=x.device)
            mask1[:, :len_keep] = 0

            mask2 = torch.ones([N, L], device=x.device)
            mask2[:, :len_keep] = 0

            # unshuffle to get the binary mask
            mask1 = torch.gather(mask1, dim=1, index=ids_restore1)

            mask2 = torch.gather(mask2, dim=1, index=ids_restore2)

            return x_masked1, mask1, x_masked2, mask2, ids_restore1, ids_restore2

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        if mask_ratio == 0.25:
            x1, mask1,  x2, mask2,  x3, mask3,  x4, mask4,ids_restore1,ids_restore2,ids_restore3,ids_restore4 = self.random_masking(x, mask_ratio)

            # append cls token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x1.shape[0], -1, -1)
            x1 = torch.cat((cls_tokens, x1), dim=1)

            cls_tokens = cls_token.expand(x2.shape[0], -1, -1)
            x2 = torch.cat((cls_tokens, x2), dim=1)

            cls_tokens = cls_token.expand(x3.shape[0], -1, -1)
            x3 = torch.cat((cls_tokens, x3), dim=1)

            cls_tokens = cls_token.expand(x4.shape[0], -1, -1)
            x4 = torch.cat((cls_tokens, x4), dim=1)
            # apply Transformer blocks
            for blk in self.blocks:
                x1 = blk(x1)
            x1 = self.norm(x1)

            for blk in self.blocks:
                x2 = blk(x2)
            x2 = self.norm(x2)

            for blk in self.blocks:
                x3 = blk(x3)
            x3 = self.norm(x3)

            for blk in self.blocks:
                x4 = blk(x4)
            x4 = self.norm(x4)

            return x1, mask1, x2, mask2,x3, mask3,x4, mask4, ids_restore1, ids_restore2,ids_restore3, ids_restore4
        else:
            x1, mask1,  x2, mask2, ids_restore1, ids_restore2 = self.random_masking(x, mask_ratio)

            # append cls token
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            cls_tokens = cls_token.expand(x1.shape[0], -1, -1)
            x1 = torch.cat((cls_tokens, x1), dim=1)

            cls_tokens = cls_token.expand(x2.shape[0], -1, -1)
            x2 = torch.cat((cls_tokens, x2), dim=1)

            # apply Transformer blocks
            for blk in self.blocks:
                x1 = blk(x1)
            x1 = self.norm(x1)

            for blk in self.blocks:
                x2 = blk(x2)
            x2 = self.norm(x2)
            return x1, mask1, x2, mask2, ids_restore1, ids_restore2
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
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio):
        if mask_ratio ==0.25:
            latent1, mask1, latent2, mask2, latent3, mask3, latent4, mask4, ids_restore1,ids_restore2,ids_restore3,ids_restore4 = self.forward_encoder(imgs, mask_ratio)
            pred1 = self.forward_decoder(latent1, ids_restore1)  # [N, L, p*p*3]
            loss1 = self.forward_loss(imgs, pred1, mask1)
            pred2 = self.forward_decoder(latent2, ids_restore2)  # [N, L, p*p*3]
            loss2 = self.forward_loss(imgs, pred2, mask2)
            pred3 = self.forward_decoder(latent3, ids_restore3)  # [N, L, p*p*3]
            loss3 = self.forward_loss(imgs, pred3, mask3)
            pred4 = self.forward_decoder(latent4, ids_restore4)  # [N, L, p*p*3]
            loss4 = self.forward_loss(imgs, pred4, mask4)
            return loss1, pred1, mask1, loss2, pred2, mask2,loss3, pred3, mask3,loss4, pred4, mask4
        else:
            latent1, mask1, latent2, mask2,ids_restore1, ids_restore2 = self.forward_encoder(
                imgs, mask_ratio)
            pred1 = self.forward_decoder(latent1, ids_restore1)  # [N, L, p*p*3]
            loss1 = self.forward_loss(imgs, pred1, mask1)
            pred2 = self.forward_decoder(latent2, ids_restore2)  # [N, L, p*p*3]
            loss2 = self.forward_loss(imgs, pred2, mask2)
            return loss1, pred1, mask1, loss2, pred2, mask2


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
