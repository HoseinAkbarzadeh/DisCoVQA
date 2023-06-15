
# third-party imports
import torch.nn as nn

from models.components.swin_transformer import SwinTransformer3D
from models.components.i3d_head import I3DHead


class SwinVideo(nn.Module):
    def __init__(self, patch_size=(2,4,4),
                        embed_dim=96,
                        depths=[2, 2, 6, 2],
                        num_heads=[3, 6, 12, 24],
                        window_size=(8,7,7),
                        mlp_ratio=4.,
                        qkv_bias=True,
                        qk_scale=None,
                        drop_rate=0.,
                        attn_drop_rate=0.,
                        drop_path_rate=0.2,
                        patch_norm=True,
                        in_channels=768,
                        num_classes=400,
                        spatial_type='avg',
                        dropout_ratio=0.5) -> None:
        super(SwinVideo, self).__init__()

        self.backbone = SwinTransformer3D(patch_size=patch_size,
                                          embed_dim=embed_dim,
                                          depths=depths,
                                          num_heads=num_heads,
                                          window_size=window_size,
                                          mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias,
                                          qk_scale=qk_scale,
                                          drop_rate=drop_rate,
                                          attn_drop_rate=attn_drop_rate,
                                          drop_path_rate=drop_path_rate,
                                          patch_norm=patch_norm)
        
        self.cls_head = I3DHead(num_classes=num_classes,
                                in_channels=in_channels,
                                spatial_type=spatial_type,
                                dropout_ratio=dropout_ratio)
        
    def forward(self, x):
        x, tau = self.backbone(x)
        return self.cls_head(x), tau
    