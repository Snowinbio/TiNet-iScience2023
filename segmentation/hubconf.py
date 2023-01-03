import torch.nn as nn
from simplecv.module import fpn
from simplecv.util import checkpoint

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

dependencies = ['torch']

from module.foroptFPN import ForOptFPN

def ForOptFPN_resnet50(pretrained=False, progress=True):
    model_cfg = dict(
        type='ForOptFPN',
        params=dict(
            resnet_encoder=dict(
                resnet_type='resnet50',
                include_conv5=True,
                batchnorm_trainable=True,
                pretrained=True,
                freeze_at=0,
                # 8, 16 or 32
                output_stride=32,
                with_cp=(False, False, False, False),
                stem3_3x3=False,
            ),
            fpn=dict(
                in_channels_list=(256, 512, 1024, 2048),
                out_channels=256,
                conv_block=fpn.default_conv_block,
                top_blocks=None,
            ),
            scene_relation=dict(
                in_channels=2048,
                channel_list=(256, 256, 256, 256),
                out_channels=256,
                shared_scene=False,
            ),
            decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                norm_fn=nn.BatchNorm2d,
                num_groups_gn=None
            ),
            num_classes=2,
        )
    )

    model = ForOptFPN(model_cfg['params'])
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[''], progress=progress)
        model_state_dict = state_dict[checkpoint.CheckPoint.MODEL]
        model.load_state_dict(model_state_dict)
        model.eval()
    else:
        return model
