import logging
import timm
from timm.models.vision_transformer import vit_base_patch16_224
from .convpass import set_Convpass




def build_net(arch_name, pretrained):
    logging.info('Building vit_convpass network ... ')
    if arch_name == 'vit_base_patch16_224':
       model = vit_base_patch16_224(pretrained, num_classes=2)
        # model = vit_base_patch16_224(pretrained)

    set_Convpass(model, 'convpass', dim=8, s=1, xavier_init=True)
    #import pdb; pdb.set_trace()

    return model

if __name__ == '__main__':
    build_net('vit_base_patch16_224')

