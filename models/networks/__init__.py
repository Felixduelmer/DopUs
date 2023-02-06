from .vesnet import *
from .unet import *
from .unet_rnn import *
from .dopus_v0 import *
from .dopus_v1 import *
from .dopus_v2 import *
from .dopus_v3 import *
from .dopus_v4 import *


def get_network(name, in_channels=2, feature_scale=4):
    model_instance = _get_model_instance(name)

    if name in ['unet', 'unetrnn']:
        model_instance = model_instance(in_channels=in_channels, feature_scale=feature_scale)
    elif name in ['vesnet']:
        model_instance = model_instance(in_channels=in_channels, feature_scale=feature_scale)
    elif name in ['dopusv0', 'dopusv1', 'dopusv2', 'dopusv3', 'dopusv4']:
        model_instance = model_instance(in_channels=in_channels, feature_scale=feature_scale)
    else:
        raise 'Model {} not available'.format(name)

    return model_instance


def _get_model_instance(name):
    return {
        'vesnet': VesNet,
        'unet': UNet,
        'unetrnn': UNetRNN,
        'dopusv0': DopUsV0,
        'dopusv1': DopUsV1,
        'dopusv2': DopUsV2,
        'dopusv3': DopUsV3,
        'dopusv4': DopUSV4,

    }[name]
