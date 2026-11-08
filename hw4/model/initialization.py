"""Initializers used only while constructing modules before strict weight loading."""
from torch import nn

def init_module_weights(module, std=0.02):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=std)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
