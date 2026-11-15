import os
import sys
from model.waft_a1 import ViTWarpV8

def fetch_model(args):
    if args.algorithm == 'waft-a1':
        model = ViTWarpV8(args)
    elif args.algorithm == 'waft-a2':
                model = WAFTv2(args)
    else:
        raise ValueError("Unknown algorithm: {}".format(args.algorithm))
    return model