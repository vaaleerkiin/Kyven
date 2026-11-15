"""Export pretrained WAFT-a1 to a dynamic-resolution Nuke TorchScript model.

Six input planes: RGB(frame), RGB(target). Two output planes: dx, dy.
Pixel ranges are sRGB 0..1; displacements are in source image pixels.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import types

import torch
from torch import Tensor, nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'WAFT-source'
sys.path.insert(0, str(SOURCE))


@torch.jit.script
def position_embedding(pos: Tensor, tokens: Tensor, image: Tensor,
                       patch: int, class_token: bool, offset: float) -> Tensor:
    height, width = image.shape[-2], image.shape[-1]
    extra = 1 if class_token else 0
    count = pos.shape[1] - extra
    if tokens.shape[1] - extra == count and height == width:
        return pos.to(tokens.dtype)
    side = int(math.sqrt(count))
    grid = pos[:, extra:].float().reshape(1, side, side, pos.shape[-1]).permute(0, 3, 1, 2)
    sy = (float(height // patch) + offset) / float(side)
    sx = (float(width // patch) + offset) / float(side)
    grid = F.interpolate(grid, scale_factor=[sy, sx], mode='bicubic', align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, -1, pos.shape[-1])
    if class_token:
        grid = torch.cat((pos[:, :1].float(), grid), dim=1)
    return grid.to(tokens.dtype)


def refine_forward(self, image):
    h, w = image.shape[-2:]
    x = self.patch_embed(image)
    x = x + position_embedding(self.pos_embed, x, image, self.patch_size, False, 0.0)
    outputs = []
    for i, block in enumerate(self.blks):
        x = block(x)
        if i in self.idx:
            outputs.append([x])
    out, p1, p2, p3, p4 = self.dpt_head(outputs, h // self.patch_size, w // self.patch_size, True)
    out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
    return {'out': out, 'path_1': p1, 'path_2': p2, 'path_3': p3, 'path_4': p4}


def prepare_tokens(self, image, masks=None):
    x = self.patch_embed(image)
    if masks is not None:
        x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
    x = x + position_embedding(self.pos_embed, x, image, self.patch_size, True, self.interpolate_offset)
    if self.register_tokens is not None:
        x = torch.cat((x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1)
    return x


class PairCore(nn.Module):
    def __init__(self, model, iterations):
        super().__init__()
        self.model = model
        self.iterations = iterations

    def forward(self, frames):
        return self.model(frames[:, :3] * 255.0, frames[:, 3:6] * 255.0,
                          iters=self.iterations)['flow'][-1]


def patch_source():
    path = SOURCE / 'model/__init__.py'
    text = path.read_text(encoding='utf-8')
    text = text.replace('from model.waft_a2 import WAFTv2\n', '')
    text = text.replace("elif args.algorithm == 'waft-a2':\n        model", "elif args.algorithm == 'waft-a2':\n        from model.waft_a2 import WAFTv2\n        model")
    path.write_text(text, encoding='utf-8')
    # Preserve symbolic output dimensions when tracing the DPT heads.
    path = SOURCE / 'thirdparty/DepthAnythingV2/depth_anything_v2/dpt.py'
    text = path.read_text(encoding='utf-8')
    text = text.replace('(int(patch_h * 14), int(patch_w * 14))', '(patch_h * 14, patch_w * 14)')
    path.write_text(text, encoding='utf-8')


def make_model(checkpoint):
    patch_source()
    import timm
    import model.backbone.waftv1_dav2 as depth
    original_depth_init = depth.DepthAnythingFeature.__init__
    def depth_init(self, encoder='vits', pretrained=True):
        original_depth_init(self, encoder, pretrained=False)
    depth.DepthAnythingFeature.__init__ = depth_init
    original_create = timm.create_model
    def create_model(*args, **kwargs):
        kwargs['pretrained'] = False
        return original_create(*args, **kwargs)
    timm.create_model = create_model
    from model.waft_a1 import ViTWarpV8
    args = argparse.Namespace(dav2_backbone='vits', network_backbone='vits', iters=5)
    model = ViTWarpV8(args).eval()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if 'state_dict' in state:
        state = state['state_dict']
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    # Strict loading: no random or missing parameters are permitted.
    model.load_state_dict(state, strict=True)
    model.refine_net.forward = types.MethodType(refine_forward, model.refine_net)
    backbone = model.da_feature.depth_anything.pretrained
    backbone.prepare_tokens_with_masks = types.MethodType(prepare_tokens, backbone)
    for layer in model.modules():
        if hasattr(layer, 'fused_attn'):
            layer.fused_attn = False
    return model


def make_devices_dynamic(module):
    graph = module.graph
    input_value = list(graph.inputs())[1]
    device = graph.create('prim::device', [input_value])
    device.output().setType(torch._C.DeviceObjType.get())
    device.insertBefore(next(graph.nodes()))
    def visit(block):
        for node in list(block.nodes()):
            if node.kind() == 'prim::Constant' and str(node.output().type()) == 'Device':
                node.output().replaceAllUsesWith(device.output())
                node.destroy()
            else:
                for child in node.blocks():
                    visit(child)
    visit(graph)
    torch._C._jit_pass_lint(graph)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/waft-a1.pth')
    parser.add_argument('--output', type=Path, default=ROOT / 'build/WAFT.pt')
    args = parser.parse_args()
    torch.set_num_threads(6)
    torch.manual_seed(42)
    model = make_model(args.checkpoint)
    core = PairCore(model, 5).eval()
    example = torch.rand(1, 6, 128, 192)
    with torch.inference_mode():
        print('Tracing pretrained WAFT-a1, 5 iterations', flush=True)
        traced = torch.jit.trace(core, example, check_trace=False)
        traced = torch.jit.freeze(traced)
        make_devices_dynamic(traced)
        errors = []
        for h, w in [(128, 192), (241, 353)]:
            x = torch.rand(1, 6, h, w)
            expected = core(x)
            actual = traced(x)
            torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.002)
            errors.append({'size': [h, w], 'max_error': (actual-expected).abs().max().item()})
            print('Dynamic size validated:', errors[-1], flush=True)
        # Flip/sign conversion belongs to the package gizmo, so .cat has the same
        # raw image/flow convention as the original RAFT Cattery model.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        traced.save(str(args.output))
        if torch.cuda.is_available():
            gpu = torch.jit.load(str(args.output), map_location='cuda')
            x = example.cuda()
            actual = gpu(x).cpu()
            expected = core(example)
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
            print('CUDA validated', flush=True)
        (ROOT / 'build/export_validation.json').write_text(json.dumps({
            'torch': torch.__version__, 'iterations': 5, 'strict_weights': True,
            'resolution_checks': errors, 'cuda': torch.cuda.is_available(),
        }, indent=2))
    print('Saved', args.output, flush=True)


if __name__ == '__main__':
    main()
