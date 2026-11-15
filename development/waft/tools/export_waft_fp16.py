"""Export a genuinely FP16 CUDA model, with float32 flow accumulation/warping.

The .cat accepts/returns float32 images; its weights and convolution/attention
activations are float16. Nuke's halfPrecision knob must remain OFF.
"""
import json
from pathlib import Path
import time
import types

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from export_waft import ROOT, make_model, make_devices_dynamic


def efficient_attention(self, x, attn_bias=None):
    b, n, c = x.shape
    qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    if hasattr(self, 'q_norm'):
        q, k = self.q_norm(q), self.k_norm(k)
    x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    x = x.transpose(1, 2).reshape(b, n, c)
    return self.proj_drop(self.proj(x))


class HalfPair(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, frames):
        from utils.utils import Padder, coords_grid
        # Cast explicitly inside the graph so Inference never has to rewrite it.
        frames = frames.to(dtype=torch.float16)
        m = self.model
        image1 = m.normalize_image(frames[:, :3] * 255.0)
        image2 = m.normalize_image(frames[:, 3:6] * 255.0)
        padder = Padder(image1.shape, factor=112)
        image1, image2 = padder.pad(image1), padder.pad(image2)
        n, _, h, w = image1.shape
        pre1 = m.da_feature(image1)['out']
        pre2 = m.da_feature(image2)['out']
        rgb1 = m.fnet(torch.cat((pre1, image1), 1))[0]
        rgb2 = m.fnet(torch.cat((pre2, image2), 1))[0]
        pre1 = F.interpolate(pre1, scale_factor=0.5, mode='bilinear', align_corners=True)
        pre2 = F.interpolate(pre2, scale_factor=0.5, mode='bilinear', align_corners=True)
        fmap1 = m.fmap_conv(torch.cat((rgb1, pre1), 1))
        fmap2 = m.fmap_conv(torch.cat((rgb2, pre2), 1))
        net = m.hidden_conv(torch.cat((fmap1, fmap2), 1))
        flow = torch.zeros(n, 2, h // 2, w // 2, device=image1.device, dtype=torch.float32)
        base = coords_grid(n, h // 2, w // 2, image1.device)
        for _ in range(5):
            coords = base + flow
            xgrid = 2.0 * coords[:, 0] / (w // 2 - 1) - 1.0
            ygrid = 2.0 * coords[:, 1] / (h // 2 - 1) - 1.0
            grid = torch.stack((xgrid, ygrid), -1)
            warped = F.grid_sample(fmap2.float(), grid, align_corners=True).half()
            x = m.warp_linear(torch.cat((fmap1, warped, net, flow.half()), 1))
            refined = m.refine_net(x)['out']
            net = m.refine_transform(torch.cat((refined, net), 1))
            flow = flow + m.flow_head(net)[:, :2].float()
        # Only the final iteration needs full-resolution convex upsampling.
        mask = (0.25 * m.upsample_weight(net)).float().reshape(n, 1, 9, 2, 2, h // 2, w // 2)
        mask = torch.softmax(mask, dim=2)
        patches = F.unfold(2.0 * flow, [3, 3], padding=1).reshape(n, 2, 9, 1, 1, h // 2, w // 2)
        out = (mask * patches).sum(dim=2).permute(0, 1, 4, 2, 5, 3).reshape(n, 2, h, w)
        return padder.unpad(out).float()


def main():
    torch.set_num_threads(6)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required to build/validate the FP16 model')
    model = make_model(ROOT / 'checkpoints/waft-a1.pth').cuda().half().eval()
    for layer in model.modules():
        if hasattr(layer, 'qkv') and hasattr(layer, 'num_heads'):
            layer.forward = types.MethodType(efficient_attention, layer)
    wrapper = HalfPair(model).eval()
    with torch.inference_mode():
        x = torch.rand(1, 6, 128, 192, device='cuda')
        print('Tracing FP16 + SDPA, five iterations', flush=True)
        traced = torch.jit.trace(wrapper, x, check_trace=False)
        traced = torch.jit.freeze(traced, optimize_numerics=False)
        make_devices_dynamic(traced)
        path = ROOT / 'build/WAFT_FP16.pt'
        traced.save(str(path))
        del traced, wrapper, model
        torch.cuda.empty_cache()
        traced = torch.jit.load(str(path), map_location='cuda')
        reference = torch.jit.load(str(ROOT / 'build/WAFT.pt'), map_location='cuda')
        checks = []
        for h, w in [(160, 240), (288, 512)]:
            images = [cv2.cvtColor(cv2.imread(str(ROOT / f'build/test_input.{i:04d}.png')), cv2.COLOR_BGR2RGB)
                      for i in [2, 3]]
            images = [cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR) for im in images]
            x = torch.from_numpy(np.concatenate(images, 2).transpose(2, 0, 1).copy()).unsqueeze(0).float().cuda()/255.0
            expected = reference(x)
            actual = traced(x)
            assert torch.isfinite(actual).all()
            center_error = (actual-expected).abs()[:, :, h//4:3*h//4, w//4:3*w//4]
            checks.append({'shape': [h, w], 'mean_error': center_error.mean().item(),
                           'max_error': center_error.max().item()})
            assert center_error.mean() < 0.2, checks[-1]
            print('FP16_CHECK', checks[-1], flush=True)
        del reference
        torch.cuda.empty_cache()
        # Warm kernels before measuring the live tensor allocation peak.
        x = torch.rand(1, 6, 288, 512, device='cuda')
        traced(x)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(3):
            result = traced(x)
        torch.cuda.synchronize()
        report = {'precision': 'FP16 weights/activations, FP32 flow', 'iterations': 5,
                  'attention': 'scaled_dot_product_attention', 'checks': checks,
                  'gpu': torch.cuda.get_device_name(), 'weights_bytes': path.stat().st_size,
                  'seconds_per_pair_512x288': (time.perf_counter()-start)/3,
                  'peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2}
        (ROOT / 'build/fp16_validation.json').write_text(json.dumps(report, indent=2))
        print('FP16_EXPORT_PASSED', report, flush=True)


if __name__ == '__main__':
    main()
