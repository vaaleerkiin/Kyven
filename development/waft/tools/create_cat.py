"""Run with NukeX: Nuke16.0.exe --nukex -i --safe -t tools/create_cat.py"""
from pathlib import Path
import os
import nuke

root = Path(__file__).resolve().parents[1]
(root / 'WAFT/Cattery/WAFT').mkdir(parents=True, exist_ok=True)
node = nuke.nodes.CatFileCreator()
variant = os.environ.get('WAFT_BUILD_VARIANT', 'FP16')
input_name, output_name = ('WAFT_FP16.pt', 'WAFT.cat') if variant == 'FP16' else ('WAFT.pt', 'WAFT_FP32.cat')
node['torchScriptFile'].setValue((root / 'build' / input_name).as_posix())
node['catFile'].setValue((root / 'WAFT/Cattery/WAFT' / output_name).as_posix())
node['channelsIn'].setValue('rgba.red, rgba.green, rgba.blue, forward.u, forward.v, backward.u')
node['channelsOut'].setValue('rgba.red, rgba.green')
node['modelId'].setValue('WAFT-a1, Princeton, 5 iterations, ' + variant + ', device-portable v5')
node['outputScale'].setValue(1.0)
nuke.execute(node, 1, 1)
print('CAT_CREATED', node['catFile'].value())
