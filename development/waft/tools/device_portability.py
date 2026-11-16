"""Keep frozen TorchScript tensors on the input device, including after load."""
import torch


def make_devices_dynamic(module):
    graph = module.graph
    first = next(graph.nodes())
    device = graph.create('prim::device', [list(graph.inputs())[1]])
    device.output().setType(torch._C.DeviceObjType.get())
    device.insertBefore(first)
    none = graph.insertConstant(None)
    none.node().moveBefore(first)
    false = graph.insertConstant(False)
    false.node().moveBefore(first)

    def visit(block):
        for node in list(block.nodes()):
            if node.kind() == 'prim::Constant' and str(node.output().type()) == 'Device':
                node.output().replaceAllUsesWith(device.output())
                node.destroy()
            elif (node.kind() == 'prim::Constant' and node.hasAttribute('value')
                  and node.kindOf('value') == 't' and node.t('value').dim() > 0):
                # freeze() turns weights and normalization arrays into constants.
                # Module.to() cannot migrate them. Keep scalar shape arithmetic on
                # CPU, but place every non-scalar tensor beside the input at use.
                transfer = graph.create('aten::to', [node.output(), device.output(),
                                        none, false, false])
                transfer.output().setType(torch._C.TensorType.get())
                transfer.insertAfter(node)
                node.output().replaceAllUsesWith(transfer.output())
                transfer.replaceInput(0, node.output())
            else:
                for child in node.blocks():
                    visit(child)

    visit(graph)
    torch._C._jit_pass_lint(graph)
