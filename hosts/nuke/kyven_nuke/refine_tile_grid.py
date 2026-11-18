"""Live native grid and yellow guidance while CPU RUN/SKIP colors update."""

from __future__ import annotations

from typing import Any


def _axis_lines(position: str, length: str) -> str:
    tile = 'max(1,parent.tile_preview_size)'
    step = 'max(1,parent.tile_preview_size-floor(parent.tile_overlap))'
    last = f'(floor((({length})-({tile}))/({step}))*({step}))'
    remainder = f'(({position})-({step})*floor(({position})/({step})))'
    end_position = f'(({position})-({tile})+3)'
    end_remainder = f'({end_position}-({step})*floor({end_position}/({step})))'
    outer = f'(({position})<3 || ({position})>=({length})-3)'
    regular_start = f'(({position})<({last})+3 && {remainder}<3)'
    regular_end = (f'(({position})>=({tile})-3 && ({position})<({last})+({tile}) '
                   f'&& {end_remainder}<3)')
    final_start = f'(({position})>=({length})-({tile}) && ({position})<({length})-({tile})+3)'
    return (f'(parent.tile_preview_size<=0 || ({length})<=({tile}) ? {outer} : '
            f'({outer} || {regular_start} || {regular_end} || {final_start}))')


def build_graph(nuke: Any, source: Any, base: Any) -> Any:
    """Draw squares and unknown pixels directly from the current Nuke controls."""

    grid = nuke.toNode('KyvenTilePreviewGrid')
    if grid is None:
        grid = nuke.nodes.Expression(name='KyvenTilePreviewGrid')
    grid.setInput(0, source)
    manual = (
        'max(0,min(width,floor(min(parent.processing_roi.x,parent.processing_roi.r))))',
        'max(0,min(height,floor(min(parent.processing_roi.y,parent.processing_roi.t))))',
        'max(0,min(width,ceil(max(parent.processing_roi.x,parent.processing_roi.r))))',
        'max(0,min(height,ceil(max(parent.processing_roi.y,parent.processing_roi.t))))',
    )
    valid = f'({manual[0]}<{manual[2]} && {manual[1]}<{manual[3]})'
    padding = 'max(0,floor(parent.context_padding))'
    auto = (
        f'max(0,parent.tile_preview_bounds.x-({padding}))',
        f'max(0,parent.tile_preview_bounds.y-({padding}))',
        f'min(width,parent.tile_preview_bounds.r+({padding}))',
        f'min(height,parent.tile_preview_bounds.t+({padding}))',
    )
    for index, (name, manual_value, auto_value, full) in enumerate(zip(
            ('gx0', 'gy0', 'gx1', 'gy1'), manual, auto, ('0', '0', 'width', 'height'))):
        grid[f'temp_name{index}'].setValue(name)
        grid[f'temp_expr{index}'].setValue(
            f'parent.crop_mode==1 ? ({valid} ? {manual_value} : {full}) : '
            f'(parent.crop_mode==0 && parent.tile_preview_bounds_valid ? {auto_value} : {full})')
    x_lines = _axis_lines('x-gx0', 'gx1-gx0')
    # The backend grid starts at the top-left, whereas Nuke image Y starts below.
    y_lines = _axis_lines('gy1-1-y', 'gy1-gy0')
    value = ('KyvenTilePreviewSwitch.which ? 0 : '
             '(x>=gx0 && x<gx1 && y>=gy0 && y<gy1 ? '
             f'({x_lines} || {y_lines}) : 0)')
    for index in range(3):
        grid[f'expr{index}'].setValue('1')
    grid['expr3'].setValue(value)
    premult = nuke.toNode('KyvenTilePreviewGridPremult')
    if premult is None:
        premult = nuke.nodes.Premult(name='KyvenTilePreviewGridPremult')
    premult.setInput(0, grid)
    merge = nuke.toNode('KyvenTilePreviewLiveGrid')
    if merge is None:
        merge = nuke.nodes.Merge2(name='KyvenTilePreviewLiveGrid', operation='over')
    merge.setInput(1, premult)

    guidance = nuke.toNode('KyvenTilePreviewLiveGuidance')
    if guidance is None:
        guidance = nuke.nodes.Copy(name='KyvenTilePreviewLiveGuidance')
    guidance.setInput(0, source)
    guidance.setInput(1, nuke.toNode('KyvenTrimapROI'))
    guidance['from0'].setValue('rgba.red')
    guidance['to0'].setValue('rgba.alpha')
    unknown = nuke.toNode('KyvenTilePreviewUnknown')
    if unknown is None:
        unknown = nuke.nodes.Expression(name='KyvenTilePreviewUnknown')
    unknown.setInput(0, guidance)
    for index in range(4):
        unknown[f'temp_name{index}'].setValue(grid[f'temp_name{index}'].value())
        unknown[f'temp_expr{index}'].setValue(grid[f'temp_expr{index}'].value())
    for index, color in enumerate(('1', '195.0/255', '35.0/255')):
        unknown[f'expr{index}'].setValue(color)
    unknown['expr3'].setValue(
        'x>=gx0 && x<gx1 && y>=gy0 && y<gy1 && a>0.01 && a<0.99 ? 200.0/255 : 0')
    yellow = nuke.toNode('KyvenTilePreviewUnknownPremult')
    if yellow is None:
        yellow = nuke.nodes.Premult(name='KyvenTilePreviewUnknownPremult')
    yellow.setInput(0, unknown)
    live = nuke.toNode('KyvenTilePreviewLiveTrimap')
    if live is None:
        live = nuke.nodes.Merge2(name='KyvenTilePreviewLiveTrimap', operation='over')
    live.setInput(0, base)
    live.setInput(1, yellow)
    merge.setInput(0, live)
    display = nuke.toNode('KyvenTilePreviewDisplay')
    if display is None:
        display = nuke.nodes.Switch(name='KyvenTilePreviewDisplay')
    display.setInput(0, merge)
    display.setInput(1, base)
    display['which'].setExpression('KyvenTilePreviewSwitch.which')
    return display
