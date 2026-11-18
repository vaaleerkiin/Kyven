"""Plain-English hover help for the public Kyven node controls.

Keep shared actions here and override controls whose meaning depends on the node.
This module deliberately has no Nuke dependency, so help can be checked offline.
"""
from __future__ import annotations

import re
from typing import Any

COMMON = {
    "profile": "Controls the inference memory/performance configuration. Start with balanced; try low_memory if GPU memory runs out. Quality may require more memory. This is not a guarantee of better results for every model.",
    "refresh_models": "Checks which models or runtime components are available locally and refreshes their status. Use after installing a model; this does not process an image.",
    "open_model_manager": "Opens the model and runtime manager. Use it to install the components required by the selected model and check their availability.",
    "live_mode": "Automatically processes the current frame when the timeline or relevant settings change. Useful for adjusting a result interactively; turn off to avoid background processing while working elsewhere.",
    "process_frame": "Processes only the current timeline frame with these settings. Use this to judge the result before processing a whole range.",
    "process_range": "Processes every frame from Range First through Range Last, including both endpoints. Prepare input snapshots with Cache Frames; use Refresh Frames after upstream edits.",
    "range_first": "First frame of the processing and input-cache range, inclusive. Raise it to skip earlier frames.",
    "range_last": "Last frame of the processing and input-cache range, inclusive. Lower it to process fewer frames while testing.",
    "cancel": "Requests cancellation of the current operation. Already completed cached frames are retained; cancellation may wait for the active processing step.",
    "input_cache_scope": "Chooses whether Cache Frames and Refresh Frames work on the full frame range or only the current timeline frame. It does not change the range used by Process Frame Range.",
    "cache_frames": "Saves missing or outdated input snapshots for the selected Input Frames scope. Reuses snapshots that are already valid. Run this before range processing.",
    "refresh_frames": "Re-exports input snapshots for the selected Input Frames scope, including previously cached frames. Use after changing upstream images, masks, or vectors, then process again to update the results.",
    "cache_root": "Base directory for input snapshots and generated results. Cache files are organized under Folder / Name / Version. Choose a writable location with enough free space.",
    "roto_name": "Name of this cache within Cache Folder. Nodes using the same folder, name and version share cached files; use a unique name for a separate object or shot.",
    "cache_version": "Selects the cache version, for example v001. Folder, Name and Version identify the files used by this node. Use New Cache Version to keep an earlier result while making another pass.",
    "new_cache_version": "Switches to a new cache version so another pass can be stored separately. Earlier cache files remain on disk.",
    "cache_location": "Shows the resolved location of this node's cached files. Use it to locate the generated files on disk.",
    "cache_folder": "Location of this node's cached files on disk. Cached results can be reused without running inference again.",
    "previous_cache_location": "Location of the previous cache version, retained so you can find an earlier result after starting a new version.",
    "load_cache": "Loads available results from the selected cache into this node. Use to reuse a saved pass without running inference again.",
    "create_matte_read": "Creates a separate Nuke Read node pointing at the cached matte sequence. Process frames first; the Read depends on those files remaining on disk.",
    "create_result_read": "Creates a separate Nuke Read node for the cached result sequence. Process frames first; moving or deleting the cache will break this Read.",
    "delete_node_cache": "Deletes cached files for this node's cache. Results must be regenerated afterward, and nodes or Reads sharing that cache may lose their files.",
    "delete_all_cache": "Deletes the Kyven cache targeted by this action, potentially affecting other Kyven nodes. Cached results will need to be regenerated. Check the confirmation before continuing.",
    "kyven_status": "Reports readiness, processing progress or the latest error. If a result does not update, read this field for the next action.",
    "start_server_control": "Starts or connects to the local Kyven inference server used by the nodes. Use if the server is stopped or unavailable.",
    "stop_server_control": "Stops the local Kyven server. Other Kyven nodes using that server will also be affected; start it again before further inference.",
    "mask_channel": "Legacy serialized control. Artist mask inputs always use alpha.",
    "crop_mode": "Auto finds the useful mask area and adds Context Padding. Manual uses your Manual ROI rectangle. Full processes the entire source. A tighter crop can save work, but keep enough surrounding image for the model.",
    "context_padding": "Extra source-image pixels around the automatically detected mask region. Increase to give the model more surrounding context; decrease for a tighter crop. Applies in Auto crop mode.",
    "processing_roi": "Rectangle used in Manual crop mode, in source-image coordinates. Enclose the area to process and enough surrounding context. Processing returns to the original source canvas.",
    "reset_roi": "Resets the manual processing rectangle to the full Source image. Use if the crop is misplaced or too small.",
}

SEGMENT = {
    "model": "Selects the SAM 2 segmentation model that turns your object points into a mask. Larger models generally need more memory and processing time. Start with the default and improve the prompts before switching models.",
    "positive_enabled": "Uses positive points to tell SAM which object to keep. Put points inside the subject; add another on a missed part rather than on its boundary.",
    "negative_enabled": "Uses negative points to tell SAM which areas to exclude. Enable and place points on unwanted background or another object that leaks into the mask.",
    "add_positive_point": "Adds another foreground point. Place it inside a missing part of the object to help include that part in the mask.",
    "add_negative_point": "Adds another exclusion point. Place it on an unwanted area to help remove that area from the mask.",
    "remove_positive_point": "Removes the last positive point. Use to undo an extra foreground prompt that makes selection worse.",
    "remove_negative_point": "Removes the last negative point. Use if an exclusion point cuts away part of the wanted subject.",
    "box_enabled": "Crops the image to Processing ROI before SAM runs. Use a smaller region to focus on one object. Anything outside the crop cannot be recovered in that pass.",
    "prompt_box": "Processing crop in source-image coordinates. Keep the whole subject inside it, including motion across the range. Points are translated into the crop and the mask is restored to full image size.",
    "reset_prompts": "Resets Processing ROI to the input image bounds. Use when the crop cuts off the subject.",
    "fill_holes": "Fills enclosed holes in the cached SAM mask. Enable if solid parts of the subject have unwanted gaps; disable to preserve real holes. Updates cached-mask post-processing without rerunning SAM.",
    "max_hole_area": "Largest enclosed hole to fill, measured in pixels of area. Increase to close bigger gaps; decrease to preserve larger openings. 0 fills all enclosed hole sizes. Only used when Fill Enclosed Holes is enabled.",
    "confidence_width": "Range used to mark uncertain pixels gray in Edge Guide. Increase to let Refine reconsider more of the edge. This is a confidence range, not a pixel-distance radius; changing it does not rerun SAM.",
    "key_frame": "Frame where the primary object prompts anchor video tracking. Choose a clear view of the subject. Tracking propagates from this frame within the selected range.",
    "set_key_frame": "Makes the current timeline frame the primary tracking key frame. Set the object points here before propagating the mask through the shot.",
    "add_correction": "Stores a correction at the current frame using its object prompts. Use on a frame where tracking drifts, then propagate again to incorporate the correction.",
    "polish_key_frame": "Polishes the tracking key frame to reduce a visible transition between it and propagated frames. Try this if the mask changes abruptly at the key frame.",
    "propagate_forward": "Tracks the selected object forward from the key frame toward Range Last, using video tracking and any stored corrections.",
    "propagate_backward": "Tracks the selected object backward from the key frame toward Range First, using video tracking and any stored corrections.",
    "propagate_both": "Tracks from the key frame in both directions within the frame range. Use for a complete shot when the best reference frame is in the middle.",
    "output_mode": "Source + Alpha keeps source RGB and puts the mask in alpha. Alpha shows the mask in RGB and alpha. Cutout multiplies RGB by the mask. Edge Guide shows white object, black background and gray uncertain pixels for Refine. Bypass passes the source through.",
}

REFINE = {
    "model": "Selects the matting model used to recover fine edges from Source and Mask. Use Refine for hair, soft boundaries and partial transparency; the input mask must already identify the intended subject.",
    "tile_size": "Size of image tiles used for matting. 0 lets the model choose automatically. Smaller tiles can reduce peak memory but give less context; larger tiles need more memory. Tile behavior depends on the selected model.",
    "tile_overlap": "Pixels shared by neighboring tiles. Increase if tile boundaries are visible; more overlap repeats work and can take longer. Used when processing is tiled.",
    "generate_trimap": "Uses the simple Cattery ViTMatte recipe: average the eroded and dilated original Mask alpha, then make values below 0.01 black, above 0.99 white, and the rest gray. Disable to apply the same cutoffs to an artist trimap without erosion/dilation.",
    "foreground_radius": "Erodes the definite-foreground region by this many pixels when generating a trimap. Increase to let the model reconsider more pixels inside the mask edge; too large can remove useful foreground guidance.",
    "background_radius": "Expands the uncertain region outside the input mask by this many pixels. Increase to let the model recover hair or detail beyond the mask; too large can include unwanted background.",
    "output_mode": "Chooses how to display the refined matte or its trimap preview. Source + Refined Alpha keeps source RGB with the refined alpha; Refined Matte shows the matte; Refined Cutout premultiplies the subject. Trimap modes help inspect the uncertain edge before inference.",
}

INPAINT = {
    "model": "Selects the LaMa object-removal model. Connect Source and a mask with white over the area to replace. The Fast model uses a fixed 512 x 512 input; a tight crop helps preserve useful detail.",
    "in_colorspace": "Colorspace of the source image entering the inpaint conversion. Match the actual incoming image so model input and returned RGB are converted correctly; a mismatch can shift brightness or color.",
    "preprocess_mask": "Applies Invert, Threshold and Model Mask Grow before sending the mask to LaMa. Disable to use the input mask without these preprocessing adjustments. Use Preview Model Mask to inspect what is sent.",
    "invert_mask": "Swaps wanted and unwanted areas when mask preprocessing is enabled. LaMa replaces white areas and preserves black areas. Enable if your input mask marks the area to keep instead.",
    "mask_threshold": "Cutoff used to turn the preprocessed mask into the model's removal area. Raising it excludes weaker gray mask pixels; lowering it includes more of the soft edge. Check Preview Model Mask while adjusting.",
    "mask_grow": "Expands the model's removal mask in source pixels; negative values shrink it. Increase if traces of the object remain at the edge. Too much growth removes extra surrounding detail. Requires mask preprocessing.",
    "preview_model_mask": "Shows the exact effective mask sent to LaMa. White will be replaced. Use this to check inversion, threshold and growth before processing.",
    "edge_softness": "Softens the returned RGB blend inward from the removal boundary. Increase to hide a hard seam; too much can leave original object color near the edge. Does not change the model mask or output mask alpha.",
    "edge_color_match": "Adjusts the generated patch toward surrounding edge colors. Increase if the patch has a visible color mismatch; keep low if matching causes an unwanted tint.",
    "quality_mode": "Standard runs normal inpainting. Refined enables optional Big-LaMa feature refinement when supported by the model; it takes additional processing time. Try it when the standard patch needs improvement.",
    "refinement_steps": "Number of optimization steps in Refined quality mode. More steps take longer and may improve the patch, but improvement is not guaranteed. Start low while testing.",
    "refinement_scales": "Number of image scales used by optional feature refinement. More scales add processing work and context at different sizes. Applies only to supported refinement models in Refined mode.",
    "refinement_strength": "Strength of optional feature refinement. Increase for a stronger optimization update; reduce if refinement becomes unstable or damages a good initial result. Used in Refined mode with a supported model.",
    "output_mode": "Result shows the completed RGB image. Result + Mask Alpha adds the effective model mask as alpha; Result Premult multiplies RGB by that mask. Generated Patch isolates the generated replacement for compositing with Source.",
}

DLSS = {
    "model": "Kyven's local DLSS neural reconstruction runtime. Use Model Manager to install and check the required worker and runtime components.",
    "processing_scale": "Scales source width and height for neural processing. 1 uses full size; 0.5 uses half width and half height, reducing work and memory but also fine detail. Output stays at Source resolution; processing is limited to 8192 pixels on the longest side.",
    "hdr_rolloff": "Compresses highlights and handles negative RGB for neural inference, then restores the neural change to the original scene-linear range. Keep enabled for HDR sources; original alpha is preserved.",
    "preserve_source_color": "Keeps Source RGB ratios while applying the reconstructed brightness and detail. Enabled by default to prevent neural color casts. Disable to retain the model's full color and relighting changes.",
    "rolloff_knee": "Sets where highlight compression starts when HDR Rolloff is enabled. Lower values compress a wider highlight range; higher values leave more of the lower range untouched.",
    "intensity": "Overall strength of neural reconstruction. Increase for a stronger change; reduce if the image looks overprocessed. Start at 1 and compare with Source output.",
    "local_tone": "Neural local-tone control. Higher values request stronger local tonal treatment; lower values make it subtler. Compare on your image because the response depends on the runtime and content.",
    "local_structure": "Neural structure/detail control. Increase to request stronger structural treatment; decrease if fine textures look artificial or unstable.",
    "skin_structure": "Separate neural structure setting for skin. -1 inherits Local Structure. Use a lower explicit value if skin looks too strongly processed; the effect depends on runtime detection.",
    "style": "Selects the runtime's reconstruction style: Default, Natural or Cinematic. Compare on a representative frame; these are model styles, not a guaranteed increase in quality.",
    "preset": "Selects a runtime-defined reconstruction preset. Start with Default and compare alternatives on the same frame; preset numbers do not form a quality scale.",
    "mask_source": "Full Frame allows changes everywhere. Mask Input uses the connected mask; Source Alpha uses source alpha as a neural guide. White allows changes, black protects and gray gives partial guidance. Output alpha stays unchanged.",
    "protect_outside": "Also protects full-resolution source RGB outside the guide when the neural correction is restored. Enable to keep the unmasked background unchanged. Used with Mask Input or Source Alpha.",
    "temporal": "Uses previous correction frames aligned with external motion vectors to reduce flicker. Requires current-to-previous vectors on Motion input. Incorrect vectors can cause trails; disable to compare a single-frame result.",
    "history_frames": "Number of previous correction frames to retain, excluding the current frame. More history can improve stability but uses more CPU memory and depends on accurate motion. The memory budget may reduce the actual count.",
    "history_budget_mb": "CPU memory limit for temporal history in MiB. Increase to retain more history at large image sizes; lower to save RAM. The effective history count can be smaller than History Buffer.",
    "history_status": "Shows information about the temporal history actually used. Check this if the requested History Buffer is reduced by memory limits.",
    "temporal_strength": "How strongly accepted previous corrections influence the current correction. Increase to reduce flicker; decrease if moving edges smear or lag. Requires temporal processing and valid external vectors.",
    "history_clamp": "Limits old corrections relative to the current correction. Lower positive values constrain history more strongly and can reduce trails; higher values allow more variation. 0 disables clamping.",
    "rejection": "Image-difference tolerance for accepting motion-aligned history. Lower values reject mismatched history more readily; higher values retain more history but can increase ghosting.",
    "scene_cut": "Image-change threshold used to reset temporal history on a cut. Lower values reset more readily; higher values tolerate larger changes. Reduce if old-shot corrections leak into a new shot.",
    "depth_rejection": "Depth-difference tolerance for rejecting temporal history. Lower values reject more aggressively at depth changes; higher values keep more history. Relevant when a valid Depth input is connected.",
    "motion_x": "Horizontal channel on the Motion input containing current-to-previous displacement. Enter the actual channel name from your vector generator; channel selection must match its output.",
    "motion_y": "Vertical channel on the Motion input containing current-to-previous displacement. Pair it with Motion X from the same vector field.",
    "motion_units": "Pixels means displacement is measured in source pixels. UV means displacement is normalized to image dimensions. Match the vector generator or history alignment will be wrong.",
    "motion_y_up": "Interprets positive vertical motion as pointing upward, as in Nuke. Match your vector generator's coordinate convention. Reversing the sign does not turn forward flow into backward flow.",
    "motion_scale_x": "Multiplier for horizontal vector displacement. 1 uses the stored magnitude; negative values flip the horizontal sign. Adjust only to match the vector encoding, not to hide inaccurate motion.",
    "motion_scale_y": "Multiplier for vertical vector displacement. 1 uses the stored magnitude; negative values flip the vertical sign. Match the vector generator's encoding.",
    "use_nuke_vectors": "Sets backward vector channel names for Nuke-style motion. Check that these channels exist on Motion input and describe current-to-previous motion.",
    "motion_consistency": "Validates backward motion against forward motion to reject unreliable history. Requires both forward channels on the same Motion input. Useful around occlusions and moving edges.",
    "forward_x": "Horizontal current-to-next vector channel for forward/backward validation. Used only when Check Forward / Backward is enabled.",
    "forward_y": "Vertical current-to-next vector channel for forward/backward validation. Use the matching partner of Forward X.",
    "consistency_threshold": "Allowed forward/backward vector disagreement in source pixels. Lower values reject more uncertain motion; higher values accept more history but may allow trails.",
    "depth_channel": "Channel on the optional Depth input used to reject history across depth changes. Match your depth pass; use raw/data values rather than a display-transformed image.",
    "reset_history": "Clears temporal history so the next frame starts without previous corrections. Use after a discontinuity or when testing temporal artifacts.",
    "live_delay": "Delay in seconds before a live update is submitted after changes. Increase to reduce repeated work while scrubbing; decrease for a quicker response.",
    "output_mode": "Result shows reconstructed RGB while preserving source alpha and auxiliary channels. Source shows the original image for comparison.",
}

BY_KIND = {"segment": SEGMENT, "refine": REFINE, "inpaint": INPAINT, "dlss": DLSS}
COMMON["first_frame"] = COMMON["range_first"]
COMMON["last_frame"] = COMMON["range_last"]
COMMON["dlss_name"] = COMMON["roto_name"]


def node_kind(node: Any) -> str:
    knobs = node.knobs()
    if "kyven_kind" in knobs:
        value = str(node["kyven_kind"].value())
        if value in BY_KIND:
            return value
    if "generate_trimap" in knobs:
        return "refine"
    if "mask_grow" in knobs:
        return "inpaint"
    if "motion_x" in knobs:
        return "dlss"
    return "segment"


def tooltip_for(name: str, kind: str) -> str:
    text = BY_KIND[kind].get(name, COMMON.get(name, ""))
    if text:
        return text
    if re.fullmatch(r"positive_point(?:_\d+)?", name):
        return "Foreground prompt in source-image X/Y coordinates. Place inside the object you want to keep. Move it onto a missed part or add another positive point if the mask omits that part."
    if re.fullmatch(r"negative_point(?:_\d+)?", name):
        return "Exclusion prompt in source-image X/Y coordinates. Place inside unwanted background or an extra object to remove it from the selection. Avoid placing it on the wanted subject."
    if re.fullmatch(r"remove_(positive|negative)_point_\d+", name):
        return "Removes this prompt point. Remove a misleading point if it makes the selected object less accurate."
    if re.fullmatch(r"correction_frame_\d+", name):
        return "Frame containing a saved tracking correction. Change the frame to move this correction to another point in the shot; propagate again to update tracking."
    if re.fullmatch(r"remove_correction_\d+", name):
        return "Removes this saved tracking correction. Subsequent tracking will no longer use its object prompts."
    return ""


def apply_tooltips(node: Any, kind: str | None = None) -> None:
    """Refresh public help on creation, upgrade and script-load repair."""
    kind = kind or node_kind(node)
    for name, knob in node.knobs().items():
        text = tooltip_for(name, kind)
        if text:
            knob.setTooltip(text)
