# ComfyUI-AlphaKey
Flat-background keying for video: `Alpha Key (flat background)` keys the background (median of each frame's border) and unmixes its colour from edge pixels (no coloured fringe); `Alpha Key Load Video` loads a clip at native resolution (pixel-art safe). Torch only, no compiled deps.
Key core is `screen_matte_vnccs.py`, copied unmodified from AHEKOT/ComfyUI_VNCCS (MIT, see LICENSE-VNCCS.txt).
