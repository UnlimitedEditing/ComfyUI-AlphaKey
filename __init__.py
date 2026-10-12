"""ComfyUI-AlphaKey: flat-background keying for video frames (key + colour unmix), no heavy deps."""
import torch
from .screen_matte_vnccs import screen_matte


class AlphaKeyScreenMatte:
    """Key a flat background out of every frame and unmix its colour from the edges.

    The background colour is the median of each frame's border (robust to noise). Alpha comes from the
    colour distance, and edge pixels are solved as C = a*F + (1-a)*B so the background fringe is removed
    from the RGB itself. Wraps VNCCS screen_matte (MIT)."""

    @classmethod
    def INPUT_TYPES(cls):
        f = lambda d, lo=0.0, hi=1.0: ("FLOAT", {"default": d, "min": lo, "max": hi, "step": 0.01})
        return {"required": {
            "images": ("IMAGE",),
            "tolerance": f(0.15), "softness": f(0.12, 0.001),
            "despill_strength": f(0.65), "edge_width": ("INT", {"default": 3, "min": 0, "max": 32}),
            "matte_cleanup": f(0.1), "foreground_recover": f(0.35),
            "edge_decontaminate": f(0.75), "edge_choke": f(0.08),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("rgba", "rgb_unmixed", "alpha")
    FUNCTION = "key"
    CATEGORY = "AlphaKey"

    def key(self, images, tolerance, softness, despill_strength, edge_width,
            matte_cleanup, foreground_recover, edge_decontaminate, edge_choke):
        rgba, alpha = [], []
        for frame in images:
            r, a, _ = screen_matte(frame[..., :3], tolerance, softness, despill_strength, edge_width,
                                   matte_cleanup, foreground_recover, edge_decontaminate, edge_choke,
                                   "straight_rgba")
            rgba.append(r.cpu()); alpha.append(a.cpu())
        rgba = torch.stack(rgba)
        return (rgba, rgba[..., :3].contiguous(), torch.stack(alpha))


class AlphaKeyLoadVideo:
    """Load a video from a URL (or staged filename) at its NATIVE size and fps - no resize, so pixel art
    stays pixel-exact. fps=0 keeps the source rate; max_frames caps the clip (0 = all)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "url": ("STRING", {"default": ""}),
            "fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 120.0}),
            "max_frames": ("INT", {"default": 0, "min": 0, "max": 100000}),
        }, "optional": {"filename": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("IMAGE", "FLOAT", "INT")
    RETURN_NAMES = ("frames", "fps", "frame_count")
    FUNCTION = "load"
    CATEGORY = "AlphaKey"

    def load(self, url, fps=0.0, max_frames=0, filename=""):
        import json, os, shutil, subprocess, tempfile
        import numpy as np
        import requests
        import folder_paths
        src = (url or "").strip() or (filename or "").strip()
        if not src:
            raise RuntimeError("AlphaKeyLoadVideo: no url or filename given")
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "in.mp4")
        try:
            if src.startswith(("http://", "https://")):
                r = requests.get(src, timeout=120, stream=True)
                r.raise_for_status()
                with open(path, "wb") as fh:
                    for chunk in r.iter_content(65536):
                        fh.write(chunk)
            else:
                shutil.copyfile(folder_paths.get_annotated_filepath(src), path)
            p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams",
                                "-of", "json", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            st = (json.loads(p.stdout or b"{}").get("streams") or [{}])[0]
            w, h = int(st["width"]), int(st["height"])
            a, _, b = str(st.get("avg_frame_rate", "0/1")).partition("/")
            src_fps = float(a) / float(b or 1) if float(b or 1) else 0.0
            out_fps = float(fps) if fps and fps > 0 else (src_fps or 24.0)
            cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={out_fps}"]
            if max_frames and max_frames > 0:
                cmd += ["-frames:v", str(int(max_frames))]
            cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            raw = np.frombuffer(p.stdout, np.uint8)
            n = raw.size // (w * h * 3)
            if n == 0:
                raise RuntimeError("AlphaKeyLoadVideo: ffmpeg returned no frames: " + p.stderr[-300:].decode("utf-8", "ignore"))
            frames = torch.from_numpy(raw[: n * w * h * 3].reshape(n, h, w, 3).astype(np.float32) / 255.0)
            return (frames, out_fps, n)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


NODE_CLASS_MAPPINGS = {"AlphaKeyScreenMatte": AlphaKeyScreenMatte, "AlphaKeyLoadVideo": AlphaKeyLoadVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"AlphaKeyScreenMatte": "Alpha Key (flat background)", "AlphaKeyLoadVideo": "Alpha Key Load Video (native size)"}
