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


class AlphaKeyCleanup:
    """Hard-edge cleanup for keyed pixel art: threshold alpha, drop edge pixels that are still background
    coloured, remove stray specks, fill pinholes, then recolour every edge pixel from the nearest solid
    interior pixel so no background colour can survive on the outline. Background colour = median of the
    border over ALL frames (stable, so edges do not jitter with per-frame noise)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "rgba": ("IMAGE",), "source": ("IMAGE",),
            "alpha_threshold": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 0.95, "step": 0.01}),
            "bg_distance": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.6, "step": 0.01}),
            "edge_depth": ("INT", {"default": 2, "min": 0, "max": 8}),
            "min_island": ("INT", {"default": 6, "min": 0, "max": 5000}),
            "fill_holes": ("INT", {"default": 4, "min": 0, "max": 5000}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("rgba", "alpha")
    FUNCTION = "clean"
    CATEGORY = "AlphaKey"

    def clean(self, rgba, source, alpha_threshold, bg_distance, edge_depth, min_island, fill_holes):
        import numpy as np
        try:
            from scipy import ndimage as ndi
        except Exception:
            ndi = None
        src = source[..., :3].cpu().numpy()
        n, h, w, _ = src.shape
        border = np.concatenate([src[:, :2].reshape(-1, 3), src[:, -2:].reshape(-1, 3),
                                 src[:, :, :2].reshape(-1, 3), src[:, :, -2:].reshape(-1, 3)])
        bg = np.median(border, axis=0)
        st = np.ones((3, 3), bool)
        out_rgba, out_a = [], []
        for i in range(n):
            a = rgba[i, ..., 3].cpu().numpy()
            rgb = src[i]
            hard = a > alpha_threshold
            if ndi is not None:
                # 1) edge-band pixels that still look like the background are not subject
                core = ndi.binary_erosion(hard, st, iterations=2) if edge_depth else hard
                near_bg = np.sqrt(((rgb - bg) ** 2).sum(-1)) < bg_distance
                hard = hard & ~((hard & ~core) & near_bg)
                # 2) stray specks and pinholes
                if min_island > 0:
                    lab, k = ndi.label(hard, st)
                    if k:
                        size = ndi.sum(hard, lab, range(1, k + 1))
                        hard = np.isin(lab, np.nonzero(size >= min_island)[0] + 1)
                if fill_holes > 0:
                    lab, k = ndi.label(~hard, st)
                    if k:
                        size = ndi.sum(~hard, lab, range(1, k + 1))
                        edge_lab = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
                        small = [j + 1 for j in range(k) if size[j] < fill_holes and (j + 1) not in edge_lab]
                        hard = hard | np.isin(lab, small)
                # 3) recolour edge pixels from the nearest trusted interior pixel
                trusted = ndi.binary_erosion(hard, st, iterations=edge_depth) if edge_depth else hard
                if trusted.any() and edge_depth:
                    _, (iy, ix) = ndi.distance_transform_edt(~trusted, return_indices=True)
                    rgb = np.where(trusted[..., None], rgb, rgb[iy, ix])
            alpha = hard.astype(np.float32)
            out = np.concatenate([np.where(hard[..., None], rgb, 0.0), alpha[..., None]], -1)
            out_rgba.append(torch.from_numpy(out.astype(np.float32))); out_a.append(torch.from_numpy(alpha))
        return (torch.stack(out_rgba), torch.stack(out_a))


NODE_CLASS_MAPPINGS = {"AlphaKeyCleanup": AlphaKeyCleanup, "AlphaKeyScreenMatte": AlphaKeyScreenMatte, "AlphaKeyLoadVideo": AlphaKeyLoadVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"AlphaKeyCleanup": "Alpha Key Cleanup (hard edges)", "AlphaKeyScreenMatte": "Alpha Key (flat background)", "AlphaKeyLoadVideo": "Alpha Key Load Video (native size)"}
