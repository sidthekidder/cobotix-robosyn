#!/usr/bin/env python3
"""Render ClickBell keypoint sidecars over their recorded camera videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont


CAMERAS = ("cam_high", "cam_right_wrist", "cam_left_wrist")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="One LeRobot dataset directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument(
        "--sheet-frames",
        default="0,9,19,29,39,49,59,69,73",
        help="Comma-separated frame indices for each contact sheet",
    )
    return parser


def _run(*args: str) -> None:
    subprocess.run(args, check=True)


def _ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as error:
        raise RuntimeError(
            "ffmpeg is required; install the executable or imageio-ffmpeg"
        ) from error


def _video_path(dataset: Path, camera: str) -> Path:
    matches = list(dataset.glob(f"videos/chunk-*/observation.images.{camera}/*.mp4"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {camera} video, found {len(matches)}")
    return matches[0]


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _annotate(
    frame: Image.Image,
    *,
    camera: str,
    frame_index: int,
    keypoint: list[float],
    geometry: dict | None = None,
) -> Image.Image:
    image = frame.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    visible = keypoint[2] > 0.5
    header = f"{camera}  frame {frame_index:02d}  "
    if visible:
        x = keypoint[0] * (width - 1)
        y = keypoint[1] * (height - 1)
        radius = max(8, round(min(width, height) * 0.025))
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=(0, 255, 80),
            width=max(3, radius // 4),
        )
        draw.line((x - radius * 1.4, y, x + radius * 1.4, y), fill=(0, 255, 80), width=3)
        draw.line((x, y - radius * 1.4, x, y + radius * 1.4), fill=(0, 255, 80), width=3)
        status = f"VISIBLE  x={keypoint[0]:.3f} y={keypoint[1]:.3f}"
        color = (0, 255, 80)
    else:
        status = "NOT VISIBLE"
        color = (255, 80, 80)
    if geometry is not None:
        mask = geometry["mask"]
        bbox = mask["bbox_xyxy_pixels"]
        if bbox is not None:
            draw.rectangle(tuple(bbox), outline=(255, 170, 0), width=3)

        press = geometry["press_point"]
        tip = geometry["right_tool_tip"]
        press_xy = press["xy_pixels"] if press["in_front"] else None
        tip_xy = tip["xy_pixels"] if tip["in_front"] else None
        if press_xy is not None and tip_xy is not None:
            draw.line((*tip_xy, *press_xy), fill=(255, 255, 255), width=2)
        for point, point_color, point_radius in (
            (press_xy, (0, 220, 255), 9),
            (tip_xy, (255, 0, 255), 8),
        ):
            if point is not None:
                px, py = point
                draw.ellipse(
                    (px - point_radius, py - point_radius, px + point_radius, py + point_radius),
                    outline=point_color,
                    width=4,
                )
        status += (
            f"  area={mask['area_pixels']} boundary={int(mask['touches_image_boundary'])}"
            f" confidence={geometry['mask_centroid_confidence']:.0f}"
        )
    font = _font(max(18, round(height * 0.045)))
    text = header + status
    box = draw.textbbox((0, 0), text, font=font)
    pad = 8
    draw.rectangle((0, 0, box[2] + 2 * pad, box[3] + 2 * pad), fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=color, font=font)
    return image


def main() -> None:
    args = _parser().parse_args()
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(line)
        for line in (dataset / "bell_keypoints.jsonl").read_text().splitlines()
        if line.strip()
    ]
    sheet_indices = [int(value) for value in args.sheet_frames.split(",")]
    ffmpeg = _ffmpeg()

    with tempfile.TemporaryDirectory(prefix="click-bell-keypoints-") as temporary:
        temporary_root = Path(temporary)
        for camera_index, camera in enumerate(CAMERAS):
            raw_dir = temporary_root / camera / "raw"
            overlay_dir = temporary_root / camera / "overlay"
            raw_dir.mkdir(parents=True)
            overlay_dir.mkdir(parents=True)
            _run(
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(_video_path(dataset, camera)),
                str(raw_dir / "%06d.png"),
            )
            frame_paths = sorted(raw_dir.glob("*.png"))
            if len(frame_paths) != len(rows):
                raise RuntimeError(
                    f"{camera}: {len(frame_paths)} video frames != {len(rows)} labels"
                )
            selected: list[Image.Image] = []
            for frame_index, frame_path in enumerate(frame_paths):
                frame = Image.open(frame_path)
                annotated = _annotate(
                    frame,
                    camera=camera,
                    frame_index=frame_index,
                    keypoint=rows[frame_index]["keypoints"][camera_index],
                    geometry=rows[frame_index].get("geometry", {}).get(camera),
                )
                annotated.save(overlay_dir / f"{frame_index + 1:06d}.png")
                if frame_index in sheet_indices:
                    selected.append(annotated.copy())
            _run(
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-framerate",
                str(args.fps),
                "-i",
                str(overlay_dir / "%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-y",
                str(output / f"{camera}_keypoints.mp4"),
            )
            columns = 3
            cell_width, cell_height = selected[0].size
            rows_count = (len(selected) + columns - 1) // columns
            sheet = Image.new("RGB", (columns * cell_width, rows_count * cell_height), "black")
            for index, frame in enumerate(selected):
                sheet.paste(frame, ((index % columns) * cell_width, (index // columns) * cell_height))
            sheet.save(output / f"{camera}_contact_sheet.jpg", quality=92)


if __name__ == "__main__":
    main()
