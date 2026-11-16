#!/usr/bin/env python3
"""Render a short, silent 16:9 explainer from native drawings and ORL samples."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import math
import shutil
import subprocess
import zipfile

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / ".cache/orl-original/att_faces.zip"
OUTPUT = ROOT / "output/video/Group27_MDA_Process_Explainer.mp4"
WIDTH, HEIGHT, FPS = 1920, 1080, 24
BG = (16, 22, 27)
PANEL = (27, 37, 44)
PANEL2 = (34, 45, 52)
INK = (242, 240, 232)
MUTED = (156, 169, 174)
GRID = (39, 51, 58)
TEAL = (54, 166, 151)
RUST = (221, 101, 67)
GOLD = (203, 148, 61)
BLUE = (98, 146, 173)
PURPLE = (151, 129, 190)

AVENIR = "/System/Library/Fonts/Avenir Next.ttc"
BASKERVILLE = "/System/Library/Fonts/Supplemental/Baskerville.ttc"


def load_font(path: str, size: int, index: int = 0) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size, index=index)


F_BODY = load_font(AVENIR, 29, 7)
F_BODY_MED = load_font(AVENIR, 29, 5)
F_BODY_BOLD = load_font(AVENIR, 29, 0)
F_SMALL = load_font(AVENIR, 20, 7)
F_LABEL = load_font(AVENIR, 22, 0)
F_STAT = load_font(BASKERVILLE, 66, 0)
F_TITLE = load_font(AVENIR, 90, 0)
F_TITLE_SMALL = load_font(AVENIR, 72, 0)
F_SUBTITLE = load_font(AVENIR, 36, 7)
F_NUMBER = load_font(BASKERVILLE, 48, 0)


def ease(value: float) -> float:
    x = max(0.0, min(1.0, value))
    return 1.0 - (1.0 - x) ** 3


def mix(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def wrap_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        line = ""
        for word in words:
            candidate = word if not line else f"{line} {word}"
            if draw.textlength(candidate, font=font) <= width or not line:
                line = candidate
            else:
                lines.append(line)
                line = word
        lines.append(line)
    return lines


def paragraph(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str,
              font: ImageFont.FreeTypeFont, fill: tuple[int, int, int],
              width: int, spacing: int = 10) -> int:
    x, y = xy
    lines = wrap_lines(draw, text, font, width)
    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
    draw.multiline_text((x, y), "\n".join(lines), font=font, fill=fill, spacing=spacing)
    return y + len(lines) * line_height + (len(lines) - 1) * spacing


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str,
          fill: tuple[int, int, int] = TEAL) -> None:
    draw.text(xy, text.upper(), font=F_LABEL, fill=fill)


def round_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
              fill=PANEL, outline=None, radius: int = 18, width: int = 2) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int],
               color=MUTED, width: int = 4, head: int = 15) -> None:
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    left = (end[0] - head * math.cos(angle - math.pi / 6), end[1] - head * math.sin(angle - math.pi / 6))
    right = (end[0] - head * math.cos(angle + math.pi / 6), end[1] - head * math.sin(angle + math.pi / 6))
    draw.polygon((end, left, right), fill=color)


def load_orl_faces() -> dict[str, Image.Image]:
    if not ARCHIVE.exists():
        raise FileNotFoundError(f"Expected original ORL archive at {ARCHIVE}")
    requested = ["s1/1.pgm", "s1/2.pgm", "s1/3.pgm", "s1/5.pgm", "s1/7.pgm", "s1/10.pgm", "s2/1.pgm", "s3/1.pgm"]
    faces: dict[str, Image.Image] = {}
    with zipfile.ZipFile(ARCHIVE) as zf:
        for name in requested:
            raw = zf.read(name)
            with Image.open(BytesIO(raw)) as source:
                faces[name] = ImageOps.autocontrast(source.convert("L")).convert("RGB")
    return faces


FACES = load_orl_faces()


def photo(draw: ImageDraw.ImageDraw, canvas: Image.Image, face_key: str,
          box: tuple[int, int, int, int], caption: str = "") -> None:
    x0, y0, x1, y1 = box
    face = FACES[face_key].resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    mask = Image.new("L", face.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, face.width - 1, face.height - 1), radius=14, fill=255)
    canvas.paste(face, (x0, y0), mask)
    draw.rounded_rectangle(box, radius=14, outline=(106, 121, 128), width=2)
    if caption:
        draw.text((x0, y1 + 13), caption, font=F_SMALL, fill=MUTED)


SCENES = [
    (3.7, "FROM PIXELS TO IDENTITY"),
    (4.0, "THE GALLERY AND THE PROBE"),
    (4.7, "KEEP THE IMAGE AXES"),
    (4.0, "LEARN A DISCRIMINATIVE SPACE"),
    (4.0, "CLASSIFY BY NEAREST NEIGHBOUR"),
    (4.8, "THE FIXED-SETTING RESULT"),
    (4.3, "A CAREFUL REPRODUCTION CLAIM"),
]
TOTAL_SECONDS = sum(duration for duration, _ in SCENES)


def background() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    # A restrained reference grid keeps the diagrams grounded without acting as decoration.
    for x in range(120, WIDTH, 240):
        draw.line((x, 110, x, 960), fill=GRID, width=1)
    draw.line((100, 965, WIDTH - 100, 965), fill=GRID, width=1)
    return image


def chrome(draw: ImageDraw.ImageDraw, scene_index: int, local_t: float) -> None:
    draw.text((112, 54), "COMP3314   /   GROUP 27", font=F_SMALL, fill=MUTED)
    draw.text((WIDTH - 260, 54), "MDA   ·   ORL", font=F_SMALL, fill=MUTED)
    draw.line((110, 100, WIDTH - 110, 100), fill=GRID, width=2)
    # Seven-step timeline; the active segment follows the current scene.
    x0, y, gap, seg_w = 110, 1010, 12, 220
    for i in range(len(SCENES)):
        x = x0 + i * (seg_w + gap)
        fill = TEAL if i == scene_index else (70, 83, 90)
        draw.rounded_rectangle((x, y, x + seg_w, y + 5), radius=2, fill=fill)
    step = f"{scene_index + 1:02d} / {len(SCENES):02d}"
    draw.text((WIDTH - 180, 992), step, font=F_SMALL, fill=MUTED)


def draw_scene_1(canvas: Image.Image, draw: ImageDraw.ImageDraw, t: float) -> None:
    label(draw, (150, 220), "A face-recognition study", TEAL)
    y = 300 + int(20 * (1 - ease(t / 0.55)))
    draw.multiline_text((145, y), "From pixels\nto identity", font=F_TITLE, fill=INK, spacing=-3)
    paragraph(draw, (150, 550), "MDA keeps a face as an image matrix, then learns a compact identity feature.", F_SUBTITLE, MUTED, 650, 12)
    names = ["s1/1.pgm", "s1/3.pgm", "s1/5.pgm", "s1/10.pgm"]
    start_x, top_y, tile_w, tile_h, gap = 920, 338, 168, 205, 24
    for i, name in enumerate(names):
        reveal = ease((t - 0.22 * i) / 0.55)
        yoff = int(30 * (1 - reveal))
        x = start_x + i * (tile_w + gap)
        photo(draw, canvas, name, (x, top_y + yoff, x + tile_w, top_y + yoff + tile_h))
    draw.text((930, 600), "One ORL identity · different captures", font=F_SMALL, fill=MUTED)
    draw.text((150, 820), "Original ORL faces   /   400 images   /   40 identities", font=F_BODY_BOLD, fill=INK)


def draw_scene_2(canvas: Image.Image, draw: ImageDraw.ImageDraw, t: float) -> None:
    label(draw, (150, 180), "The experiment", TEAL)
    draw.text((145, 238), "Known examples in. Held-out faces out.", font=F_TITLE_SMALL, fill=INK)
    draw.text((150, 333), "Each identity is split into gallery images and probe images.", font=F_SUBTITLE, fill=MUTED)

    gallery_box = (160, 445, 930, 760)
    probe_box = (1120, 445, 1740, 760)
    round_box(draw, gallery_box, PANEL)
    round_box(draw, probe_box, PANEL)
    label(draw, (195, 468), "Gallery  /  fit U and V", TEAL)
    label(draw, (1155, 468), "Probe  /  test identity", RUST)
    for i, key in enumerate(["s1/1.pgm", "s1/2.pgm", "s1/3.pgm"]):
        a = ease((t - 0.15 * i) / 0.5)
        x = 215 + i * 220
        photo(draw, canvas, key, (x, 535 + int(18 * (1 - a)), x + 160, 695 + int(18 * (1 - a))))
    photo(draw, canvas, "s1/10.pgm", (1320, 535, 1500, 755))
    draw_arrow(draw, (960, 605), (1080, 605), MUTED, 4, 18)
    draw.text((975, 560), "fit only on gallery", font=F_SMALL, fill=MUTED)

    stats = [("400", "grayscale images"), ("40", "identities"), ("10", "images each"), ("112 × 92", "rows × columns")]
    for i, (number, caption) in enumerate(stats):
        x = 175 + i * 420
        draw.text((x, 840), number, font=F_STAT, fill=TEAL if i < 3 else RUST)
        draw.text((x, 914), caption, font=F_SMALL, fill=MUTED)


def draw_scene_3(canvas: Image.Image, draw: ImageDraw.ImageDraw, t: float) -> None:
    label(draw, (150, 180), "The MDA representation", TEAL)
    draw.text((145, 238), "Keep the image axes.", font=F_TITLE_SMALL, fill=INK)
    draw.text((150, 327), "A 112 × 92 face becomes a compact 6 × 6 feature in the fixed study.", font=F_SUBTITLE, fill=MUTED)

    # Input face matrix X with a sparse grid overlay.
    face_x, face_y = 185, 445
    photo(draw, canvas, "s1/7.pgm", (face_x, face_y, face_x + 208, face_y + 252))
    for k in range(1, 5):
        x = face_x + k * 208 // 5
        y = face_y + k * 252 // 5
        draw.line((x, face_y + 2, x, face_y + 250), fill=(95, 117, 123), width=1)
        draw.line((face_x + 2, y, face_x + 206, y), fill=(95, 117, 123), width=1)
    draw.text((195, 718), "X  /  112 rows × 92 columns", font=F_SMALL, fill=INK)

    draw_arrow(draw, (440, 565), (610, 565), TEAL, 4, 18)
    draw.text((472, 520), "U transpose", font=F_BODY_BOLD, fill=TEAL)
    draw.text((460, 620), "row projection", font=F_SMALL, fill=MUTED)

    # Intermediate mode-reduced matrix (schematic cells, not data values).
    grid_x, grid_y = 650, 490
    cell_w, cell_h, gap = 26, 24, 5
    for row in range(6):
        for col in range(7):
            shade = PANEL2 if (row + 2 * col) % 4 else (61, 105, 107)
            draw.rounded_rectangle((grid_x + col * (cell_w + gap), grid_y + row * (cell_h + gap),
                                    grid_x + col * (cell_w + gap) + cell_w, grid_y + row * (cell_h + gap) + cell_h),
                                   radius=3, fill=shade)
    draw.text((650, 688), "row mode reduced", font=F_SMALL, fill=MUTED)

    draw_arrow(draw, (900, 565), (1070, 565), RUST, 4, 18)
    draw.text((963, 512), "V", font=F_TITLE_SMALL, fill=RUST)
    draw.text((918, 620), "column projection", font=F_SMALL, fill=MUTED)

    out_x, out_y, out_cell, out_gap = 1135, 465, 31, 8
    for row in range(6):
        for col in range(6):
            color = TEAL if (row * 2 + col) % 5 == 0 else PANEL2
            draw.rounded_rectangle((out_x + col * (out_cell + out_gap), out_y + row * (out_cell + out_gap),
                                    out_x + col * (out_cell + out_gap) + out_cell, out_y + row * (out_cell + out_gap) + out_cell),
                                   radius=4, fill=color)
    draw.text((1135, 720), "Z = transpose(U) × X × V", font=F_BODY_BOLD, fill=INK)
    draw.text((1135, 760), "6 × 6 output descriptor", font=F_SMALL, fill=MUTED)
    draw.text((1490, 510), "36 learned\nfeature values", font=F_SUBTITLE, fill=TEAL, spacing=6)

    round_box(draw, (165, 825, 1745, 915), PANEL)
    draw.text((200, 850), "MDA/2-1 learns one mode.", font=F_BODY_BOLD, fill=INK)
    draw.text((760, 850), "MDA/2-2 alternates row U and column V updates.", font=F_BODY, fill=MUTED)


def draw_scene_4(canvas: Image.Image, draw: ImageDraw.ImageDraw, t: float) -> None:
    label(draw, (150, 180), "What the objective rewards", TEAL)
    draw.text((145, 238), "Separate identities; compact each identity.", font=F_TITLE_SMALL, fill=INK)
    draw.text((150, 328), "The discriminant tensor criterion balances between-class and within-class scatter.", font=F_SUBTITLE, fill=MUTED)

    x0, y0, x1, y1 = 220, 820, 1690, 420
    draw.line((x0, y0, x1, y0), fill=(92, 108, 115), width=2)
    draw.line((x0, y0, x0, y1), fill=(92, 108, 115), width=2)
    draw.text((x0 + 20, y0 + 25), "feature direction 1", font=F_SMALL, fill=MUTED)
    draw.text((x0 - 92, y1 - 8), "feature direction 2", font=F_SMALL, fill=MUTED)

    start_a = [(420, 600), (670, 470), (770, 760), (990, 540), (1140, 690), (550, 790)]
    start_b = [(730, 700), (900, 430), (1080, 500), (1280, 780), (1450, 530), (1170, 430)]
    target_a = [(530, 590), (590, 535), (615, 625), (560, 650), (505, 565), (630, 570)]
    target_b = [(1320, 590), (1375, 535), (1285, 630), (1340, 650), (1400, 585), (1300, 530)]
    progress = ease((t - 0.25) / 2.6)
    for starts, targets, color, center in [(start_a, target_a, TEAL, (565, 590)), (start_b, target_b, RUST, (1340, 590))]:
        positions = [(mix(s[0], q[0], progress), mix(s[1], q[1], progress)) for s, q in zip(starts, targets)]
        for x, y in positions:
            draw.line((x, y, center[0], center[1]), fill=(50, 71, 75), width=1)
            draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=color)
        draw.ellipse((center[0] - 14, center[1] - 14, center[0] + 14, center[1] + 14), outline=INK, width=3)
    label(draw, (420, 730), "same identity → stay compact", TEAL)
    label(draw, (1190, 730), "different identities → move apart", RUST)
    draw.text((150, 900), "Conceptual feature-space sketch · points are illustrative, not experimental measurements", font=F_SMALL, fill=MUTED)


def draw_scene_5(canvas: Image.Image, draw: ImageDraw.ImageDraw, t: float) -> None:
    label(draw, (150, 180), "Recognition step", TEAL)
    draw.text((145, 238), "Each probe asks: which gallery feature is closest?", font=F_TITLE_SMALL, fill=INK)
    draw.text((150, 330), "Project the held-out face, then compare it with the labelled gallery.", font=F_SUBTITLE, fill=MUTED)

    x0, y0, x1, y1 = 270, 825, 1510, 420
    draw.line((x0, y0, x1, y0), fill=(91, 109, 116), width=2)
    draw.line((x0, y0, x0, y1), fill=(91, 109, 116), width=2)
    draw.text((x1 - 240, y0 + 28), "projected feature space", font=F_SMALL, fill=MUTED)
    gallery = [((715, 570), "A", TEAL), ((1240, 500), "B", RUST), ((1280, 720), "C", GOLD)]
    q_start, q_end = (450, 470), (830, 545)
    p = ease((t - 0.35) / 1.8)
    query = (mix(q_start[0], q_end[0], p), mix(q_start[1], q_end[1], p))
    # Dashed guides to non-nearest features and a solid segment to the closest one.
    for (gx, gy), name, color in gallery:
        if name == "A":
            draw.line((query, (gx, gy)), fill=TEAL, width=4)
        else:
            dx, dy = gx - query[0], gy - query[1]
            length = math.hypot(dx, dy)
            steps = 16
            for k in range(0, steps, 2):
                a = k / steps
                b = min((k + 1) / steps, 1)
                draw.line(((query[0] + a * dx, query[1] + a * dy), (query[0] + b * dx, query[1] + b * dy)), fill=(67, 81, 88), width=2)
        draw.ellipse((gx - 17, gy - 17, gx + 17, gy + 17), fill=color)
        draw.text((gx + 25, gy - 22), f"gallery {name}", font=F_SMALL, fill=INK)
    draw.ellipse((query[0] - 16, query[1] - 16, query[0] + 16, query[1] + 16), fill=INK, outline=RUST, width=5)
    draw.text((query[0] - 28, query[1] - 55), "probe", font=F_SMALL, fill=RUST)
    if t > 2.0:
        draw.ellipse((gallery[0][0][0] - 30, gallery[0][0][1] - 30, gallery[0][0][0] + 30, gallery[0][0][1] + 30), outline=INK, width=3)

    round_box(draw, (1550, 455, 1780, 760), PANEL)
    label(draw, (1582, 490), "1-NN", TEAL)
    paragraph(draw, (1582, 545), "Choose the nearest gallery face; use its identity label.", F_BODY, INK, 165, 10)
    draw.text((300, 900), "Euclidean distance in the projected feature space · diagram is illustrative", font=F_SMALL, fill=MUTED)


def draw_scene_6(canvas: Image.Image, draw: ImageDraw.ImageDraw, t: float) -> None:
    label(draw, (150, 155), "Group 27 · fixed-setting study", TEAL)
    draw.text((145, 212), "MDA/2-1 leads; MDA/2-2 is scale-sensitive.", font=F_TITLE_SMALL, fill=INK)
    draw.text((150, 300), "Mean probe accuracy (%) · 30 matched splits · fixed dimensions", font=F_SUBTITLE, fill=MUTED)

    left, top, plot_w, plot_h = 200, 810, 1040, 440
    for value in range(0, 101, 20):
        y = top - value / 100 * plot_h
        draw.line((left, y, left + plot_w, y), fill=GRID, width=2)
        draw.text((left - 72, y - 17), f"{value}%", font=F_SMALL, fill=MUTED)
    for i, gallery_size in enumerate([2, 3, 4, 5]):
        x = left + i * plot_w / 3
        draw.line((x, top - plot_h, x, top), fill=(32, 44, 50), width=1)
        draw.text((x - 8, top + 18), str(gallery_size), font=F_BODY_BOLD, fill=MUTED)
    draw.text((510, 872), "gallery images per identity", font=F_SMALL, fill=MUTED)
    series = [
        ("MDA/2-1", [81.4, 89.9, 92.0, 94.7], TEAL),
        ("MDA/2-2 unit-L2", [34.8, 52.6, 75.8, 87.5], RUST),
        ("MDA/2-2 within-scatter", [5.4, 23.9, 64.0, 81.4], GOLD),
    ]
    reveal = ease((t - 0.2) / 2.6)
    for name, values, color in series:
        coords = [(left + i * plot_w / 3, top - value / 100 * plot_h) for i, value in enumerate(values)]
        visible_segments = reveal * (len(coords) - 1)
        for i in range(len(coords) - 1):
            frac = max(0, min(1, visible_segments - i))
            if frac <= 0:
                continue
            a, b = coords[i], coords[i + 1]
            end = (mix(a[0], b[0], frac), mix(a[1], b[1], frac))
            draw.line((a, end), fill=color, width=5)
        for i, (x, y) in enumerate(coords):
            if i <= visible_segments + 0.02:
                draw.ellipse((x - 9, y - 9, x + 9, y + 9), fill=color)

    round_box(draw, (1325, 385, 1770, 770), PANEL)
    label(draw, (1365, 420), "At two gallery images", RUST)
    draw.text((1365, 485), "34.8%", font=F_STAT, fill=RUST)
    draw.text((1365, 565), "unit-L2 axes", font=F_SMALL, fill=MUTED)
    draw.text((1365, 620), "5.4%", font=F_STAT, fill=GOLD)
    draw.text((1365, 700), "within-scatter axes", font=F_SMALL, fill=MUTED)
    paragraph(draw, (1325, 805), "Same 1-NN classifier; axis scaling changes distances.", F_SMALL, INK, 420, 4)

    legends = [(TEAL, "MDA/2-1"), (RUST, "MDA/2-2 unit-L2"), (GOLD, "MDA/2-2 within-scatter")]
    legend_x = [200, 485, 815]
    for (color, name), x in zip(legends, legend_x):
        draw.line((x, 932, x + 34, 932), fill=color, width=4)
        draw.text((x + 46, 918), name, font=F_SMALL, fill=MUTED)


def draw_scene_7(canvas: Image.Image, draw: ImageDraw.ImageDraw, t: float) -> None:
    label(draw, (150, 185), "What the evidence supports", TEAL)
    draw.text((145, 242), "A careful result is still a useful result.", font=F_TITLE_SMALL, fill=INK)

    round_box(draw, (150, 390, 860, 780), PANEL)
    label(draw, (195, 430), "Our fixed comparison", TEAL)
    paragraph(draw, (195, 490), "MDA/2-1 led at every tested gallery size. MDA/2-2 changed sharply when its projection axes were scaled differently.", F_SUBTITLE, INK, 610, 15)

    round_box(draw, (920, 390, 1770, 780), PANEL2)
    label(draw, (965, 430), "Paper-proximate, not exact", RUST)
    paragraph(draw, (965, 490), "The paper leaves several run settings open. Our full-grid dimension choices also use probe scores, so their best values are optimistic.", F_SUBTITLE, INK, 740, 15)

    draw.line((150, 850, 1770, 850), fill=GRID, width=2)
    draw.text((150, 885), "Yan et al.  ·  IEEE Transactions on Image Processing  ·  2007", font=F_BODY, fill=MUTED)
    draw.text((150, 930), "COMP3314  /  GROUP 27", font=F_LABEL, fill=TEAL)


DRAWERS = [draw_scene_1, draw_scene_2, draw_scene_3, draw_scene_4, draw_scene_5, draw_scene_6, draw_scene_7]


def render_frame(scene_index: int, local_t: float) -> Image.Image:
    image = background()
    draw = ImageDraw.Draw(image)
    chrome(draw, scene_index, local_t)
    DRAWERS[scene_index](image, draw, local_t)
    fade = min(1.0, local_t / 0.24, (SCENES[scene_index][0] - local_t) / 0.24)
    if fade < 1.0:
        base = Image.new("RGB", (WIDTH, HEIGHT), BG)
        image = Image.blend(base, image, ease(fade))
    return image


def render_video() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode the MP4")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUTPUT),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    frame_total = math.ceil(TOTAL_SECONDS * FPS)
    scene_bounds: list[tuple[float, float]] = []
    cursor = 0.0
    for duration, _ in SCENES:
        scene_bounds.append((cursor, cursor + duration))
        cursor += duration
    try:
        for frame_index in range(frame_total):
            now = frame_index / FPS
            scene_index = next((i for i, (_, end) in enumerate(scene_bounds) if now < end), len(SCENES) - 1)
            start, _ = scene_bounds[scene_index]
            frame = render_frame(scene_index, now - start)
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    status = process.wait()
    if status:
        raise subprocess.CalledProcessError(status, command)
    print(f"Created {OUTPUT} ({frame_total} frames, {TOTAL_SECONDS:.1f}s, {WIDTH}x{HEIGHT}, {FPS} fps)")


if __name__ == "__main__":
    render_video()
