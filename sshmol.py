#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终端分子查看器（sixel + python-sixel 版）

特性：
- 读取 xyz 文件
- 鼠标滚轮 / 键盘 +-：缩放
- 方向键：旋转
- 按 q 退出
- 画布固定为 1200×600（超出终端时会溢出），最后一行留给提示
- 通过当前目录或 ~ 目录下的 .sshmol/config.json 配置多套主题，控制颜色/球半径等

依赖：
  pip install --user numpy pillow sixel

用法：
  python mol_sixel_viewer.py molecule.xyz
"""

import curses
import math
import os
import sys
import tempfile
import json
import time

import numpy as np
from PIL import Image, ImageDraw
from sixel import converter

# -------------------- 默认配置 --------------------

DEFAULT_CONFIG = {
    "default_theme": "light",
    "themes": {
        "light": {
            "background": [255, 255, 255],
            "bond_color": [150, 150, 150],
            "bond_width": 2,
            # 分子占画布的比例（0~1，越大越贴边）
            "fill_ratio": 0.8,
            # 估计每个字符的像素大小（宽, 高）
            "pixel_per_char": [8, 16],
            # 元素风格：颜色 + 基础半径（zoom=1 时）
            "element_style": {
                "H": {"color": [240, 240, 240], "radius": 6},
                "C": {"color": [50, 50, 50], "radius": 10},
                "N": {"color": [50, 50, 200], "radius": 10},
                "O": {"color": [200, 30, 30], "radius": 10},
                "F": {"color": [30, 200, 30], "radius": 10},
                "P": {"color": [255, 165, 0], "radius": 12},
                "S": {"color": [255, 215, 0], "radius": 12},
                "Cl": {"color": [0, 200, 0], "radius": 12},
                "Br": {"color": [165, 42, 42], "radius": 12},
                "I": {"color": [148, 0, 211], "radius": 14}
            }
        },
        "dark": {
            "background": [20, 20, 20],
            "bond_color": [220, 220, 220],
            "bond_width": 2,
            "fill_ratio": 0.7,
            "pixel_per_char": [8, 16],
            "element_style": {
                "H": {"color": [200, 200, 200], "radius": 6},
                "C": {"color": [230, 230, 230], "radius": 10},
                "N": {"color": [120, 160, 255], "radius": 10},
                "O": {"color": [255, 130, 130], "radius": 10},
                "F": {"color": [140, 255, 140], "radius": 10},
                "P": {"color": [255, 200, 120], "radius": 12},
                "S": {"color": [255, 230, 120], "radius": 12}
            }
        }
    }
}

CONFIG_CANDIDATES = [
    os.path.join(os.getcwd(), ".sshmol", "config.json"),
    os.path.join(os.path.expanduser("~"), ".sshmol", "config.json"),
    os.path.join(os.path.expanduser("~"), ".mol_sixel_viewer.json"),
]

MAX_IMG_W = 1200
MAX_IMG_H = 600
FAST_MODE_COLORS = 16
HIGH_MODE_COLORS = 256
HIGH_QUALITY_IDLE_SECONDS = 1.0
MIN_ZOOM = 0.1
MAX_ZOOM = 10.0
SLOW_RENDER_THRESHOLD = 0.12

# 简单的共价半径（Å），只为判断是否画键
COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}


# -------------------- 配置与主题 --------------------

def deep_update(base, overrides):
    """递归地用 overrides 更新 base 字典."""
    for k, v in overrides.items():
        if (
            isinstance(v, dict)
            and k in base
            and isinstance(base[k], dict)
        ):
            deep_update(base[k], v)
        else:
            base[k] = v


def load_config():
    """加载用户配置（如果存在），与默认配置合并。"""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # 深拷贝
    chosen_path = None
    for path in CONFIG_CANDIDATES:
        if path and os.path.isfile(path):
            chosen_path = path
            break

    if chosen_path:
        try:
            with open(chosen_path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            deep_update(cfg, user_cfg)
        except Exception as e:
            sys.stderr.write(
                f"[mol_sixel_viewer] 配置文件加载失败（{chosen_path}）：{e}\n"
            )
    return cfg


def get_theme(config, theme_name=None):
    """根据配置获取主题."""
    themes = config.get("themes", {})
    if theme_name is None:
        theme_name = config.get("default_theme", "light")

    theme = themes.get(theme_name)
    if theme is None and themes:
        # 退回第一个主题
        theme = next(iter(themes.values()))
    if theme is None:
        theme = DEFAULT_CONFIG["themes"]["light"]

    # 补全缺失字段
    base = DEFAULT_CONFIG["themes"]["light"]
    merged = json.loads(json.dumps(base))
    deep_update(merged, theme)
    return merged


def clamp_zoom(value):
    return max(MIN_ZOOM, min(MAX_ZOOM, value))


def compute_base_scale(max_radius, img_w, img_h, theme):
    """Compute how much to scale projected coordinates to fit the canvas."""
    fill_ratio = float(theme.get("fill_ratio", 0.8)) if theme else 0.8
    fill_ratio = max(0.1, min(fill_ratio, 0.98))
    safe_radius = max(float(max_radius), 1e-3)
    return (min(img_w, img_h) * fill_ratio) / (2.0 * safe_radius)


# -------------------- 分子数据处理 --------------------

def parse_xyz(path):
    """读取 xyz 文件，返回 (symbols, coords)，coords 为 (N,3) numpy 数组."""
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 3:
        raise ValueError("XYZ 文件太短，看起来不合法")

    try:
        n_atoms = int(lines[0].split()[0])
    except Exception as e:
        raise ValueError("XYZ 第一行不是原子数") from e

    if len(lines) < n_atoms + 2:
        raise ValueError("XYZ 文件的原子行数量不够")

    symbols = []
    coords = []
    for line in lines[2: 2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            continue
        sym = parts[0]
        x, y, z = map(float, parts[1:4])
        symbols.append(sym)
        coords.append([x, y, z])

    coords = np.array(coords, dtype=float)
    # 坐标居中
    center = coords.mean(axis=0)
    coords -= center
    return symbols, coords


def build_bonds(symbols, coords):
    """根据简单的共价半径判断是否画键，返回 bond 列表 (i, j)."""
    n = len(symbols)
    bonds = []
    for i in range(n):
        ri = COVALENT_RADII.get(symbols[i], 0.8)
        for j in range(i + 1, n):
            rj = COVALENT_RADII.get(symbols[j], 0.8)
            cutoff = 1.2 * (ri + rj)
            d = np.linalg.norm(coords[i] - coords[j])
            if 0.1 < d < cutoff:
                bonds.append((i, j))
    return bonds


# -------------------- 旋转与投影 --------------------

def rot_x(angle):
    """绕全局 x 轴的 3x3 旋转矩阵."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def rot_y(angle):
    """绕全局 y 轴的 3x3 旋转矩阵."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def project_atoms(coords, R, zoom, img_w, img_h, base_scale):
    """
    把 3D 坐标用当前旋转矩阵 R 投影到 2D 像素坐标.
    base_scale: 已经根据分子整体 + 画布尺寸算好的固定缩放因子
    """
    rotated = coords @ R.T  # (N,3)
    xs = rotated[:, 0] * (base_scale * zoom) + img_w / 2.0
    ys = -rotated[:, 1] * (base_scale * zoom) + img_h / 2.0
    zs = rotated[:, 2]
    return xs, ys, zs


# -------------------- 绘图与 sixel 输出 --------------------

def _apply_color_budget(img, colors):
    """将图像量化到指定颜色数量，减小 sixel 数据量."""
    if colors >= 256:
        colors = min(256, colors)
        method = Image.MEDIANCUT
    else:
        colors = max(2, colors)
        method = Image.FASTOCTREE
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img.quantize(colors=colors, method=method).convert("RGB")


def _draw_high_quality_sphere(canvas, color, r, center):
    diameter = max(2, int(round(r * 2)))
    sphere = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
    sphere_draw = ImageDraw.Draw(sphere, "RGBA")
    sphere_draw.ellipse((0, 0, diameter - 1, diameter - 1), fill=color + (255,))

    # 简单高光/阴影
    highlight = tuple(min(255, int(c * 1.2)) for c in color) + (120,)
    shadow = tuple(int(c * 0.5) for c in color) + (80,)
    offset = int(diameter * 0.15)
    sphere_draw.ellipse(
        (
            offset,
            offset,
            diameter - 1,
            diameter - 1,
        ),
        outline=shadow,
        width=max(1, diameter // 20),
    )
    sphere_draw.ellipse(
        (
            diameter * 0.25,
            diameter * 0.2,
            diameter * 0.65,
            diameter * 0.55,
        ),
        fill=highlight,
    )

    x, y = center
    canvas.paste(
        sphere,
        (
            int(round(x - diameter / 2)),
            int(round(y - diameter / 2)),
        ),
        sphere,
    )


def draw_molecule(
    symbols,
    coords,
    bonds,
    R,
    zoom,
    img_w,
    img_h,
    base_scale=None,
    theme=None,
    quality="fast",
):
    """生成一张 PNG 图像（PIL Image），简单球棒模型."""
    if theme is None and isinstance(base_scale, dict):  # 兼容旧版调用顺序
        theme = base_scale
        base_scale = None

    if theme is None:
        theme = DEFAULT_CONFIG["themes"]["light"]

    if base_scale is None:
        radii = np.linalg.norm(coords, axis=1) if len(coords) else np.array([1e-3])
        max_r = float(radii.max()) if radii.size else 1e-3
        base_scale = compute_base_scale(max_r, img_w, img_h, theme)

    bg = tuple(theme["background"])
    bond_color = tuple(theme["bond_color"])
    bond_width = int(theme.get("bond_width", 2))

    element_style = theme.get("element_style", {})

    xs, ys, zs = project_atoms(coords, R, zoom, img_w, img_h, base_scale)

    aa_scale = 2 if quality == "high" else 1
    canvas_size = (img_w * aa_scale, img_h * aa_scale)
    if quality == "high":
        img = Image.new("RGBA", canvas_size, bg + (255,))
        draw = ImageDraw.Draw(img, "RGBA")
    else:
        img = Image.new("RGB", canvas_size, bg)
        draw = ImageDraw.Draw(img)

    xs_draw = xs * aa_scale
    ys_draw = ys * aa_scale

    # sort index by z -> 远处先画，近处后画
    order = np.argsort(zs)

    # 先画键（在原子下面）
    line_width = max(1, int(round(bond_width * aa_scale)))
    for i, j in bonds:
        x1, y1 = xs_draw[i], ys_draw[i]
        x2, y2 = xs_draw[j], ys_draw[j]
        draw.line((x1, y1, x2, y2), fill=bond_color, width=line_width)

    # 再画原子球
    for idx in order:
        sym = symbols[idx]
        x, y = xs_draw[idx], ys_draw[idx]

        style = element_style.get(sym, {})
        color = tuple(style.get("color", [80, 80, 80]))
        r0 = float(style.get("radius", 8))

        # 半径随 zoom 缩放（限定范围避免太离谱）
        r = max(1.0, min(80.0, r0 * zoom)) * aa_scale
        if quality == "high":
            _draw_high_quality_sphere(img, color, r, (x, y))
        else:
            bbox = (x - r, y - r, x + r, y + r)
            draw.ellipse(bbox, fill=color, outline=(0, 0, 0))

    if aa_scale > 1:
        img = img.resize((img_w, img_h), Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def sixel_show_image(png_path, top_row=1, left_col=1):
    """
    在终端中用 python-sixel 显示 png 图像。
    top_row/left_col：从第几行、第几列开始画（1-based）
    """
    # 把光标移到指定行列（图像左上角），并清空其后的区域，避免上一帧的残影
    sys.stdout.write(f"\x1b[{top_row};{left_col}H")
    sys.stdout.write("\x1b[J")
    sys.stdout.flush()

    c = converter.SixelConverter(png_path)
    c.write(sys.stdout)

    # 防止光标停在图片里
    sys.stdout.write("\n")
    sys.stdout.flush()


# -------------------- 主 viewer --------------------

def viewer(stdscr, xyz_path, theme):
    # 解析分子
    symbols, coords = parse_xyz(xyz_path)
    bonds = build_bonds(symbols, coords)

    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)
    curses.mousemask(curses.ALL_MOUSE_EVENTS)

    # 旋转矩阵，初始稍微倾斜一点
    R = rot_y(0.8) @ rot_x(0.5)
    zoom = 1.0

    # 用 coords 的范数最大值作为分子“半径”
    radii = np.linalg.norm(coords, axis=1)
    max_r = float(radii.max()) if radii.size else 1e-3

    # 根据终端尺寸估一个图片大小（尽量撑满，最后一行留给提示）
    img_w = MAX_IMG_W
    img_h = MAX_IMG_H

    base_scale = compute_base_scale(max_r, img_w, img_h, theme)

    dirty = True        # 是否需要重绘
    quality_mode = "fast"
    last_render_quality = None
    last_input_time = time.monotonic()

    # 临时目录保存 PNG
    tmpdir = tempfile.mkdtemp(prefix="mol_sixel_viewer_")
    png_path = os.path.join(tmpdir, "frame.png")

    try:
        exit_requested = False
        while True:
            events_processed = 0
            while True:
                ch = stdscr.getch()
                if ch == -1:
                    break
                events_processed += 1
                now = time.monotonic()
                last_input_time = now

                if ch == ord("q"):
                    exit_requested = True
                    break

                step = 0.05
                if ch == curses.KEY_LEFT:
                    R = rot_y(-step) @ R
                    dirty = True
                    quality_mode = "fast"
                elif ch == curses.KEY_RIGHT:
                    R = rot_y(step) @ R
                    dirty = True
                    quality_mode = "fast"
                elif ch == curses.KEY_UP:
                    R = rot_x(-step) @ R
                    dirty = True
                    quality_mode = "fast"
                elif ch == curses.KEY_DOWN:
                    R = rot_x(step) @ R
                    dirty = True
                    quality_mode = "fast"
                elif ch in (ord("+"), ord("=")):
                    zoom = clamp_zoom(zoom * 1.1)
                    dirty = True
                    quality_mode = "fast"
                elif ch in (ord("-"), ord("_")):
                    zoom = clamp_zoom(zoom / 1.1)
                    dirty = True
                    quality_mode = "fast"
                elif ch == curses.KEY_RESIZE:
                    dirty = True
                elif ch == curses.KEY_MOUSE:
                    try:
                        _id, _mx, _my, _z, bstate = curses.getmouse()
                    except Exception:
                        bstate = 0

                    if bstate & curses.BUTTON4_PRESSED:
                        zoom = clamp_zoom(zoom * 1.1)
                        dirty = True
                        quality_mode = "fast"
                    if bstate & curses.BUTTON5_PRESSED:
                        zoom = clamp_zoom(zoom / 1.1)
                        dirty = True
                        quality_mode = "fast"

            if exit_requested:
                break

            now = time.monotonic()

            if events_processed == 0 and not dirty:
                if (
                    last_render_quality != "high"
                    and (now - last_input_time) >= HIGH_QUALITY_IDLE_SECONDS
                ):
                    quality_mode = "high"
                    dirty = True
                    continue
                curses.napms(10)
                continue

            if dirty:
                dirty = False
                render_start = time.monotonic()
                img = draw_molecule(
                    symbols,
                    coords,
                    bonds,
                    R,
                    zoom,
                    img_w,
                    img_h,
                    base_scale,
                    theme,
                    quality=quality_mode,
                )
                color_budget = (
                    HIGH_MODE_COLORS if quality_mode == "high" else FAST_MODE_COLORS
                )
                img = _apply_color_budget(img, color_budget)
                img.save(png_path)
                last_render_quality = quality_mode

                sixel_show_image(png_path, top_row=1, left_col=1)

                term_h, term_w = stdscr.getmaxyx()
                mode_label = (
                    "高质量256色" if quality_mode == "high" else "快速16色"
                )
                info = (
                    f"XYZ: {os.path.basename(xyz_path)} | 方向键旋转，"
                    f"滚轮/+- 缩放，q 退出 | zoom={zoom:.2f} | {mode_label}"
                )
                try:
                    stdscr.move(term_h - 1, 0)
                    stdscr.clrtoeol()
                    stdscr.addstr(term_h - 1, 0, info[: term_w - 1])
                    stdscr.refresh()
                except curses.error:
                    pass

                render_duration = time.monotonic() - render_start
                # 渲染太慢时就保持快速模式，而不是跳帧避免闪烁
                if events_processed > 0 and render_duration > SLOW_RENDER_THRESHOLD:
                    quality_mode = "fast"

    finally:
        # 清理临时文件
        try:
            for fn in os.listdir(tmpdir):
                os.remove(os.path.join(tmpdir, fn))
            os.rmdir(tmpdir)
        except Exception:
            pass


def main():
    if len(sys.argv) < 2:
        print("用法: python mol_sixel_viewer.py molecule.xyz")
        sys.exit(1)

    xyz_path = sys.argv[1]
    if not os.path.isfile(xyz_path):
        print(f"找不到文件: {xyz_path}")
        sys.exit(1)

    # 加载配置和主题
    config = load_config()
    # 这里暂时不做命令行切换主题，如需切换可在配置里改 default_theme
    theme = get_theme(config, theme_name=None)

    curses.wrapper(lambda stdscr: viewer(stdscr, xyz_path, theme))


if __name__ == "__main__":
    main()
