#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终端分子查看器（sixel + python-sixel 版）

特性：
- 读取 xyz 文件
- 鼠标左键拖动：绕屏幕水平/垂直轴旋转分子
- 鼠标滚轮：缩放
- 方向键：旋转（无鼠标时也可用）
- 按 q 退出
- 画布大小由终端字符行列决定，尽量 100% 填满终端（最后一行留给提示）
- 通过 ~/.mol_sixel_viewer.json 配置多套主题，控制颜色/球半径等

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

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".mol_sixel_viewer.json")

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
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            deep_update(cfg, user_cfg)
        except Exception as e:
            # 配置坏了就忽略，用默认
            sys.stderr.write(f"[mol_sixel_viewer] 配置文件加载失败：{e}\n")
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

def draw_molecule(symbols, coords, bonds, R, zoom, img_w, img_h, base_scale, theme):
    """生成一张 PNG 图像（PIL Image），简单球棒模型."""
    bg = tuple(theme["background"])
    bond_color = tuple(theme["bond_color"])
    bond_width = int(theme.get("bond_width", 2))

    element_style = theme.get("element_style", {})

    xs, ys, zs = project_atoms(coords, R, zoom, img_w, img_h, base_scale)

    img = Image.new("RGB", (img_w, img_h), bg)
    draw = ImageDraw.Draw(img)

    # sort index by z -> 远处先画，近处后画
    order = np.argsort(zs)

    # 先画键（在原子下面）
    for i, j in bonds:
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[j], ys[j]
        draw.line((x1, y1, x2, y2), fill=bond_color, width=bond_width)

    # 再画原子球
    for idx in order:
        sym = symbols[idx]
        x, y = xs[idx], ys[idx]

        style = element_style.get(sym, {})
        color = tuple(style.get("color", [80, 80, 80]))
        r0 = float(style.get("radius", 8))

        # 半径随 zoom 缩放（限定范围避免太离谱）
        r = max(1.0, min(80.0, r0 * zoom))
        bbox = (x - r, y - r, x + r, y + r)
        draw.ellipse(bbox, fill=color, outline=(0, 0, 0))

    return img


def sixel_show_image(png_path, top_row=1, left_col=1):
    """
    在终端中用 python-sixel 显示 png 图像。
    top_row/left_col：从第几行、第几列开始画（1-based）
    """
    # 把光标移到指定行列（图像左上角）
    sys.stdout.write(f"\x1b[{top_row};{left_col}H")
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

    # 用分子整体半径决定 base_scale（固定），只在窗口大小或配置变化时重算
    # 这里用 coords 的范数最大值作为分子“半径”
    radii = np.linalg.norm(coords, axis=1)
    max_r = max(radii.max(), 1e-3)

    # 根据终端尺寸估一个图片大小（尽量撑满，最后一行留给提示）
    term_h, term_w = stdscr.getmaxyx()
    cell_w, cell_h = theme.get("pixel_per_char", [8, 16])
    cell_w = int(cell_w or 8)
    cell_h = int(cell_h or 16)

    img_w = max(200, term_w * cell_w)
    img_h = max(200, max(term_h - 1, 1) * cell_h)

    fill_ratio = float(theme.get("fill_ratio", 0.8))
    fill_ratio = max(0.1, min(fill_ratio, 0.98))

    # 让分子半径在画布中占据 fill_ratio * min(img_w, img_h)/2
    base_scale = (min(img_w, img_h) * fill_ratio) / (2.0 * max_r)

    last_mouse = None   # 上一个鼠标位置 (x, y)
    dirty = True        # 是否需要重绘

    # 临时目录保存 PNG
    tmpdir = tempfile.mkdtemp(prefix="mol_sixel_viewer_")
    png_path = os.path.join(tmpdir, "frame.png")

    try:
        while True:
            ch = stdscr.getch()

            # 退出
            if ch == ord("q"):
                break

            # 方向键：绕屏幕固定轴旋转
            step = 0.05
            if ch == curses.KEY_LEFT:
                R = rot_y(-step) @ R
                dirty = True
            elif ch == curses.KEY_RIGHT:
                R = rot_y(step) @ R
                dirty = True
            elif ch == curses.KEY_UP:
                R = rot_x(-step) @ R
                dirty = True
            elif ch == curses.KEY_DOWN:
                R = rot_x(step) @ R
                dirty = True

            # 终端大小改变 -> 重新计算画布尺寸和 base_scale
            if ch == curses.KEY_RESIZE:
                term_h, term_w = stdscr.getmaxyx()
                img_w = max(200, term_w * cell_w)
                img_h = max(200, max(term_h - 1, 1) * cell_h)
                base_scale = (min(img_w, img_h) * fill_ratio) / (2.0 * max_r)
                dirty = True

            # 鼠标事件
            if ch == curses.KEY_MOUSE:
                try:
                    _id, mx, my, _z, bstate = curses.getmouse()
                except Exception:
                    bstate = 0

                # 滚轮放大/缩小
                if bstate & curses.BUTTON4_PRESSED:
                    zoom *= 1.1
                    dirty = True
                if bstate & curses.BUTTON5_PRESSED:
                    zoom /= 1.1
                    zoom = max(0.1, min(zoom, 10.0))
                    dirty = True

                # 左键拖动：dx 控制 yaw，dy 控制 pitch（绕屏幕水平轴）
                left_events = (
                    curses.BUTTON1_PRESSED
                    | curses.BUTTON1_CLICKED
                    | curses.BUTTON1_DOUBLE_CLICKED
                    | curses.BUTTON1_TRIPLE_CLICKED
                )

                if bstate & left_events:
                    if last_mouse is None:
                        last_mouse = (mx, my)
                    else:
                        dx = mx - last_mouse[0]
                        dy = my - last_mouse[1]
                        R = rot_y(dx * 0.03) @ R
                        R = rot_x(dy * 0.03) @ R
                        last_mouse = (mx, my)
                        dirty = True

                if bstate & curses.BUTTON1_RELEASED:
                    last_mouse = None

            # 没事件且不需要重画时，稍微休息一下
            if ch == -1 and not dirty:
                curses.napms(10)
                continue

            # 需要重画：生成 PNG + 输出 sixel + 更新底部提示
            if dirty:
                dirty = False

                img = draw_molecule(
                    symbols, coords, bonds, R, zoom, img_w, img_h, base_scale, theme
                )
                img.save(png_path)

                # 在第 1 行开始输出 sixel 图片（填满终端，除最后一行）
                sixel_show_image(png_path, top_row=1, left_col=1)

                # 更新底部提示行
                term_h, term_w = stdscr.getmaxyx()
                info = (
                    f"XYZ: {os.path.basename(xyz_path)} | 鼠标拖动/方向键旋转，"
                    f"滚轮缩放，q 退出 | zoom={zoom:.2f} | theme"
                )
                try:
                    stdscr.move(term_h - 1, 0)
                    stdscr.clrtoeol()
                    stdscr.addstr(term_h - 1, 0, info[: term_w - 1])
                    stdscr.refresh()
                except curses.error:
                    pass

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
