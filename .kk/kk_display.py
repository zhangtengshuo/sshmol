# kk_display.py
# 只在首次显示时画图，之后按上下键只更新文字区域，不重刷图像

import sys
import os
import shutil
import base64
import termios
import tty
from typing import Dict, List, Any


# ======================
# 屏幕控制
# ======================

def enter_alt_screen():
    sys.stdout.write("\033[?1049h\033[?25l")
    sys.stdout.flush()


def exit_alt_screen():
    sys.stdout.write("\033[?1049l\033[?25h")
    sys.stdout.flush()


def move_cursor(row: int, col: int = 1):
    """移动光标到第 row 行第 col 列"""
    sys.stdout.write(f"\033[{row};{col}H")
    sys.stdout.flush()


def clear_below(row: int):
    """清除 row 行以下内容"""
    sys.stdout.write(f"\033[{row};1H\033[J")
    sys.stdout.flush()


# ======================
# SIXEL 显示
# ======================

def show_png_in_terminal(png_path: str) -> bool:
    """
    只调用一次，在屏幕顶部输出图片。
    后续绝不重复输出 SIXEL，从而避免闪屏。
    """
    try:
        from sixel import converter
        move_cursor(1, 1)
        c = converter.SixelConverter(png_path)
        c.write(sys.stdout)
        sys.stdout.flush()
        return True
    except Exception:
        pass

    if shutil.which("img2sixel"):
        move_cursor(1, 1)
        os.system(f"img2sixel '{png_path}'")
        return True

    return False


# ======================
# 键盘读取
# ======================

def read_key() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        c1 = sys.stdin.read(1)
        if c1 == "\x1b":
            c2 = sys.stdin.read(1)
            if c2 == "[":
                c3 = sys.stdin.read(1)
                if c3 == "A": return "UP"
                if c3 == "B": return "DOWN"
            return ""
        return c1
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ======================
# CI selection（按你之前的要求）
# ======================

def _first_two_configs(ci_by_step, step, subproject_key, root):
    info_step = ci_by_step.get(step, {})
    info_sub = info_step.get(subproject_key, {})
    info = info_sub.get(root)
    if not info:
        return []
    cfgs = info.get("configs", [])
    if not cfgs:
        return []

    # 权重大 → 选 2 个
    top_weight = sorted(cfgs, key=lambda c: c["weight"], reverse=True)[:2]
    # 显示按 idx 排序
    return sorted(top_weight, key=lambda c: c["idx"])


# ======================
# 文本区域刷新（关键）
# ======================

TEXT_START = 23   # 假设图片高度占 26 行，文字从 27 行开始


def print_step_info(
    out_path: str,
    steps: List[Dict[str, Any]],
    step: int,
    opt_root: int,
    ci_by_step: Dict[int, Dict[str, Any]],
    state_targets: List[Dict[str, Any]],
    state_energies: Dict[int, Dict[str, Any]],
):
    clear_below(TEXT_START)
    move_cursor(TEXT_START)

    nsteps = len(steps)
    H2KCAL = 627.509474
    step_idx = max(1, min(step, nsteps))
    info_step = ci_by_step.get(step_idx, {})
    energy_step = state_energies.get(step_idx, {})

    if len(state_targets) == 1:
        summary = (
            f"[kk] File: {out_path}   Steps: {nsteps} (current: {step_idx})   "
            f"rlxroot: {opt_root}"
        )
    else:
        tracked = ", ".join(
            f"{t['group_label']} r{t['root']}" for t in state_targets
        )
        summary = (
            f"[kk] File: {out_path}   Steps: {nsteps} (current: {step_idx})   "
            f"Targets: {tracked}"
        )
    print(summary)

    def get_energy(sub_key: str, root: int, fallback: Any = None):
        sub_data = energy_step.get(sub_key, {})
        info = sub_data.get(root)
        if info:
            label = info.get("raw_label") or info.get("method", "")
            return info.get("energy"), label
        if fallback is not None:
            return fallback, "Geom"
        return None, ""

    def group_label(name: str) -> str:
        if not name:
            return "State"
        label = name.lstrip(".") or name
        return label

    target_subprojects = {(t["subproject"] or "") for t in state_targets}
    is_interspin = len(state_targets) == 2 and len(target_subprojects) == 2

    LEVEL_W = 14
    ROOT_W = 5
    E_W = 12
    DELTA_W = 6
    CONFIG_W = 28
    WEIGHT_W = 6

    def pad_text(text: str, width: int, align: str = "left") -> str:
        if text is None:
            text = ""
        text = str(text)
        if align == "right":
            return text.rjust(width)
        if align == "center":
            return text.center(width)
        return text.ljust(width)

    def format_header(include_level: bool, show_delta: bool) -> str:
        parts = []
        if include_level:
            parts.append(pad_text("Level", LEVEL_W))
        parts.append(pad_text("Root", ROOT_W))
        parts.append(pad_text("E (Eh)", E_W, "right"))
        if show_delta:
            parts.append(pad_text("ΔE", DELTA_W, "right"))
        parts.append(pad_text("Major Config", CONFIG_W))
        parts.append(pad_text("Weight", WEIGHT_W, "right"))
        return " ".join(parts)

    def format_line(
        include_level: bool,
        show_delta: bool,
        level_text: str,
        root_text: str,
        energy_text: str,
        delta_text: str,
        config_text: str,
        weight_text: str,
    ) -> str:
        parts = []
        if include_level:
            parts.append(pad_text(level_text, LEVEL_W))
        parts.append(pad_text(root_text, ROOT_W))
        parts.append(pad_text(energy_text, E_W, "right"))
        if show_delta:
            parts.append(pad_text(delta_text, DELTA_W, "right"))
        parts.append(pad_text(config_text, CONFIG_W))
        parts.append(pad_text(weight_text, WEIGHT_W, "right"))
        return " ".join(parts)

    group_map: Dict[str, Dict[str, Any]] = {}
    group_order: List[str] = []
    show_neighbors = len(state_targets) == 1

    for tgt in state_targets:
        key = tgt["subproject"]
        entry = group_map.setdefault(
            key,
            dict(
                label=tgt.get("group_label") or group_label(key),
                roots=[],
            ),
        )
        entry["roots"].append(dict(root=tgt["root"], is_target=True))
        if key not in group_order:
            group_order.append(key)

    if show_neighbors:
        tgt = state_targets[0]
        key = tgt["subproject"]
        extra = group_map.setdefault(
            key,
            dict(label=group_label(key), roots=[]),
        )
        for neighbor in (tgt["root"] - 1, tgt["root"] + 1):
            if neighbor > 0 and all(r["root"] != neighbor for r in extra["roots"]):
                extra["roots"].append(dict(root=neighbor, is_target=False))

    rendered_groups: List[Dict[str, Any]] = []

    fallback_energy = steps[step_idx - 1]["energy"] if steps else None
    show_delta_column = not is_interspin
    include_level_flags = {key: True for key in group_order}
    if is_interspin and len(group_order) >= 2:
        include_level_flags[group_order[1]] = False

    for key in group_order:
        block = group_map.get(key)
        if not block:
            continue
        title = block["label"]
        roots = block["roots"]
        # 保持 targets 在前
        roots.sort(key=lambda r: (not r["is_target"], r["root"]))
        include_level = include_level_flags.get(key, True)
        sub_lines: List[str] = []
        sub_lines.append(f"[{title}]")
        header_line = format_header(include_level, show_delta_column)
        sub_lines.append("\033[1;7m" + header_line + "\033[0m")

        ref_energy = None
        target_line_idx = None
        for r in roots:
            sub_key = key
            fb = fallback_energy if (show_neighbors and r["is_target"] and not sub_key) else None
            energy, method = get_energy(sub_key, r["root"], fb)
            if show_delta_column and ref_energy is None and r["is_target"] and energy is not None:
                ref_energy = energy

            if show_delta_column and energy is not None and ref_energy is not None:
                dE = (energy - ref_energy) * H2KCAL
                dE_s = f"{dE:+6.2f}"
            else:
                dE_s = ""

            E_s = f"{energy:.4f}" if energy is not None else ""
            level = method or ""

            cfgs = _first_two_configs(ci_by_step, step_idx, key, r["root"])
            if cfgs:
                c0 = cfgs[0]
                conf0 = f"{c0['idx']:7d} {c0['conf']}"
                w0 = f"{c0['weight']:.2f}"
            else:
                conf0 = ""
                w0 = ""

            root_label = f"{r['root']}{'*' if r['is_target'] else ''}"
            prefix = "\033[1m" if r["is_target"] else ""
            suffix = "\033[0m" if r["is_target"] else ""

            line1 = format_line(
                include_level,
                show_delta_column,
                level if include_level else "",
                root_label,
                E_s,
                dE_s,
                conf0,
                w0,
            )
            line_index = len(sub_lines)
            sub_lines.append(prefix + line1 + suffix)
            if r["is_target"] and target_line_idx is None:
                target_line_idx = line_index

            if len(cfgs) > 1:
                c1 = cfgs[1]
                conf1 = f"{c1['idx']:7d} {c1['conf']}"
                w1 = f"{c1['weight']:.2f}"
                line2 = format_line(
                    include_level,
                    show_delta_column,
                    "",
                    "",
                    "",
                    "",
                    conf1,
                    w1,
                )
                sub_lines.append(prefix + line2 + suffix)

        rendered_groups.append(dict(lines=sub_lines, target_line_idx=target_line_idx))

    if is_interspin and len(rendered_groups) == 2:
        left = rendered_groups[0]
        right = rendered_groups[1]
        left_lines = left["lines"]
        right_lines = right["lines"]
        width_left = max(len(line) for line in left_lines) if left_lines else 0
        rows = max(len(left_lines), len(right_lines))
        delta_col_width = max(len("ΔE (kcal/mol)"), DELTA_W + 4)
        delta_lines = ["" for _ in range(rows)]
        if rows > 1:
            delta_lines[1] = "ΔE (kcal/mol)"

        delta_value = ""
        if len(state_targets) >= 2:
            left_t = state_targets[0]
            right_t = state_targets[1]
            left_energy, _ = get_energy(left_t["subproject"], left_t["root"])
            right_energy, _ = get_energy(right_t["subproject"], right_t["root"])
            if left_energy is not None and right_energy is not None:
                diff = (right_energy - left_energy) * H2KCAL
                delta_value = f"{diff:+6.2f}"

        target_line = left.get("target_line_idx")
        if delta_value and isinstance(target_line, int) and target_line < rows:
            delta_lines[target_line] = delta_value

        for i in range(rows):
            l_line = left_lines[i] if i < len(left_lines) else ""
            r_line = right_lines[i] if i < len(right_lines) else ""
            delta_text = delta_lines[i] if i < len(delta_lines) else ""
            print(f"{l_line:<{width_left + 2}}{delta_text:^{delta_col_width}}{r_line}")
    elif len(rendered_groups) == 2:
        left = rendered_groups[0]["lines"]
        right = rendered_groups[1]["lines"]
        width_left = max(len(line) for line in left) if left else 0
        rows = max(len(left), len(right))
        for i in range(rows):
            l_line = left[i] if i < len(left) else ""
            r_line = right[i] if i < len(right) else ""
            print(f"{l_line:<{width_left + 4}}{r_line}")
    else:
        for group in rendered_groups:
            lines = group["lines"]
            for line in lines:
                print(line)
            print()

    print()
    print("[kk] ↑ / ↓ 切换步数，q 退出。", end="", flush=True)


# ======================
# 主交互流程（重要）
# ======================

def interactive_show(out_path, png_path, steps, ci_by_step, opt_root):
    enter_alt_screen()

    # 1) 进入备用屏幕后，首先显示图片（只做一次）
    show_png_in_terminal(png_path)

    # 初始步为最后一步
    step = len(steps)
    print_step_info(out_path, steps, step, opt_root, ci_by_step)

    # 2) 循环，只刷新文字区域，不刷新图片
    while True:
        key = read_key()
        if key == "q":
            break
        elif key == "UP":
            if step > 1:
                step -= 1
                print_step_info(out_path, steps, step, opt_root, ci_by_step)
        elif key == "DOWN":
            if step < len(steps):
                step += 1
                print_step_info(out_path, steps, step, opt_root, ci_by_step)

    exit_alt_screen()

