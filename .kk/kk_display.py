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

def _first_two_configs(ci_by_step, step, root):
    info_step = ci_by_step.get(step, {})
    info = info_step.get(root)
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
    ci_by_step: Dict[int, Dict[str, Any]]
):
    clear_below(TEXT_START)

    move_cursor(TEXT_START)

    nsteps = len(steps)
    info = ci_by_step.get(step, {})
    H2KCAL = 627.509474

    # 当前优化 root 的能量
    opt_info = info.get(opt_root)
    e_opt = opt_info.get("energy") if opt_info else None

    print(
        f"[kk] File: {out_path}   Steps: {nsteps} (current: {step})   rlxroot: {opt_root}"
    )

    # 表头（反白）
    hdr = (
        "Root  E (Eh)        ΔE (kcal/mol)   Major Config                Weight"
    )
    print("\033[1;7m" + hdr + "\033[0m")

    # 打印三个 root（opt_root 前后）
    for r in [opt_root - 1, opt_root, opt_root + 1]:
        if r <= 0:
            continue

        ri = info.get(r)
        if not ri:
            continue

        E = ri.get("energy")
        if E is not None and e_opt is not None and r != opt_root:
            dE = (E - e_opt) * H2KCAL
            dE_s = f"{dE:+.2f}"
        else:
            dE_s = ""

        E_s = f"{E: .6f}" if E is not None else " " * 10

        cfgs = _first_two_configs(ci_by_step, step, r)

        if r == opt_root:
            root_label = f"{r}*"
            prefix = "\033[1m"
            suffix = "\033[0m"
        else:
            root_label = f"{r}"
            prefix = ""
            suffix = ""

        if cfgs:
            c0 = cfgs[0]
            conf0 = f"{c0['idx']:7d} {c0['conf']}"
            w0 = f"{c0['weight']: .4f}"
        else:
            conf0 = ""
            w0 = ""

        line1 = (
            f"{root_label:<4} {E_s:>12} {dE_s:>14}   "
            f"{conf0:<26} {w0:>7}"
        )
        print(prefix + line1 + suffix)

        if len(cfgs) > 1:
            c1 = cfgs[1]
            conf1 = f"{c1['idx']:7d} {c1['conf']}"
            w1 = f"{c1['weight']: .4f}"
            line2 = (
                f"{'':4} {'':12} {'':14}   "
                f"{conf1:<26} {w1:>7}"
            )
            print(prefix + line2 + suffix)

    print("\n[kk] ↑ / ↓ 切换步数，q 退出。", end="", flush=True)


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

