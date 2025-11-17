# kk_parse.py
# 负责解析 OpenMolcas .out：几何优化步 + CI 信息

import glob
import os
import re
from typing import Dict, List, Any


def find_openmolcas_out() -> str:
    """在当前目录查找最新的 OpenMolcas .out 文件"""
    outs = glob.glob("*.out")
    if not outs:
        return ""
    candidates = []
    for f in outs:
        try:
            with open(f, "r", errors="ignore") as fh:
                head = fh.read(5000)
            if "OpenMolcas" in head or "OPENMOLCAS" in head:
                candidates.append(f)
        except Exception:
            continue
    if not candidates:
        return ""
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def parse_geom_stats(text: str) -> List[Dict[str, Any]]:
    """
    每一个 “Energy Statistics for Geometry Optimization” 区块视为一个几何步。
    返回 steps 列表（按时间顺序）:
      steps[i] = {
        pos, iter, energy, grad_norm, grad_max,
        disp_rms, disp_max, grad_rms, grad_max_v,
        disp_thr, disp_max_thr, grad_thr, grad_max_thr
      }
    """
    stats_iter = list(re.finditer(r"Energy Statistics for Geometry Optimization", text))
    steps: List[Dict[str, Any]] = []

    for i, m in enumerate(stats_iter):
        start = m.start()
        end = stats_iter[i + 1].start() if i + 1 < len(stats_iter) else len(text)
        segment = text[start:end]
        lines = segment.splitlines()

        # 找 Iter      Energy 行
        hdr_idx = None
        for j, l in enumerate(lines):
            if l.strip().startswith("Iter      Energy"):
                hdr_idx = j
                break
        if hdr_idx is None:
            continue

        # 表格里的数字行
        tbl_lines = []
        for l in lines[hdr_idx + 1 :]:
            if re.match(r"\s*\d+\s", l):
                tbl_lines.append(l)
            elif l.strip() == "":
                continue
            elif l.strip().startswith("+----------------------------------+"):
                break

        if not tbl_lines:
            continue

        last_line = tbl_lines[-1]
        parts = last_line.split()
        if len(parts) < 5:
            continue

        it = int(parts[0])
        en = float(parts[1])
        grad_norm = float(parts[3])
        grad_max = abs(float(parts[4]))

        disp_rms = disp_thr = grad_rms = grad_thr = None
        disp_max = disp_max_thr = grad_max_val = grad_max_thr = None

        for l in lines:
            if l.strip().startswith("+ RMS +"):
                nums = re.findall(r"([+-]?\d+\.\d+E[+-]\d+)", l)
                if len(nums) >= 4:
                    disp_rms, disp_thr, grad_rms, grad_thr = map(float, nums[:4])
            if l.strip().startswith("+ Max +"):
                nums = re.findall(r"([+-]?\d+\.\d+E[+-]\d+)", l)
                if len(nums) >= 4:
                    disp_max, disp_max_thr, grad_max_val, grad_max_thr = map(float, nums[:4])

        steps.append(
            dict(
                pos=start,
                iter=it,
                energy=en,
                grad_norm=grad_norm,
                grad_max=grad_max,
                disp_rms=disp_rms,
                disp_max=disp_max,
                grad_rms=grad_rms,
                grad_max_v=grad_max_val,
                disp_thr=disp_thr,
                disp_max_thr=disp_max_thr,
                grad_thr=grad_thr,
                grad_max_thr=grad_max_thr,
            )
        )

    return steps


def parse_ci_block(text: str, start_pos: int) -> Dict[str, Any]:
    """解析一个 CI block"""
    segment = text[start_pos : start_pos + 1200]
    lines = segment.splitlines()
    idx = 1  # 行 0 是标题

    energy = None
    if idx < len(lines) and "energy" in lines[idx]:
        m = re.search(r"energy=\s*([+-]?\d+\.\d+)", lines[idx])
        if m:
            try:
                energy = float(m.group(1))
            except ValueError:
                energy = None
        idx += 1

    while idx < len(lines) and "conf/sym" not in lines[idx]:
        idx += 1
    if idx >= len(lines):
        return dict(energy=energy, configs=[])

    idx += 1
    configs = []
    while idx < len(lines):
        line = lines[idx]
        if not line.strip():
            break
        parts = line.split()
        if len(parts) < 4:
            break
        try:
            index = int(parts[0])
        except ValueError:
            break
        conf = " ".join(parts[1:-2])
        try:
            coeff = float(parts[-2])
            weight = float(parts[-1])
        except ValueError:
            break
        configs.append(
            dict(idx=index, conf=conf, coeff=coeff, weight=weight)
        )
        idx += 1

    return dict(energy=energy, configs=configs)


def build_ci_by_step(text: str, steps: List[Dict[str, Any]]) -> Dict[int, Dict[int, Dict[str, Any]]]:
    """
    把 CI block 按几何步分组：
      ci_by_step[step_index][root] = {energy, configs}
    """
    geom_positions = [s["pos"] for s in steps]
    ci_by_step: Dict[int, Dict[int, Dict[str, Any]]] = {}

    ci_matches = list(
        re.finditer(
            r"printout of CI-coefficients larger than  0.05 for root\s+(\d+)",
            text,
        )
    )

    for m in ci_matches:
        pos = m.start()
        root = int(m.group(1))

        step_idx = None
        for i, gpos in enumerate(geom_positions, start=1):
            if pos < gpos:
                step_idx = i
                break
        if step_idx is None:
            step_idx = len(geom_positions)

        info = parse_ci_block(text, pos)
        ci_by_step.setdefault(step_idx, {})
        ci_by_step[step_idx][root] = info

    return ci_by_step


def find_optimized_root(text: str) -> int:
    """从 rlxroot= 里解析优化态编号，没有就默认 1"""
    m = list(
        re.finditer(r"rlxroot\s*=\s*([0-9,\s]+)", text, flags=re.IGNORECASE)
    )
    if m:
        last = m[-1]
        g = last.group(1)
        nums = re.findall(r"\d+", g)
        if nums:
            return int(nums[0])
    return 1

