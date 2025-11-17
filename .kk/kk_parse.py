# kk_parse.py
# 负责解析 OpenMolcas .out：几何优化步 + CI 信息

import bisect
import glob
import os
import re
from typing import Dict, List, Any, Tuple


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


def _build_subproject_markers(text: str) -> Tuple[List[int], List[str]]:
    markers_pos: List[int] = []
    markers_val: List[str] = []
    pattern = re.compile(r">>>\s+EXPORT\s+SubProject[ \t]*=[ \t]*([^\r\n]*)")
    for m in pattern.finditer(text):
        name = m.group(1).strip()
        if not name:
            name = None
        markers_pos.append(m.start())
        markers_val.append(name)
    return markers_pos, markers_val


def list_subprojects(text: str) -> List[str]:
    """Ordered unique SubProject names (non-empty)."""
    seen = set()
    ordered: List[str] = []
    pattern = re.compile(r">>>\s+EXPORT\s+SubProject[ \t]*=[ \t]*([^\r\n]*)")
    for m in pattern.finditer(text):
        name = m.group(1).strip()
        if not name:
            continue
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def _subproject_for_pos(pos: int, markers_pos: List[int], markers_val: List[str]) -> str:
    if not markers_pos:
        return ""
    idx = bisect.bisect_right(markers_pos, pos) - 1
    if idx < 0:
        return ""
    name = markers_val[idx]
    return name or ""


def _step_index_from_pos(pos: int, geom_positions: List[int]) -> int:
    if not geom_positions:
        return 1
    idx = bisect.bisect_right(geom_positions, pos)
    if idx <= 0:
        return 1
    if idx > len(geom_positions):
        return len(geom_positions)
    return idx


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


def _parse_mixed_ci_block(text: str, start_pos: int) -> Dict[int, Dict[str, Any]]:
    segment = text[start_pos : start_pos + 4000]
    lines = segment.splitlines()
    idx = 0
    results: Dict[int, Dict[str, Any]] = {}

    while idx < len(lines):
        line = lines[idx]
        m = re.search(r"MIXED state nr\.\s+(\d+)", line)
        if not m:
            idx += 1
            continue
        root = int(m.group(1))
        idx += 1
        # Skip headers until the "Conf" line
        while idx < len(lines) and "Conf" not in lines[idx]:
            idx += 1
        if idx >= len(lines):
            break
        idx += 1
        configs = []
        while idx < len(lines):
            line = lines[idx]
            if not line.strip():
                break
            parts = line.split()
            if len(parts) < 5:
                break
            try:
                conf_idx = int(parts[0])
            except ValueError:
                break
            conf = " ".join(parts[1:-2])
            try:
                coeff = float(parts[-2])
                weight = float(parts[-1])
            except ValueError:
                break
            configs.append(dict(idx=conf_idx, conf=conf, coeff=coeff, weight=weight))
            idx += 1
        results[root] = dict(configs=configs)
        idx += 1
    return results


def build_ci_by_step(text: str, steps: List[Dict[str, Any]]) -> Dict[int, Dict[str, Dict[int, Dict[str, Any]]]]:
    """
    把 CI block 按几何步 + SubProject 分组：
      ci_by_step[step_index][subproject_key][root] = {configs, source}
    优先使用 CASPT2 (MIXED) 的组态信息，没有则回退到 CASSCF。
    """
    geom_positions = [s["pos"] for s in steps]
    markers_pos, markers_val = _build_subproject_markers(text)
    ci_by_step: Dict[int, Dict[str, Dict[int, Dict[str, Any]]]] = {}
    SOURCE_PRIORITY = {"CASSCF": 0, "CASPT2": 1}

    def store(step_idx: int, subproject_key: str, root: int, info: Dict[str, Any], source: str):
        bucket = ci_by_step.setdefault(step_idx, {}).setdefault(subproject_key, {})
        prev = bucket.get(root)
        if prev and SOURCE_PRIORITY.get(prev.get("source", "CASSCF"), 0) > SOURCE_PRIORITY.get(source, 0):
            return
        new_info = dict(info)
        new_info["source"] = source
        bucket[root] = new_info

    for m in re.finditer(
        r"printout of CI-coefficients larger than  0.05 for root\s+(\d+)", text
    ):
        pos = m.start()
        root = int(m.group(1))
        step_idx = _step_index_from_pos(pos, geom_positions)
        subproject_key = _subproject_for_pos(pos, markers_pos, markers_val)
        info = parse_ci_block(text, pos)
        store(step_idx, subproject_key, root, info, "CASSCF")

    for m in re.finditer(r"\+\+ Mixed CI coefficients:", text):
        pos = m.start()
        step_idx = _step_index_from_pos(pos, geom_positions)
        subproject_key = _subproject_for_pos(pos, markers_pos, markers_val)
        block = _parse_mixed_ci_block(text, pos)
        for root, info in block.items():
            store(step_idx, subproject_key, root, info, "CASPT2")

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


def parse_ediff_targets(text: str) -> List[int]:
    """解析 Constraints 中 Ediff 后面的态编号"""
    m = re.search(
        r"Constraints([\s\S]{0,500}?)End of constraints",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return []
    block = m.group(1)
    ediff = re.search(r"Ediff\s+([0-9\s]+)", block, flags=re.IGNORECASE)
    if not ediff:
        return []
    nums = re.findall(r"\d+", ediff.group(1))
    return [int(x) for x in nums]


def collect_state_energies(
    text: str, steps: List[Dict[str, Any]]
) -> Dict[int, Dict[str, Dict[int, Dict[str, Any]]]]:
    """
    收集每一步、每个 SubProject、每个 root 的能量，优先 CASPT2。
    energies[step][subproject][root] = {energy, method, raw_label}
    """
    geom_positions = [s["pos"] for s in steps]
    markers_pos, markers_val = _build_subproject_markers(text)
    energies: Dict[int, Dict[str, Dict[int, Dict[str, Any]]]] = {}

    METHOD_PRIORITY = {"CASSCF": 0, "CASPT2": 1}

    def store(pos: int, root: int, method: str, raw_label: str, energy: float):
        step_idx = _step_index_from_pos(pos, geom_positions)
        subproject_key = _subproject_for_pos(pos, markers_pos, markers_val)
        root_bucket = energies.setdefault(step_idx, {}).setdefault(subproject_key, {})
        prev = root_bucket.get(root)
        priority = METHOD_PRIORITY.get(method, 0)
        if prev and METHOD_PRIORITY.get(prev.get("method", "CASSCF"), 0) > priority:
            return
        root_bucket[root] = dict(energy=energy, method=method, raw_label=raw_label)

    for m in re.finditer(
        r"RASSCF root number\s+(\d+)\s+Total energy:\s+([+-]?\d+\.\d+)", text
    ):
        root = int(m.group(1))
        energy = float(m.group(2))
        store(m.start(), root, "CASSCF", "RASSCF", energy)

    for m in re.finditer(
        r"([A-Z0-9\-]+CASPT2)\s+Root\s+(\d+)\s+Total energy:\s+([+-]?\d+\.\d+)",
        text,
    ):
        raw = m.group(1)
        root = int(m.group(2))
        energy = float(m.group(3))
        store(m.start(), root, "CASPT2", raw, energy)

    return energies

