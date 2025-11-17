# kk_plot.py
# 负责绘图到 PNG：单张图，左轴能量(线性)，右轴收敛量(log)
# 所有步骤的线都是完整曲线，仅用竖直半透明绿色条标示当前步。

from typing import List, Dict, Any


def make_plot(
    steps: List[Dict[str, Any]],
    step_idx: int,
    png_path: str,
    energy_tracks: List[Dict[str, Any]],
):
    """
    单张图：
      - 左轴：Energy (kcal/mol)，线性坐标，黑线，零点为所有几何步中最低能量
      - 右轴：Disp/Grad RMS/Max + 各自阈值（log 坐标）
      - 能量与收敛曲线 luôn 使用全部 step 的数据（完整曲线）

    特性：
      - 白色不透明背景
      - 对每一条收敛曲线分别判断收敛：
          未达到对应阈值: 小圆点
          达到对应阈值:   较小的方块
      - 收敛限虚线与主线同色，不加入图例
      - 不显示标题和 x 轴文字（只保留刻度）
      - 能量纵轴范围固定为 0 ~ 全程最大相对能量（略加余量）
      - 收敛纵轴范围在所有步/所有量上统一（略加余量），log 坐标
      - 横坐标左右留少量余量，避免端点点被裁
      - 当前步 step_idx 用竖直半透明绿色条标示
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import math

    n = len(steps)
    if n == 0:
        return

    # 保证 step_idx 合法
    if step_idx < 1:
        step_idx = 1
    if step_idx > n:
        step_idx = n

    # 全部步的横坐标
    x_all = list(range(1, n + 1))

    # ----- 能量：kcal/mol，相对所有轨道最低能量 -----
    H2KCAL = 627.509474
    if not energy_tracks:
        energy_tracks = [dict(name="Energy", values=[s["energy"] for s in steps])]

    all_vals = [v for track in energy_tracks for v in track["values"] if v is not None]
    if all_vals:
        e_min = min(all_vals)
    else:
        e_min = min(s["energy"] for s in steps)

    track_curves = []
    max_e_rel = 0.0
    for track in energy_tracks:
        rel_vals = []
        for v in track["values"]:
            if v is None:
                rel_vals.append(float("nan"))
            else:
                rel = (v - e_min) * H2KCAL
                rel_vals.append(rel)
                if rel > max_e_rel:
                    max_e_rel = rel
        track_curves.append(dict(name=track.get("name", "Energy"), values=rel_vals))

    if max_e_rel <= 0.0:
        max_e_rel = 1.0
    # 稍微留一点上边距
    y_en_min = -0.3
    y_en_max = max_e_rel * 1.05

    # ----- 收敛量：全部步 -----
    disp_rms_all = [s["disp_rms"] for s in steps]
    disp_max_all = [s["disp_max"] for s in steps]
    grad_rms_all = [s["grad_rms"] for s in steps]
    grad_max_all = [s["grad_max_v"] for s in steps]

    def safe(arr):
        return [v if (v is not None and v > 0.0) else float("nan") for v in arr]

    y_disp_rms_all = safe(disp_rms_all)
    y_disp_max_all = safe(disp_max_all)
    y_grad_rms_all = safe(grad_rms_all)
    y_grad_max_all = safe(grad_max_all)

    # 阈值取最后一步的
    t = steps[-1]
    disp_thr = t.get("disp_thr")
    disp_max_thr = t.get("disp_max_thr")
    grad_thr = t.get("grad_thr")
    grad_max_thr = t.get("grad_max_thr")

    # 收敛量全局 ylim 统一（包含所有点和阈值）
    conv_vals = []
    for arr in (y_disp_rms_all, y_disp_max_all, y_grad_rms_all, y_grad_max_all):
        for v in arr:
            if isinstance(v, float) and v > 0.0 and not math.isnan(v):
                conv_vals.append(v)
    for thr in (disp_thr, disp_max_thr, grad_thr, grad_max_thr):
        if thr is not None and thr > 0.0:
            conv_vals.append(thr)

    if conv_vals:
        ymin = min(conv_vals)
        ymax = max(conv_vals)
        # log 轴上稍微加一点 margin
        y_conv_min = ymin * 0.7 if ymin > 0 else 1e-8
        y_conv_max = ymax * 1.3
    else:
        # fallback
        y_conv_min = 1e-8
        y_conv_max = 1.0

    # 配色：结构一套，梯度一套
    color_disp_rms = "#1f77b4"  # 蓝
    color_disp_max = "#6baed6"  # 淡蓝
    color_grad_rms = "#d62728"  # 红
    color_grad_max = "#ff9896"  # 淡红

    # 单张图
    fig, ax_en = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("white")
    ax_en.set_facecolor("white")

    # ===== 当前步竖直半透明绿色条（打在最底层）=====
    # 使用 axvspan 会自动映射到 twin 右轴（同一 x）
    ax_en.axvspan(
        step_idx - 0.5, step_idx + 0.5,
        facecolor="green",
        alpha=0.15,
        zorder=0.1
    )

    # ===== 左轴：能量（kcal/mol），完整曲线，黑线 + 菱形点 =====
    energy_colors = ["black", "#9467bd", "#2ca02c", "#ff7f0e", "#8c564b"]
    energy_markers = ["D", "o", "s", "^", "v"]
    energy_handles = []
    for idx_track, track in enumerate(track_curves):
        color = energy_colors[idx_track % len(energy_colors)]
        marker = energy_markers[idx_track % len(energy_markers)]
        handle, = ax_en.plot(
            x_all,
            track["values"],
            color=color,
            marker=marker,
            markersize=6,
            linewidth=2.0,
            label=track["name"],
        )
        energy_handles.append(handle)

    ax_en.set_ylabel("Energy (kcal/mol)")
    ax_en.grid(True, linestyle=":")
    # x 轴左右留一点余量
    ax_en.set_xlim(0.8, n + 0.2)
    # 能量 y 轴固定
    ax_en.set_ylim(y_en_min, y_en_max)

    # ===== 右轴：收敛量（log），完整曲线 =====
    ax_conv = ax_en.twinx()
    ax_conv.set_facecolor("white")

    # 画主线（无 marker）
    line_disp_rms, = ax_conv.plot(
        x_all, y_disp_rms_all, color=color_disp_rms, linewidth=1, label="Disp RMS"
    )
    line_disp_max, = ax_conv.plot(
        x_all, y_disp_max_all, color=color_disp_max, linewidth=1, label="Disp Max"
    )
    line_grad_rms, = ax_conv.plot(
        x_all, y_grad_rms_all, color=color_grad_rms, linewidth=1, label="Grad RMS"
    )
    line_grad_max, = ax_conv.plot(
        x_all, y_grad_max_all, color=color_grad_max, linewidth=1, label="Grad Max"
    )

    # 阈值线：与对应线同色、虚线，不入图例
    if disp_thr is not None:
        ax_conv.axhline(
            disp_thr, color=color_disp_rms,
            linewidth=0.8, linestyle="--"
        )
    if disp_max_thr is not None:
        ax_conv.axhline(
            disp_max_thr, color=color_disp_max,
            linewidth=0.8, linestyle="--"
        )
    if grad_thr is not None:
        ax_conv.axhline(
            grad_thr, color=color_grad_rms,
            linewidth=0.8, linestyle="--"
        )
    if grad_max_thr is not None:
        ax_conv.axhline(
            grad_max_thr, color=color_grad_max,
            linewidth=0.8, linestyle="--"
        )

    # 工具函数：根据本条曲线的阈值分成“小圆点/较小方块”
    def split_by_threshold(xvals, yvals, thr):
        xs_small, ys_small, xs_big, ys_big = [], [], [], []
        if thr is None:
            for xi, yv in zip(xvals, yvals):
                if not (isinstance(yv, float) and yv > 0.0 and not math.isnan(yv)):
                    continue
                xs_small.append(xi)
                ys_small.append(yv)
            return xs_small, ys_small, xs_big, ys_big
        for xi, yv in zip(xvals, yvals):
            if not (isinstance(yv, float) and yv > 0.0 and not math.isnan(yv)):
                continue
            if yv <= thr:
                xs_big.append(xi)
                ys_big.append(yv)
            else:
                xs_small.append(xi)
                ys_small.append(yv)
        return xs_small, ys_small, xs_big, ys_big

    # 为每条曲线画小圆点/较小方块
    for yvals, color, thr in [
        (y_disp_rms_all, color_disp_rms, disp_thr),
        (y_disp_max_all, color_disp_max, disp_max_thr),
        (y_grad_rms_all, color_grad_rms, grad_thr),
        (y_grad_max_all, color_grad_max, grad_max_thr),
    ]:
        xs_small, ys_small, xs_big, ys_big = split_by_threshold(x_all, yvals, thr)
        if xs_small:
            ax_conv.scatter(xs_small, ys_small, s=10, color=color, marker="o")
        if xs_big:
            ax_conv.scatter(xs_big, ys_big, s=25, color=color, marker="s")  # 小方块

    ax_conv.set_yscale("log")
    ax_conv.set_ylabel("Convergence (log)")
    ax_conv.set_ylim(y_conv_min, y_conv_max)
    ax_conv.grid(False)

    # 图例只含 4 条主线，放右上
    ax_conv.legend(fontsize=7, loc="upper right")

    if len(energy_handles) > 1:
        ax_en.legend(handles=energy_handles, loc="upper left", fontsize=7)

    fig.tight_layout()
    fig.savefig(
        png_path,
        dpi=150,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )
    plt.close(fig)

