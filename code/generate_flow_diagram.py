"""Generate the forecast-method schematic used as Figure 1."""

from pathlib import Path
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = Path(os.getenv('HUXT_FIGURE_DIR', ROOT / 'figures'))


def box(ax, xy, width, height, text, color):
    patch = FancyBboxPatch(
        xy, width, height, boxstyle='round,pad=0.025',
        facecolor=color, edgecolor='0.25', linewidth=1.5,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text,
            ha='center', va='center', fontsize=10)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=16,
                                linewidth=1.6, color='0.25'))


def main():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis('off')

    box(ax, (0.3, 3.0), 2.2, 1.1, 'Previous 27.27 days\nof hourly OMNI $V$', '#dceefa')
    box(ax, (3.1, 3.0), 2.2, 1.1, 'Remove ICMEs and\ninterpolate gaps', '#e8e8e8')
    box(ax, (5.9, 3.0), 2.2, 1.1, 'Corotate observations\ninto longitude', '#dceefa')
    box(ax, (8.7, 3.0), 2.8, 1.1, 'Map from 1 AU to 0.1 AU\nand apply CNN correction', '#dceefa')
    box(ax, (4.5, 0.8), 3.0, 1.1, 'HUXt propagation\nfrom 0.1 to 1 AU', '#dff1df')
    box(ax, (8.7, 0.8), 2.8, 1.1, '1- to 27-day solar-wind\nforecast at Earth', '#f9e2c7')

    arrow(ax, (2.5, 3.55), (3.1, 3.55))
    arrow(ax, (5.3, 3.55), (5.9, 3.55))
    arrow(ax, (8.1, 3.55), (8.7, 3.55))
    arrow(ax, (10.1, 3.0), (7.2, 1.9))
    arrow(ax, (7.5, 1.35), (8.7, 1.35))

    ax.text(6, 4.65, 'InSitu-HUXt forecast construction', ha='center',
            va='center', fontsize=15, fontweight='bold')
    OUTDIR.mkdir(parents=True, exist_ok=True)
    output = OUTDIR / 'flow_diagram.pdf'
    fig.savefig(output, bbox_inches='tight')
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
