import numpy as np

import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib.patches as patches

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIG_SIZE = 14


def plot_motif_sets(series, motif_sets, gt=None, dimension_names=None, show_representative=False, legend=True):
    if series.ndim == 1:
        series = np.expand_dims(series, axis=1)
    if dimension_names is None:
        dimension_names = [f"dim {d+1}" for d in range(series.shape[1])]

    n = len(series)
    fig, axs = plt.subplots(len(motif_sets) + 1, 1, figsize=(12, (len(motif_sets) + 1) * 2), sharex=True, sharey=True)

    if np.array(axs).ndim == 0:
        axs = [axs]

    # axs[0].set_prop_cycle(cycler(color=["tab:blue", "tab:green", "tab:red"]))
    axs[0].set_prop_cycle(cycler(color=[u'#00407a', u'#2ca02c', u'#c00000']))
    axs[0].plot(range(len(series)), series, lw=1.5)
    if legend:
        axs[0].legend(dimension_names, fontsize=SMALL_SIZE)
    axs[0].set_xlim((0, n))

    if gt is not None:
        plot_ground_truth_ax(axs[0], gt, n)

    for i, motif_set in enumerate(motif_sets):
        representative_motif = None
        if type(motif_set) is tuple:
            representative_motif = motif_set[0]
            motif_set = motif_set[1]
        if motif_set is None: # not found
            axs[i+1].set_title(f"Motif Set {i+1} not discovered", fontsize=BIG_SIZE)
            continue
        axs[i+1].set_title(f"Motif Set {i+1}, k: {len(motif_set)}", fontsize=BIG_SIZE)
        for j, (s_m, e_m) in enumerate(motif_set):
            axs[i+1].set_prop_cycle(cycler(color=[u'#00407a', u'#2ca02c', u'#c00000']))
            # axs[i+1].set_prop_cycle(None)
            axs[i+1].plot(range(s_m, e_m), series[s_m : e_m, :], alpha=1, lw=1.5)
            axs[i+1].axvline(x=s_m, c='k', linestyle=':', lw=0.25)
            axs[i+1].axvline(x=e_m, c='k', linestyle=':', lw=0.25)
            if show_representative and ( (representative_motif is None and j==0) or (s_m, e_m) == representative_motif ):
                axs[i+1].axvspan(s_m, e_m, alpha=0.3, color='gray')
                text_x = (s_m + ((e_m - s_m) // 2)) / float(n)
                text_y = 0.90
                axs[i+1].text(text_x, text_y, r'$ \alpha $', horizontalalignment='center', verticalalignment='center', transform=axs[i+1].transAxes, fontsize=MEDIUM_SIZE)

    plt.tight_layout()
    return fig, axs

def plot_ground_truth_ax(ax, gt, n):
    for key in gt.keys():
        for (s, e) in gt[key]:
            ax.axvline(x=s, c='k', linestyle=':', lw=0.25)
            ax.axvline(x=e, c='k', linestyle=':', lw=0.25)
            
            text_x = (s + ((e - s) // 2)) / float(n)
            text_y = 0.90
            ax.text(text_x, text_y, str(key), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=MEDIUM_SIZE)
    plt.tight_layout()
    return ax

def plot_segment_bars(series, ax, motif_sets):
    n = len(series)
    for i, motif_set in enumerate(motif_sets):
        color = plt.cm.tab10(i % 10)
        for s_m, e_m in motif_set:
            ax.axvspan(s_m, e_m, alpha=0.3, color=color)
            ax.axvline(x=s_m, linestyle='-', lw=1, color=color)
            ax.axvline(x=e_m, linestyle='-', lw=1, color=color)
    ax.set_xlim((0, n))
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_facecolor("lightgray")
    return ax

def plot_segment_bars_multiple_lines(series, ax, motif_sets, show_representative=False):
    n = len(series)
    m = len(motif_sets)
    for i, motif_set in enumerate(motif_sets):
        representative_motif = None
        if type(motif_set) is tuple:
            representative_motif = motif_set[0]
            motif_set = motif_set[1]
        color = plt.cm.tab10(i % 10)
        if i != m-1:
            ax.axhline(y=i+1.5, linestyle='-', lw=1, color='gray', alpha=0.5)
        if motif_set is None: # not found
            ax.text(n/2, i+1, f"—", horizontalalignment='center', verticalalignment='center', fontsize=SMALL_SIZE)
            continue
        for j, (s_m, e_m) in enumerate(motif_set):
            ax.add_patch(patches.Rectangle((s_m, i+0.5), e_m-s_m, 1, alpha=0.5, facecolor=color, edgecolor='black'))
            if show_representative and ( (representative_motif is None and j==0) or (s_m, e_m) == representative_motif ):
                ax.plot((s_m+e_m)/2, i+1, '.', color=color)
    ax.set_xlim((0, n))
    ax.set_ylim((0.5, m+0.5))
    ax.invert_yaxis()
    # ax.set_yticklabels([])
    # ax.set_yticks([])
    ax.set_yticks([1, m])
    # ax.set_ylabel('motif sets')
    ax.set_facecolor("lightgray")
    return ax
    #TODO

def plot_sm(s1, s2, sm, path=None, figsize=(5, 5), colorbar=False, matshow_kwargs=None, ts_kwargs={'linewidth':1.5, 'ls':'-'}):
    from matplotlib import gridspec
    from cycler import cycler

    width_ratios = [0.9, 5]
    if colorbar:
        height_ratios=[0.8, 5, 0.15]
    else:
        height_ratios = width_ratios

    fig = plt.figure(figsize=figsize, frameon=True)
    gs = gridspec.GridSpec(2 + colorbar, 2, wspace=5, hspace=5,
                           height_ratios=height_ratios,
                           width_ratios=width_ratios)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_axis_off()

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_prop_cycle(None)
    ax1.set_axis_off()
    ax1.plot(range(len(s2)), s2, **ts_kwargs)
    ax1.set_xlim([-0.5, len(s2) - 0.5])

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_prop_cycle(None)
    ax2.set_axis_off()
    ax2.plot(-s1, range(len(s1), 0, -1), **ts_kwargs)
    ax2.set_ylim([0.5, len(s1) + 0.5])

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_aspect(1)
    ax3.tick_params(axis='both', which='both', labeltop=False, labelleft=False, labelright=False, labelbottom=False)

    
    kwargs = {} if matshow_kwargs is None else matshow_kwargs
    img = ax3.matshow(sm, **kwargs)
    
    cax = None
    if colorbar:
        cax = fig.add_subplot(gs[2, 1])
        fig.colorbar(img, cax=cax, orientation='horizontal')

    gs.tight_layout(fig)

    # Align the subplots:
    ax1pos = ax1.get_position().bounds
    ax2pos = ax2.get_position().bounds
    ax3pos = ax3.get_position().bounds
    ax2.set_position((ax2pos[0], ax2pos[1] + ax2pos[3] - ax3pos[3], ax2pos[2], ax3pos[3])) # adjust the time series on the left vertically
    if len(s1) < len(s2):
        ax3.set_position((ax3pos[0], ax2pos[1] + ax2pos[3] - ax3pos[3], ax3pos[2], ax3pos[3])) # move the time series on the left and the distance matrix upwards
    if len(s1) > len(s2):
        ax3.set_position((ax1pos[0], ax3pos[1], ax3pos[2], ax3pos[3])) # move the time series at the top and the distance matrix to the left
        ax1.set_position((ax1pos[0], ax1pos[1], ax3pos[2], ax1pos[3])) # adjust the time series at the top horizontally
    if len(s1) == len(s2):
        ax1.set_position((ax3pos[0], ax1pos[1], ax3pos[2], ax1pos[3])) # adjust the time series at the top horizontally
    
    ax = fig.axes
    return fig, ax, cax

def plot_local_warping_paths(axs, paths, **kwargs):
    for p in paths:
        axs[3].plot(p[:, 1], p[:, 0], 'r', **kwargs)
    return axs