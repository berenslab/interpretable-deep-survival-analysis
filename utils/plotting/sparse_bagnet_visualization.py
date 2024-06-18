import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage import feature, transform
from sklearn.cluster import DBSCAN
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import cv2
import os
import sys

# Import from sibling directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + ".")

# Import from parent directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "..")

# Import from project directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "../..")

## Global settings
colorbar_labels = ["Lower\nRisk", "Higher\nRisk"]
activation_colors = ["green", "purple"]
bb_colors = ["green", "purple"]

def get_highest_activation_coords(heatmap, percentile=90):
    """
    Get the coordinates of the highest activations in the heatmap. Reverse sign of heatmap for 
    negative class. Among pos&neg values(!), the percentile will functions as a lower threshold of 
    values to consider.

    Args:
        heatmap (numpy.ndarray): The heatmap.
        percentile (float): Consider only values above this percentile for coordinates.

    Returns:
        numpy.ndarray: The coordinates of the highest activations.
    """
    abs_thresh = np.percentile(np.abs(heatmap), percentile)
    row_indices, col_indices = np.where(heatmap > abs_thresh)
    max_elements = heatmap[row_indices, col_indices]
    max_elements_order = np.argsort(max_elements)[::-1]
    row_indices, col_indices = row_indices[max_elements_order], col_indices[max_elements_order]

    coords = np.array((col_indices, row_indices)).T

    return coords
    
def get_topk_patches(heatmap, k=10, bb_size=60, allowed_overlap=20, percentile=90, absolute=False, return_activations=False):
    """
    Get the top k patches for the positive class based on the heatmap. For the negative class, reverse
    the sign of the heatmap. The patches are selected based on the highest activations in the heatmap. 
    However, if absolute is set to True, the patches are selected based on the highest absolute
    activations in the heatmap, i.e. both positive and negative activations are considered.

    Args:
        heatmap (numpy.ndarray): The heatmap.
        k (int): The maximum number of patches to retrieve.
        bb_size (int): The size of the bounding boxes.
        allowed_overlap (int): The allowed overlap between bounding boxes. Caveat: 
            This is only applied when using the "highres" method. When using "lowres", the overlap is
            set to 0, as an overlap of 1 pixel in low resolution space is too much in the upsampled space.
        percentile (float): Consider only values above this percentile for box coordinates.
        absolute (bool): Whether to consider absolute values of the heatmap, i.e. both pos and 
            neg activations, and track label assignment of patches.
        return_activations (bool): Whether to return the activations of the patches.

    Returns:
        tuple (list, list): The list of top k patches and the list of labels
            or tuple of lists of patches, labels and activations.
    """

    if absolute:
        heatmap_original = heatmap.copy()
        heatmap = np.abs(heatmap)

    # Activation coords sorted by activation (highest first)
    coords = get_highest_activation_coords(heatmap, percentile=percentile)

    # Activation coords as centers of rectangles
    offset = int(bb_size/2)
    bb_size = 2 * offset
    all_rectangles = [Rect(Point(x0-offset, y0-offset), Point(x0+offset, y0+offset)) for x0, y0 in coords]
    
    # Move rectangles into image boundaries, if necessary    
    all_rectangles = [r.move_into_image(heatmap.shape[1], heatmap.shape[0]) for r in all_rectangles]

    final_rectangles = []
    final_rectangles_labels = []
    final_activations = []
    remaining_rectangles = all_rectangles.copy()

    while len(final_rectangles) < k:
        # Stop if no more rectangles
        if len(remaining_rectangles) == 0:
            break
        
        # Get highest activation
        activations = []

        for i, rect in enumerate(remaining_rectangles):
            rect_activation = np.mean(heatmap[rect.top_left.y:rect.bottom_right.y, rect.top_left.x:rect.bottom_right.x])
            activations.append(rect_activation)

        # Get highest activation
        max_activation = np.max(activations)
        max_activation_id = np.argmax(activations)
        best_rect = remaining_rectangles[max_activation_id]

        # Prevent too much overlap
        overlaps = False
        for rect in final_rectangles:
            if rect.intersects(best_rect, allowed_overlap=allowed_overlap):
                overlaps = True
                # Remove best_rect from remaining_rectangles
                remaining_rectangles.pop(max_activation_id)
                break
        
        if not overlaps:
            # Add best_rect to final_rectangles
            final_rectangles.append(best_rect)
            final_activations.append(max_activation)
            if absolute:
                rect_label = np.sign(np.mean(heatmap_original[best_rect.top_left.y:best_rect.bottom_right.y, best_rect.top_left.x:best_rect.bottom_right.x]))
            else:
                rect_label = None
            final_rectangles_labels.append(rect_label)
            remaining_rectangles.pop(max_activation_id)

            # Delete all remaining rectangles that overlap largely with best_rect to speedup
            for i, rect in enumerate(remaining_rectangles):
                if rect.intersects(best_rect, allowed_overlap=int(bb_size*.95)): #bb_size-5
                    remaining_rectangles.pop(i)
    
    if return_activations:
        return final_rectangles, final_rectangles_labels, final_activations
    
    return final_rectangles, final_rectangles_labels
    


def get_ax_with_bb(ax, img, heatmap, k=10, bb_size=66, allowed_overlap=None, labels=None, method="lowres", return_patches=False, plot_img=True, labels_fontsize=11):
    """Get the matplotlib axis object with the image and bounding boxes.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axis to add the plot to.
        img (torch.Tensor or numpy.ndarray): The input image.
        heatmap (torch.Tensor or numpy.ndarray): The heatmap.
        k (int): The maximum number of bounding boxes to plot.
        bb_size (int): The size of the bounding boxes.
        allowed_overlap (int): The allowed overlap between bounding boxes. Caveat:
            This is only applied when using the "highres" method. When using "lowres", the overlap is
            set to 0, as an overlap of 1 pixel in low resolution space is too much in the upsampled space.
        label (int): The label of the class (0 for negative, 1 for positive).
        method (str): The method for patch selection ("lowres" or "highres").
        return_patches (bool): Whether to return the list of patches.
        plot_img (bool): Whether to plot the image.
        hires (bool): Whether the image is high resolution.

    Returns:
        matplotlib.axes.Axes or tuple: The matplotlib axis object or tuple of axis object and list of patches.
    """
    if allowed_overlap is None:
        # Overlap of 1 pixel in low resolution space is more than 1/3 in the upsampled space
        # Hence, don't allow any overlap when running on low dim.
        allowed_overlap = bb_size//3 if heatmap.shape[0] == img.shape[0] else 0

    # No need to handle batches
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        # remove batch dimension
        if img.shape[0] == 1:
            img = img[0]
        # to imshow format (H, W, C)
        img = img.transpose(1, 2, 0)

    if isinstance(heatmap, torch.Tensor):
        # remove batch dimension
        heatmap = heatmap.detach().cpu().numpy()[0]
    
    if labels is None or labels == [0,1]:
        labels = [-1,1]
    elif not isinstance(labels, (list, tuple, np.ndarray)):
        labels = -1 if labels == 0 else labels
        labels = [labels]

    # Reverse sign to plot negative activations in case of negative class (label=0)
    if len(labels) == 1:
        heatmap = -heatmap if labels[0] == -1 else heatmap
        combined = False
    else:
        combined = True

    ## On low res heatmap
    if method == "lowres":
        # Get top k patches
        # Downsample bb_size to heatmap dimensions
        downsample_factor = heatmap.shape[0] / img.shape[0]
        bb_size_lowres = int(bb_size * downsample_factor)
        overlap_lowres = int(allowed_overlap * downsample_factor)

        upsampling_factor = img.shape[0] / heatmap.shape[0]

        patches_lowres, patch_labels = get_topk_patches(heatmap, k=k, bb_size=bb_size_lowres, allowed_overlap=overlap_lowres, absolute = combined)

        # Upsample patches to imgs sizes, i.e. from heatmap dimensions to image dimensions
        def _upscale(c: tuple):
            def _upscale_scalar(c):
                return int(c * upsampling_factor)#+ upsampling_factor/2)
            return (_upscale_scalar(c[0]), _upscale_scalar(c[1]))
        
        center_coords = [r.center for r in patches_lowres]
        center_coords = [_upscale(c) for c in center_coords]

        # Create patches from center coords
        offset = bb_size/2
        patches = [Rect(Point(int(x0-offset), int(y0-offset)), Point(int(x0+offset), int(y0+offset))) for x0, y0 in center_coords]

        # Move rectangles into image boundaries, if necessary
        patches = [r.move_into_image(img.shape[1], img.shape[0]) for r in patches]

    elif method == "highres":
        heatmap = transform.resize(heatmap, (img.shape[0], img.shape[0]), anti_aliasing=True)
        patches, patch_labels = get_topk_patches(heatmap, k=k, bb_size=bb_size, allowed_overlap=allowed_overlap, absolute=combined)

    else:
        raise ValueError(f"unknown method: {method}")
    
    labels = patch_labels if combined else [labels[0]] * len(patches)

    # Plot image with bounding boxes
    if plot_img:
        ax.imshow(img)
    for i, patch in enumerate(patches):
        color = bb_colors[1] if labels[i] == 1 else bb_colors[0]
        
        lw = 0.7 # if labels[i] == 1 else 2 #1.5
        ax.add_patch(Rectangle((patch.top_left.x, patch.top_left.y), width = bb_size, height = bb_size, fill= False, linewidth=lw, edgecolor=color, facecolor='none'))

        # Add number
        ax.text(x=patch.top_left.x, y=patch.top_left.y-5, s=str(i+1), fontsize=labels_fontsize, c="black")
    ax.axis('off')

    if return_patches:
        return ax, patches, labels
    return ax

def plot_image_and_patches(img, heatmap, figsize=(15,11), k=6, bb_size=66, allowed_overlap=None, labels=[0,1], method="lowres", return_axs=False, heatmap_style=None, axs=None, labels_fontsize=11, labels_above_image=True, add_colormarker="auto"):
    """Plot the image with bounding boxes around the most important areas.

    Args:
        img (torch.Tensor or numpy.ndarray): The input image.
        heatmap (torch.Tensor or numpy.ndarray): The heatmap.
        figsize (tuple): The figure size.
        k (int): The maximum number of bounding boxes to plot.
        bb_size (int): The size of the bounding boxes.
        allowed_overlap (int): The allowed overlap between bounding boxes. Caveat:
            This is only applied when using the "highres" method. When using "lowres", the overlap is
            set to 0, as an overlap of 1 pixel in low resolution space is too much in the upsampled space.
        labels (list): The labels of the class (0 for negative, 1 for positive). Default is [0,1].
        method (str): The method for patch selection (on "lowres" or "highres" heatmap).
        return_ax (bool): Whether to return the matplotlib axis object.
        heatmap_style (str): The style of the heatmap ("area" or "raw"). None for no heatmap.

    Returns:
        matplotlib.axes.Axes or None: The matplotlib axis object or None.
    """
   
    if axs:
        ax0, axs1 = axs
    else:
        fig = plt.figure(figsize=figsize)
        figs = fig.subfigures(1, 2, width_ratios=[2,0.85])
        ax0 = figs[0].subplots(1, 1, gridspec_kw={'hspace': 0.025, 'wspace': 0.025})

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        # remove batch dimension
        if img.shape[0] == 1:
            img = img[0]
        # to imshow format (H, W, C)
        img = img.transpose(1, 2, 0)
    
    ax0.imshow(img)

    if heatmap_style:
        ax0 = plot_heatmap_on_axes(heatmap, extent=(0, img.shape[1], img.shape[0], 0), ax=ax0, percentile=99, with_legend=False, style=heatmap_style, only_legend=False)

    ax0, patches, labels = get_ax_with_bb(ax=ax0, img=img, heatmap=heatmap, k=k, labels=labels, bb_size=bb_size, allowed_overlap=allowed_overlap, return_patches=True, method=method, plot_img=False, labels_fontsize=labels_fontsize)    
    
    if axs is None:
        axs1 = figs[1].subplots(4, 2, gridspec_kw={'hspace': 0.025, 'wspace': 0.025})

    # Plot miniatures of patches
    for ip, ax in enumerate(axs1.flatten()):
        if ip >= len(patches):
            break
        # _img = img.numpy().transpose(1,2,0)
        patch = patches[ip]
        patch_img = img[patch.top_left.y:patch.bottom_right.y, patch.top_left.x:patch.bottom_right.x, :]
        patch_img = increase_contrast(patch_img)

        patch_img = add_border(patch_img, color="black", thickness=1)

        if labels_above_image:
            # Add 5px white space for number
            patch_img = np.pad(patch_img, ((6,0), (0,6), (0,0)), mode='constant', constant_values=1)
            y_offset = 8
        else: 
            y_offset = 16

        ax.imshow(patch_img)
        
        if add_colormarker in [True, "true", "auto"]:
            if len(np.unique(labels)) != 2:
                # Only one class present, determine which
                if isinstance(heatmap, torch.Tensor):
                    heatmap = heatmap.detach().cpu().numpy()[0]
                color = activation_colors[1] if np.sign(np.mean(heatmap)) == 1 else activation_colors[0]
            else:
                color = activation_colors[1] if labels[ip] == 1 else activation_colors[0]
        
            if labels_above_image:
                # Add colored rectangle to indicate class
                ax.add_patch(Rectangle((16,1), width = 7, height = 2, fill= True, linewidth=1, edgecolor=color, facecolor=color, alpha=0.8))
            else:
                # Add colored rectangle to indicate class
                ax.add_patch(Rectangle((16,4), width = 7, height = 2, fill= True, linewidth=1, edgecolor=color, facecolor=color, alpha=0.8))
        
        ax.text(1, y_offset, f"{ip+1}", color="black", fontsize=labels_fontsize)

    for ax in axs1.flatten():
        ax.axis("off")
    
    if return_axs:
        return ax0, axs1



def get_heatmap_on_edges(img, heatmap, ax=None, percentile=99, with_legend=False, style="area", only_legend=False, shrink_colorbar=0.93, figsize=(3.18,3), colorbar_with_transparency=True):
    """
    Get the heatmap overlaid on the edges of the image.

    Args:
        img (torch.Tensor or numpy.ndarray): The input image.
        heatmap (torch.Tensor or numpy.ndarray): The heatmap.
        ax (matplotlib.axes.Axes): The matplotlib axis to add the plot to.
        percentile (float): The percentile data range that the colormap covers.
        with_legend (bool): Whether to include a colorbar legend.
        style (str): The style of the heatmap ("area" or "raw").
        only_legend (bool): Whether to only show the colorbar legend.
        shrink_colorbar (float): The shrink factor for the colorbar legend.
        figsize (tuple): The figure size.

    Returns:
        matplotlib.axes.Axes: The matplotlib axis object.
    """
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        if img.shape[0] == 1:
            img = img[0]
        # # to imshow format (H, W, C)
        img = img.transpose(1, 2, 0)
    
    # Edges overlay
    y_channel = img[:, :, 1]
    y_channel = (y_channel - np.min(y_channel)) / (np.max(y_channel) - np.min(y_channel))
    channel = y_channel

    edges = feature.canny(channel, sigma=.01).astype(float)
    edges[edges < np.percentile(edges, 1)] = np.nan
    edges[:5, :] = np.nan
    edges[-5:, :] = np.nan
    edges[:, :5] = np.nan
    edges[:, -5:] = np.nan

    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()[0]

    extent = 0, img.shape[1], img.shape[0], 0

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.imshow(
        edges, 
        extent=extent, 
        interpolation='bilinear',
        cmap=plt.cm.Greys,
        alpha=1)

   
    ax = plot_heatmap_on_axes(heatmap, extent, ax=ax, percentile=percentile, with_legend=with_legend, style=style, alpha=.75, only_legend=only_legend, shrink_colorbar=shrink_colorbar, colorbar_with_transparency=colorbar_with_transparency)

    return ax


def plot_heatmap_on_edges(img, heatmap, ax=None, percentile=99, with_legend=False, style="area", figsize=(3.18,3), colorbar_with_transparency=True):
    """Plot the heatmap overlaid on the edges of the image.
    
    Args:
        img (torch.Tensor or numpy.ndarray): The input image.
        heatmap (torch.Tensor or numpy.ndarray): The heatmap.
        ax (matplotlib.axes.Axes): The matplotlib axis to add the plot to.
        percentile (float): The percentile data range that the colormap covers.
        with_legend (bool): Whether to include a colorbar legend.
        style (str): The style of the heatmap ("area" or "raw").
        figsize (tuple): The figure size.
    """
    get_heatmap_on_edges(img, heatmap, ax=ax, percentile=percentile, with_legend=with_legend, style=style, figsize=figsize, colorbar_with_transparency=colorbar_with_transparency)

def plot_image_and_heatmap_on_edges(img, heatmap, percentile=99, with_legend: bool = False, style="area", figsize=(6,3), colorbar_with_transparency=True, axs=None, return_axs=False):
    """Plot the image next to the heatmap overlaid on the edges.

    Args:
        img (torch.Tensor or numpy.ndarray): The input image.
        heatmap (torch.Tensor or numpy.ndarray): The heatmap.
        percentile (float): The percentile data range that the colormap covers.
        with_legend (bool): Whether to include a colorbar legend.
        style (str): The style of the heatmap ("area" or "raw").
        figsize (tuple): The figure size.
    """
    grid_ratio = [1,1] if not with_legend else [1,1] # 1,1 suits when using style.txt
    grid_spec = {'width_ratios': grid_ratio, 'height_ratios': [1], 'wspace': 0.05, 'hspace': 0.05}

    if axs is None:
        fig, axs = plt.subplots(1, len(grid_ratio), figsize=figsize, gridspec_kw=grid_spec, squeeze=True)

    kwargs = {}
    if with_legend:
        kwargs["shrink_colorbar"] = 0.85

    axs[1] = get_heatmap_on_edges(img, heatmap, ax=axs[1], percentile=percentile, with_legend=with_legend, style=style, colorbar_with_transparency=colorbar_with_transparency, **kwargs)
   
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        if img.shape[0] == 1:
            img = img[0]
        # to imshow format (H, W, C)
        img = img.transpose(1, 2, 0)

    axs[0].imshow(img)
    for ax in axs[:2]:
        ax.axis("off")
    
    if return_axs:
        return axs[0], axs[1]

def plot_heatmap_on_axes(heatmap, extent, ax=None, percentile=99, with_legend=False, style="area", alpha=.5, only_legend=False, shrink_colorbar=0.93, colorbar_with_transparency=True):
    """Plot the heatmap on the given matplotlib axis.

    Args:
        heatmap (torch.Tensor or numpy.ndarray): The heatmap.
        extent (tuple): The extent the heatmap should have in np coords, i.e. image dimensions 
            (0, img.shape[1], img.shape[0], 0) -> (left, right, bottom, top).
        ax (matplotlib.axes.Axes): The matplotlib axis to add the plot to.
        percentile (float): The percentile data range that the colormap covers.
        with_legend (bool): Whether to include a colorbar legend.
        style (str): The style of the heatmap ("area" or "raw").
        alpha (float): The alpha value for the heatmap.
        only_legend (bool): Whether to only show the colorbar legend.
        shrink_colorbar (float): The shrink factor for the colorbar legend.

    Returns:
        matplotlib.axes.Axes: The matplotlib axis object.
    """
    if ax is None:
        ax = plt.gca()
    if only_legend:
        l = ax
        fig, ax = plt.subplots(1, 1, figsize=(1, 5))

    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()[0]
    
    abs_max = np.percentile(np.abs(heatmap), percentile)
    
    # Plot blue-to-transparent (negative) and transparent-to-red (positive) heatmaps separately
    if style == "area":
        # Colormap with transparency
        transp = np.concatenate(
            (
                plt.cm.binary(np.linspace(0.99, 1, 32))[:, :-1], 
                np.linspace(0, 0.01, 32).reshape(-1, 1)
            ), 
            axis=1
        )

        # Individual colormaps 
        if "blue" in activation_colors[0].lower():
            clr_blues = plt.cm.Blues(np.linspace(0.5, 1, 96))
        elif "green" in activation_colors[0].lower():
            clr_blues = plt.cm.Greens(np.linspace(0.5, 1, 96))
        else:
            raise ValueError("unknown color for colormap")
        if "red" in activation_colors[1].lower():
            clr_red = plt.cm.Reds(np.linspace(0.5, 1, 96))
        elif "purple" in activation_colors[1].lower():
            clr_red = plt.cm.Purples(np.linspace(0.5, 1, 96))
        else:
            raise ValueError("unknown color for colormap")

        # Individual colormaps with transparency
        clrs_bl_t = np.vstack((transp, clr_blues))[::-1]
        clrs_rd_t = np.vstack((transp, clr_red))

        # Combine
        clrs_bl_t_rd = np.vstack((clrs_bl_t, clrs_rd_t))

        # Complete colormaps
        map_bl_t_rd = mcolors.LinearSegmentedColormap.from_list('map_bl_t_rd', clrs_bl_t_rd)

        # Plot heatmap
        overlay_ax = ax.imshow(
            heatmap, 
            extent=extent, 
            interpolation='bilinear', 
            cmap=map_bl_t_rd,
            vmin=-abs_max,
            vmax=abs_max,
            alpha=alpha
        )

        if with_legend:          
            # Plot a colorbar from blue (neg) to red (pos)
            # keep original unit/scale and a symmetric range
            if colorbar_with_transparency:
                # Fake ax for colormesh 
                _ax = plt.gcf().add_axes([0, 0, 0, 0])
                cb = plt.colorbar(
                    _ax.pcolormesh(
                        heatmap, 
                        cmap=map_bl_t_rd,
                        norm=plt.Normalize(vmin=-abs_max, vmax=abs_max),                        
                    ), 
                ax=ax if not only_legend else l, shrink=shrink_colorbar, pad=0.01
                )
                
            else:
                cb = plt.colorbar(
                    plt.cm.ScalarMappable(
                        cmap=plt.cm.RdBu_r,
                        norm=plt.Normalize(
                            vmin=-abs_max,
                            vmax=abs_max
                        )
                    ),
                    ax=ax if not only_legend else l,
                    shrink=shrink_colorbar,
                    pad=0.01
                )
                        
    else:
        cmap = plt.cm.RdBu_r
        cmap.set_bad(alpha=0)     

        abs_max = np.percentile(np.abs(heatmap), percentile)  

        # Plot heatmap
        overlay_ax = ax.imshow(
            heatmap,
            extent=extent,
            cmap=cmap,
            interpolation='bilinear',
            vmin=-abs_max,
            vmax=abs_max,
            alpha=alpha
        )

        if with_legend:
            # Plot a colorbar from blue (neg) to red (pos)
            if colorbar_with_transparency:
                cb = plt.colorbar(overlay_ax, ax=ax if not only_legend else l, shrink=shrink_colorbar, pad=0.01)
            else:
                cb = plt.colorbar(
                    plt.cm.ScalarMappable(
                        cmap=cmap,
                        norm=plt.Normalize(
                            vmin=-abs_max,
                            vmax=abs_max
                        )
                    ),
                    ax=ax if not only_legend else l,
                    shrink=shrink_colorbar,
                    pad=0.01
                )

    if with_legend:
        if colorbar_labels is None:
            cb.set_ticks([])
        elif colorbar_labels == "scale":
            pass
        elif isinstance(colorbar_labels, list) and len(colorbar_labels) == 2:
            cb.set_ticks([-abs_max*0.85, abs_max*0.85])
            cb.set_ticklabels(colorbar_labels)
        
        cb.ax.tick_params(labelsize=6)
    
    plt.axis('off')

    if only_legend:
        # delete everything but l
        for a in fig.axes:
            if a != l:
                a.remove()

    return ax if not only_legend else l
    

def plot_heatmap_on_image(img, heatmap, ax=None, percentile=99, with_legend=False, style="area", figsize=(3.18,3), colorbar_with_transparency=True, shrink_colorbar = 0.93, alpha=0.5, return_ax=False, border_color=None, border_width=3):
    """Plot the heatmap overlaid on the image.

    Args:
        img (torch.Tensor or numpy.ndarray): The input image.
        heatmap (torch.Tensor or numpy.ndarray): The heatmap.
        ax (matplotlib.axes.Axes): The matplotlib axis to add the plot to.
        percentile (float): The percentile data range that the colormap covers.        
        with_legend (bool): Whether to include a colorbar legend.
        style (str): The style of the heatmap ("area" or "raw").
        figsize (tuple): The figure size.

    Returns:
        matplotlib.axes.Axes: The matplotlib axis object.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
        if img.shape[0] == 1:
            img = img[0]
        # to (H, W, C)
        img = img.transpose(1, 2, 0)

    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.detach().cpu().numpy()[0]
    
    if border_color and border_width > 0:
        img = add_border(img, color=border_color, thickness=border_width)

    extent = 0, img.shape[1], img.shape[0], 0
    ax.imshow(img, extent=extent, alpha=1)

    ax = plot_heatmap_on_axes(heatmap, extent, ax=ax, percentile=percentile, with_legend=with_legend, style=style, colorbar_with_transparency=colorbar_with_transparency, shrink_colorbar=shrink_colorbar, alpha=alpha)
    ax.axis('off')

    if return_ax:
        return ax

def add_border(img, color="red", thickness=3):
    img = img_to_cv2(img)
    color = np.array(mcolors.to_rgba(color)) * 255
    img = cv2.copyMakeBorder(img, thickness, thickness, thickness, thickness, cv2.BORDER_CONSTANT, value=color)
    return img_from_cv2(img)

def increase_contrast(img, method = 'clahe+contrast'):
    """Increase contrast of an image.

    Args:
        img (numpy.ndarray): The input image.
        method (str): The method for increasing contrast, either "clahe", "contrast" or "clahe+contrast".

    Returns:
        numpy.ndarray: The image with increased contrast.
    """
    if method is None:
        return img
    
    elif 'clahe' in method and 'contrast' in method:
        img1 = img_to_cv2(apply_historgram_equalization(img))
        img2 = img_to_cv2(apply_brightness_contrast(img))

        # Overlay img2 with lowered opacity on img1
        opacity = 0.5
        img1 = cv2.addWeighted(img2, opacity, img1, 1 - opacity, 0)

        return img_from_cv2(img1)
    
    elif any([m in method for m in ['clahe', 'histogram']]):
        img = apply_historgram_equalization(img)

    elif 'contrast' in method:
        # img = apply_alpha_contrast(img)
        img = apply_brightness_contrast(img)

    else:
        raise NotImplementedError('unknown method')
    
    return img

def img_to_cv2(img):
    return (img * 255).astype(np.uint8)

def img_from_cv2(img):
    return img.astype(np.float32) / 255

def apply_historgram_equalization(image, cliplimit=4.0, tilesize=8):
    """Apply histogram equalization to an image to increase contrast.

    Args:
        image (numpy.ndarray): The input image.
        cliplimit (float): The clip limit for contrast limiting.
        tilesize (int): The tile size for contrast limiting.
    
    Returns:
        numpy.ndarray: The image with increased contrast.
    """
    # Image to cv2 format
    image = img_to_cv2(image)

    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=(tilesize, tilesize))

    # Apply CLAHE to the grayscale image
    l = clahe.apply(l)

    # Stack the equalized channel to create an RGB image
    lab = cv2.merge([l,a,b])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return img_from_cv2(image)

def apply_brightness_contrast(input_img, brightness = 0, contrast = 20):
    """Apply brightness and contrast to an image.
    
    Args:
        input_img (numpy.ndarray): The input image.
        brightness (int): The brightness value.
        contrast (int): The contrast value.
        
    Returns:
        numpy.ndarray: The image with adjusted brightness and contrast.
    """
    # Image to cv2 format
    input_img = img_to_cv2(input_img)

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return img_from_cv2(buf)

class Point:
    """A point in 2D space.
    
    Args:
        xcoord (int): The x-coordinate.
        ycoord (int): The y-coordinate.
        
        Attributes:
        x (int): The x-coordinate.
        y (int): The y-coordinate.
    
    """
    def __init__(self, xcoord=0, ycoord=0):
        self.x = xcoord
        self.y = ycoord

class Rect:
    """A rectangle in 2D space.

    Args:
        top_left (Point): The top left corner of the rectangle.
        bottom_right (Point): The bottom right corner of the rectangle.
    
        Attributes:
        top_left (Point): The top left corner of the rectangle.
        bottom_right (Point): The bottom right corner of the rectangle.
        center (tuple): The center of the rectangle.
    """

    def __init__(self, top_left, bottom_right):
        self.top_left = top_left
        self.bottom_right = bottom_right
        self.center = ((top_left.x + bottom_right.x) / 2, (top_left.y + bottom_right.y) / 2)
    
    def intersects(self, other, allowed_overlap=20):
        box_1 = {"xmin": self.top_left.x, "xmax": self.bottom_right.x, "ymin": self.top_left.y, "ymax": self.bottom_right.y}
        box_2 = {"xmin": other.top_left.x, "xmax": other.bottom_right.x, "ymin": other.top_left.y, "ymax": other.bottom_right.y}
        return not (box_1["xmax"] < box_2["xmin"] + allowed_overlap
                or box_1["xmin"] > box_2["xmax"] - allowed_overlap
                or box_1["ymax"] < box_2["ymin"] + allowed_overlap
                or box_1["ymin"] > box_2["ymax"] - allowed_overlap)
    
    def clip(self, min_x, max_x, min_y, max_y):
        self.top_left.x = max(min_x, self.top_left.x)
        self.top_left.y = max(min_y, self.top_left.y)
        self.bottom_right.x = min(max_x, self.bottom_right.x)
        self.bottom_right.y = min(max_y, self.bottom_right.y)
        return self

    def move_into_image(self, max_x, max_y, margin=0):
        """Moves rectangle into image boundaries if necessary, keeping its size. 
        margin is the amount of pixels that should be kept free from the image border."""
        if self.top_left.x < 0 + margin:
            self.bottom_right.x += abs(self.top_left.x)
            self.top_left.x = 0 + margin
        if self.top_left.y < 0 + margin:
            self.bottom_right.y += abs(self.top_left.y)
            self.top_left.y = 0 + margin
        if self.bottom_right.x > max_x - margin:
            self.top_left.x -= self.bottom_right.x - max_x
            self.bottom_right.x = max_x - margin
        if self.bottom_right.y > max_y - margin:
            self.top_left.y -= self.bottom_right.y - max_y
            self.bottom_right.y = max_y - margin
        return self

    def cut_out(self, img):
        """Returns the part of the image that is covered by the rectangle.
        
        Args:
            img (numpy.ndarray): The image.
        
        Returns:
            numpy.ndarray: The part of the image that is covered by the rectangle.
        """
        return img[self.top_left.y:self.bottom_right.y, self.top_left.x:self.bottom_right.x]

