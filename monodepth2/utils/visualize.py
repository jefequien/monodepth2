import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

def vis_depth(depth, vmin=None, vmax=None):
    disp = depth.copy()
    disp[depth != 0] = 1 / depth[depth != 0]
    return vis_disp(disp, vmin=vmin, vmax=vmax)

def vis_disp(disp, vmin=None, vmax=None):
    if vmin is None:
        vmin = disp.min()
    if vmax is None:
        vmax = np.percentile(disp, 95)
        
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(disp)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)
    return im
