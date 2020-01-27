import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

def vis_depth(depth):
    # Saving colormapped depth image
    vmax = np.percentile(depth, 95)
    normalizer = mpl.colors.Normalize(vmin=depth.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
    im = pil.fromarray(colormapped_im)
    return im