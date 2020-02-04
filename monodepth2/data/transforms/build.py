from . import transforms as T

def build_transforms(cfg, is_train=True):
    """Returns tranform that creates a single training item from data.

    Values correspond to torch tensors.
    Keys in the dictionary are either strings or tuples:

        ("color",     <frame_id>, <scale>)      for raw colour images,
        ("color_aug", <frame_id>, <scale>)      for augmented colour images,
        ("K",         <frame_id>, <scale>)      for camera intrinsics,
        ("inv_K",     <frame_id>, <scale>)      for camera intrinsics inverted,
        ("ext_T"      <frame_id>, <scale>)      for camera extrinsics,

    <frame_id> is either:
        an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
    or
        "s" for the opposite image in the stereo pair.

    <scale> is an integer representing the scale of the image relative to the fullsize image:
        -1      images at native resolution as loaded from disk
        0       images resized to (self.width,      self.height     )
        1       images resized to (self.width // 2, self.height // 2)
        2       images resized to (self.width // 4, self.height // 4)
        3       images resized to (self.width // 8, self.height // 8)
    """

    if is_train:
        width = cfg.INPUT.WIDTH
        height = cfg.INPUT.HEIGHT
        aux_ids = cfg.INPUT.AUX_IDS
        scales = cfg.INPUT.SCALES
        flip_horizontal_prob = 0.0
        brightness = cfg.INPUT.BRIGHTNESS
        contrast = cfg.INPUT.CONTRAST
        saturation = cfg.INPUT.SATURATION
        hue = cfg.INPUT.HUE
    else:
        width = cfg.INPUT.WIDTH
        height = cfg.INPUT.HEIGHT
        aux_ids = ['map']
        scales = [0]
        flip_horizontal_prob = 0.0
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    transform = T.Compose(
        [
            T.Resize(height, width),
            T.ColorJitter(brightness, contrast, saturation, hue),
            # T.RandomHorizontalFlip(flip_horizontal_prob),
            T.PrepareImageInputs(scales, height, width),
            T.PrepareCalibInputs(scales, height, width), 
            T.PrepareAuxInputs(aux_ids, scales, height, width), 
            T.ToTensorInputs(),
        ]
    )
    return transform