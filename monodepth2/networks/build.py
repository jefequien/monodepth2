from .resnet_encoder import ResnetEncoder
from .depth_decoder import DepthDecoder
from .pose_decoder import PoseDecoder

def build_models(cfg):
    depth_encoder = ResnetEncoder(cfg.MODEL.NUM_LAYERS, pretrained=True)
    depth_decoder = DepthDecoder(depth_encoder.num_ch_enc, cfg.MODEL.SCALES)

    pose_encoder = ResnetEncoder(cfg.MODEL.NUM_LAYERS, pretrained=True, num_input_images=2)
    pose_decoder = PoseDecoder(
                    pose_encoder.num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)


    models = {}
    models["depth_encoder"] = depth_encoder
    models["depth_decoder"] = depth_decoder
    models["pose_encoder"] = pose_encoder
    models["pose_decoder"] = pose_decoder
    return models