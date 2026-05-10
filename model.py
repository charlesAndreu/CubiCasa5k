import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from floortrans.models import hg_furukawa_original

from dataloader import n_segmentation_classes


class CubiCasa5KUnet(smp.Unet):

    def __init__(self, args, logger):
        segmentation_map = args.segmentation_map
        n_output_channels = n_segmentation_classes(segmentation_map)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_type = args.model
        assert isinstance(model_type, str) and model_type.startswith("unet")

        logger.info(
            f"Using {model_type} model with {n_output_channels} channels for {segmentation_map} segmentation map"
        )
        super().__init__(
            encoder_name=model_type.split("-")[1],
            encoder_weights="imagenet",
            in_channels=3,
            classes=n_output_channels,
        )
        self.to(device)
        logger.info("Unet model loaded")


class CubiCasa5KFurukawa(hg_furukawa_original):

    def __init__(self, args, logger):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        segmentation_map = args.segmentation_map
        self.n_out = n_segmentation_classes(segmentation_map)

        logger.info("No model specified, using furukawa model")
        logger.info(
            f"Using furukawa model with {self.n_out} channels for {segmentation_map} segmentation map"
        )
        super().__init__(n_heatmap_channels=0, n_output_channels=51)

        resume = bool(args.resume_from)
        if not resume:
            self.init_weights()
            if args.furukawa_weights:
                logger.info(
                    "Loading furukawa model weights from checkpoint '{}'".format(
                        args.furukawa_weights
                    )
                )
                checkpoint = torch.load(args.furukawa_weights, map_location=device)
                self.load_state_dict(checkpoint["model_state"])
        else:
            logger.info(
                "Skipping init_weights / --furukawa-weights; will load full state from --resume-from"
            )

        self.conv4_ = nn.Conv2d(256, self.n_out, bias=True, kernel_size=1)
        self.upsample = nn.ConvTranspose2d(
            self.n_out, self.n_out, kernel_size=4, stride=4
        )
        if not resume:
            for m in [self.conv4_, self.upsample]:
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

        self.n_output_channels = self.n_out

        if resume:
            logger.info("Resuming model weights from checkpoint '%s'", args.resume_from)
            checkpoint = torch.load(args.resume_from, map_location=device)
            self.load_state_dict(checkpoint["model_state"])

        self.to(device)


def cubi_casa5k_model(args, logger):
    _m = args.model
    if isinstance(_m, str) and _m.startswith("unet"):
        return CubiCasa5KUnet(args, logger)
    return CubiCasa5KFurukawa(args, logger)

