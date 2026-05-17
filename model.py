import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from floortrans.models import hg_furukawa_original

from dataloader import n_segmentation_classes
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


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
        logger.info("No model specified, using furukawa model")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        segmentation_map = getattr(args, "segmentation_map", None)
        if segmentation_map is None:
            # train_full: 21 heatmap channels (sigmoid) + 4 room logits + 4 icon logits
            self.n_out = 21 + 4 + 4
            n_heatmap_channels = 21
            logger.info(
                f"Using furukawa model with {self.n_out} channels for heatmaps + room + icon"
                f" (n_heatmap_channels={n_heatmap_channels}, sigmoid on heatmap logits)"
            )
        else:
            # train_simple: single segmentation head, all channels are class logits
            # --> no sigmoid (cross-entropy is applied on raw logits)
            self.n_out = n_segmentation_classes(segmentation_map)
            n_heatmap_channels = 0
            logger.info(
                f"Using furukawa model with {self.n_out} channels for {segmentation_map} segmentation map"
                f" (n_heatmap_channels={n_heatmap_channels})"
            )
        super().__init__(
            n_heatmap_channels=n_heatmap_channels, n_output_channels=51
        )

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


class CubiCasa5KSegFormer(nn.Module):

    def __init__(self, args, logger):
        super().__init__()
        model_name = getattr(
            args,
            "segformer_model_name",
            "nvidia/segformer-b0-finetuned-ade-512-512",
        )
        segmentation_map = args.segmentation_map
        n_labels = n_segmentation_classes(segmentation_map)

        logger.info(
            "Loading SegFormer %s with num_labels=%d (%s)",
            model_name,
            n_labels,
            segmentation_map,
        )

        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=n_labels,
            ignore_mismatched_sizes=True,
        )

        mean = torch.tensor(self.processor.image_mean, dtype=torch.float32).view(
            1, 3, 1, 1
        )
        std = torch.tensor(self.processor.image_std, dtype=torch.float32).view(
            1, 3, 1, 1
        )
        self.register_buffer("_mean", mean)
        self.register_buffer("_std", std)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (N, 3, H, W), [-1, 1] from cubicasa augmentations
        x01 = (images + 1.0) * 0.5
        x01 = x01.clamp(0.0, 1.0)
        pixel_values = (x01 - self._mean) / (self._std + 1e-8)

        out = self.segformer(pixel_values=pixel_values)
        logits = out.logits

        if logits.shape[-2:] != images.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=images.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return logits


def cubi_casa5k_simple_model(args, logger):
    _m = args.model
    if isinstance(_m, str) and _m.lower() == "segformer":
        return CubiCasa5KSegFormer(args, logger)
    if isinstance(_m, str) and _m.startswith("unet"):
        return CubiCasa5KUnet(args, logger)
    return CubiCasa5KFurukawa(args, logger)


def cubi_casa5k_full_model(args, logger):
    return CubiCasa5KFurukawa(args, logger)
