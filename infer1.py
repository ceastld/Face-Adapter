import os
import sys
import cv2
import random
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.utils as ttf
import torchvision.transforms as transforms
from accelerate.utils import set_seed
from transformers import CLIPVisionModel, CLIPImageProcessor
from diffusers import ControlNetModel, UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler

import data.datasets_faceswap as datasets_faceswap
import face_adapter.model_seg_unet as model_seg_unet
from face_adapter.model_to_token import Image2Token, ID2Token
from face_adapter_pipline import StableDiffusionFaceAdapterPipeline, draw_pts70_batch

from insightface.app import FaceAnalysis
from third_party import model_parsing
import third_party.model_resnet_d3dfr as model_resnet_d3dfr
import third_party.d3dfr.bfm as bfm
import third_party.insightface_backbone_conv as model_insightface_backbone


class Config:
    def __init__(self):
        self.checkpoint = "./checkpoints"
        self.output = "./output"
        self.source = "./example/src"
        self.target = "./example/tgt"
        self.crop_ratio = 0.81
        self.cache_dir = "./hub"
        self.base_model = "runwayml/stable-diffusion-v1-5"
        self.use_cache = False


class ModelManager:
    def __init__(self, config):
        self.config = config
        self.device = "cuda"
        self.weight_dtype = torch.float16
        self.load_models()

    def load_models(self):
        self.controlnet = ControlNetModel.from_pretrained(os.path.join(self.config.checkpoint, "controlnet"), torch_dtype=self.weight_dtype).to(
            self.device
        )
        self.pipe = StableDiffusionFaceAdapterPipeline.from_pretrained(
            self.config.base_model,
            controlnet=self.controlnet,
            torch_dtype=self.weight_dtype,
            cache_dir=self.config.cache_dir if self.config.use_cache else None,
            local_files_only=self.config.use_cache,
            requires_safety_checker=False,
        ).to(self.device)

        # Load other models similarly...
        self.net_d3dfr = (
            model_resnet_d3dfr.getd3dfr_res50(os.path.join(self.config.checkpoint, "third_party/d3dfr_res50_nofc.pth")).eval().to(self.device)
        )
        self.bfm_facemodel = bfm.BFM(
            focal=1015 * 256 / 224, image_size=256, bfm_model_path=os.path.join(self.config.checkpoint, "third_party/BFM_model_front.mat")
        ).to(self.device)
        self.net_arcface = model_insightface_backbone.getarcface(os.path.join(self.config.checkpoint, "third_party/insightface_glint360k.pth")).to(
            self.device
        )
        self.clip_image_processor = CLIPImageProcessor()
        self.net_vision_encoder = CLIPVisionModel.from_pretrained(os.path.join(self.config.checkpoint, "vision_encoder")).to(self.device)
        self.net_image2token = Image2Token(
            visual_hidden_size=self.net_vision_encoder.vision_model.config.hidden_size, text_hidden_size=768, max_length=77, num_layers=3
        ).to(self.device)
        self.net_id2token = ID2Token(id_dim=512, text_hidden_size=768, max_length=77, num_layers=3).to(self.device)
        self.net_seg_res18 = model_seg_unet.UNet().eval().to(self.device)
        self.app = FaceAnalysis(
            name="antelopev2", root=os.path.join(self.config.checkpoint, "third_party"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))


class ImageProcessor:
    def __init__(self, config: Config):
        self.config: Config = config
        self.pil2tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.5, std=0.5)])

    def load_image(self, image_path):
        return Image.open(image_path).convert("RGB")

    def preprocess_image(self, image_pil, app):
        face_info = app.get(cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR))
        face_info = sorted(face_info, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * x["bbox"][3] - x["bbox"][1])[-1]
        dets = face_info["bbox"]

        if self.config.crop_ratio > 0:
            bbox = dets[0:4]
            bbox_size = max(bbox[2] - bbox[0], bbox[2] - bbox[0])
            bbox_x = 0.5 * (bbox[2] + bbox[0])
            bbox_y = 0.5 * (bbox[3] + bbox[1])
            x1 = bbox_x - bbox_size * self.config.crop_ratio
            x2 = bbox_x + bbox_size * self.config.crop_ratio
            y1 = bbox_y - bbox_size * self.config.crop_ratio
            y2 = bbox_y + bbox_size * self.config.crop_ratio
            bbox_pts4 = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1]], dtype=np.float32)
        else:
            bbox = dets[0:4].reshape((2, 2))
            bbox_pts4 = datasets_faceswap.get_box_lm4p(bbox)

        warp_mat_crop = datasets_faceswap.transformation_from_points(bbox_pts4, datasets_faceswap.mean_box_lm4p_512)
        image_crop512 = cv2.warpAffine(np.array(image_pil), warp_mat_crop, (512, 512), flags=cv2.INTER_LINEAR)
        image_pil = Image.fromarray(image_crop512)
        face_info = app.get(cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR))
        face_info = sorted(face_info, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * x["bbox"][3] - x["bbox"][1])[-1]
        pts5 = face_info["kps"]
        warp_mat = datasets_faceswap.get_affine_transform(pts5, datasets_faceswap.mean_face_lm5p_256)
        image_crop256 = cv2.warpAffine(np.array(image_pil), warp_mat, (256, 256), flags=cv2.INTER_LINEAR)
        return image_pil, image_crop256, warp_mat

    def convert_batch_to_nprgb(self, batch, nrow):
        grid_tensor = ttf.make_grid(batch * 0.5 + 0.5, nrow=nrow)
        im_rgb = (255 * grid_tensor.permute(1, 2, 0).cpu().numpy()).astype("uint8")
        return im_rgb


class Inference:
    def __init__(self, config: Config, model_manager: ModelManager, image_processor: ImageProcessor):
        self.config = config
        self.model_manager = model_manager
        self.image_processor = image_processor

    def infer(self):
        os.makedirs(self.config.output, exist_ok=True)
        save_drive_path = os.path.join(self.config.output, "drive")
        os.makedirs(save_drive_path, exist_ok=True)
        save_swap_path = os.path.join(self.config.output, "swap")
        os.makedirs(save_swap_path, exist_ok=True)
        save_concat_path = os.path.join(self.config.output, "concat")
        os.makedirs(save_concat_path, exist_ok=True)

        src_img_list = [x for x in os.listdir(self.config.source) if x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg")]
        src_img_list.sort()
        drive_img_list = [x for x in os.listdir(self.config.target) if x.endswith(".png") or x.endswith(".jpg") or x.endswith(".jpeg")]

        for src_im_file in src_img_list:
            src_im_name = src_im_file.split(".")[0]
            src_img_path = os.path.join(self.config.source, src_im_file)
            src_im_pil = self.image_processor.load_image(src_img_path)
            src_im_pil, src_im_crop256, warp_mat = self.image_processor.preprocess_image(src_im_pil, self.model_manager.app)

            for drive_im_file in drive_img_list:
                drive_im_name = drive_im_file.split(".")[0]
                drive_img_path = os.path.join(self.config.target, drive_im_file)
                drive_im_pil = self.image_processor.load_image(drive_img_path)
                drive_im_pil, drive_im_crop256, drive_warp_mat = self.image_processor.preprocess_image(drive_im_pil, self.model_manager.app)

                # Perform face reenactment
                reenactment_result = self.perform_face_reenactment(src_im_pil, src_im_crop256, drive_im_pil, drive_im_crop256, drive_warp_mat)
                reenactment_result.save(os.path.join(save_drive_path, src_im_file), quality=100)

                # Perform face swapping
                swap_result = self.perform_face_swapping(src_im_pil, src_im_crop256, drive_im_pil, drive_im_crop256, drive_warp_mat)
                swap_result.save(os.path.join(save_swap_path, src_im_file), quality=100)

                # Save concatenated results
                self.save_concatenated_results(
                    src_im_pil, drive_im_pil, reenactment_result, swap_result, src_im_name, drive_im_name, save_concat_path
                )

    def perform_face_reenactment(self, src_im_pil, src_im_crop256, drive_im_pil, drive_im_crop256, drive_warp_mat):
        src_im_tensor = self.image_processor.pil2tensor(src_im_pil).view(1, 3, 512, 512).to(self.model_manager.device)
        drive_im_tensor = self.image_processor.pil2tensor(drive_im_pil).view(1, 3, 512, 512).to(self.model_manager.device)

        src_d3d_coeff = self.model_manager.net_d3dfr(src_im_crop256)
        gt_d3d_coeff = self.model_manager.net_d3dfr(drive_im_crop256)
        gt_d3d_coeff[:, 0:80] = src_d3d_coeff[:, 0:80]
        gt_pts68 = self.model_manager.bfm_facemodel.get_lm68(gt_d3d_coeff)

        im_pts70 = draw_pts70_batch(gt_pts68, gt_d3d_coeff[:, 257:], drive_warp_mat.reshape((1, 2, 3)), 512, return_pt=True)
        im_pts70 = im_pts70.to(drive_im_tensor)
        face_masks_tar = (self.model_manager.net_seg_res18(torch.cat([src_im_tensor, im_pts70], dim=1)) > 0.5).float()
        controlnet_image = im_pts70 * face_masks_tar + src_im_tensor * (1 - face_masks_tar)
        controlnet_image = controlnet_image.to(dtype=self.model_manager.weight_dtype)

        face_masks_tar_pad = F.pad(face_masks_tar, (16, 16, 16, 16), "constant", 0)
        blend_mask = F.max_pool2d(face_masks_tar_pad, kernel_size=17, stride=1, padding=8)
        blend_mask = F.avg_pool2d(blend_mask, kernel_size=17, stride=1, padding=8)
        blend_mask = blend_mask[:, :, 16:528, 16:528]

        faceid = self.model_manager.net_arcface(F.interpolate(src_im_crop256, [128, 128], mode="bilinear"))
        encoder_hidden_states_src = self.model_manager.net_id2token(faceid).to(dtype=self.model_manager.weight_dtype)

        last_hidden_state = self.model_manager.net_vision_encoder(
            self.model_manager.clip_image_processor(images=src_im_pil, return_tensors="pt")
            .pixel_values.view(-1, 3, 224, 224)
            .to(self.model_manager.device)
        ).last_hidden_state
        controlnet_encoder_hidden_states_src = self.model_manager.net_image2token(last_hidden_state).to(dtype=self.model_manager.weight_dtype)

        empty_prompt_token = (
            torch.load("empty_prompt_embedding.pth").view(1, 77, 768).to(dtype=self.model_manager.weight_dtype).to(self.model_manager.device)
        )

        set_seed(999)
        generator = torch.manual_seed(0)
        image = self.model_manager.pipe(
            prompt_embeds=encoder_hidden_states_src,
            negative_prompt_embeds=empty_prompt_token,
            controlnet_prompt_embeds=controlnet_encoder_hidden_states_src,
            controlnet_negative_prompt_embeds=empty_prompt_token,
            image=controlnet_image,
            num_inference_steps=25,
            generator=generator,
            guidance_scale=5.0,
        ).images[0]

        res_tensor = self.image_processor.pil2tensor(image).view(1, 3, 512, 512).to(drive_im_tensor)
        res_tensor = res_tensor * blend_mask + src_im_tensor * (1 - blend_mask)
        return Image.fromarray((res_tensor[0] * 127.5 + 128).cpu().numpy().astype("uint8").transpose(1, 2, 0))

    def perform_face_swapping(self, src_im_pil, src_im_crop256, drive_im_pil, drive_im_crop256, drive_warp_mat):
        src_im_tensor = self.image_processor.pil2tensor(src_im_pil).view(1, 3, 512, 512).to(self.model_manager.device)
        drive_im_tensor = self.image_processor.pil2tensor(drive_im_pil).view(1, 3, 512, 512).to(self.model_manager.device)

        src_d3d_coeff = self.model_manager.net_d3dfr(src_im_crop256)
        gt_d3d_coeff = self.model_manager.net_d3dfr(drive_im_crop256)
        gt_d3d_coeff[:, 0:80] = src_d3d_coeff[:, 0:80]
        gt_pts68 = self.model_manager.bfm_facemodel.get_lm68(gt_d3d_coeff)

        im_pts70 = draw_pts70_batch(gt_pts68, gt_d3d_coeff[:, 257:], drive_warp_mat.reshape((1, 2, 3)), 512, return_pt=True)
        im_pts70 = im_pts70.to(drive_im_tensor)
        face_masks_tar = (self.model_manager.net_seg_res18(torch.cat([src_im_tensor, im_pts70], dim=1)) > 0.5).float()
        controlnet_image_swap = im_pts70 * face_masks_tar + drive_im_tensor * (1 - face_masks_tar)
        controlnet_image_swap = controlnet_image_swap.to(dtype=self.model_manager.weight_dtype)

        face_masks_tar_pad = F.pad(face_masks_tar, (16, 16, 16, 16), "constant", 0)
        blend_mask = F.max_pool2d(face_masks_tar_pad, kernel_size=17, stride=1, padding=8)
        blend_mask = F.avg_pool2d(blend_mask, kernel_size=17, stride=1, padding=8)
        blend_mask = blend_mask[:, :, 16:528, 16:528]

        faceid = self.model_manager.net_arcface(F.interpolate(src_im_crop256, [128, 128], mode="bilinear"))
        encoder_hidden_states_src = self.model_manager.net_id2token(faceid).to(dtype=self.model_manager.weight_dtype)

        last_hidden_state = self.model_manager.net_vision_encoder(
            self.model_manager.clip_image_processor(images=drive_im_pil, return_tensors="pt")
            .pixel_values.view(-1, 3, 224, 224)
            .to(self.model_manager.device)
        ).last_hidden_state
        controlnet_encoder_hidden_states_tar = self.model_manager.net_image2token(last_hidden_state).to(dtype=self.model_manager.weight_dtype)

        empty_prompt_token = (
            torch.load("empty_prompt_embedding.pth").view(1, 77, 768).to(dtype=self.model_manager.weight_dtype).to(self.model_manager.device)
        )

        generator = torch.manual_seed(0)
        image = self.model_manager.pipe(
            prompt_embeds=encoder_hidden_states_src,
            negative_prompt_embeds=empty_prompt_token,
            controlnet_prompt_embeds=controlnet_encoder_hidden_states_tar,
            controlnet_negative_prompt_embeds=empty_prompt_token,
            image=controlnet_image_swap,
            num_inference_steps=25,
            generator=generator,
            guidance_scale=5.0,
        ).images[0]

        swap_res_tensor = self.image_processor.pil2tensor(image).view(1, 3, 512, 512).to(drive_im_tensor)
        swap_res_tensor = swap_res_tensor * blend_mask + drive_im_tensor * (1 - blend_mask)
        return Image.fromarray((swap_res_tensor[0] * 127.5 + 128).cpu().numpy().astype("uint8").transpose(1, 2, 0))

    def save_concatenated_results(self, src_im_pil, drive_im_pil, reenactment_result, swap_result, src_im_name, drive_im_name, save_concat_path):
        src_im_tensor = self.image_processor.pil2tensor(src_im_pil).view(1, 3, 512, 512)
        drive_im_tensor = self.image_processor.pil2tensor(drive_im_pil).view(1, 3, 512, 512)
        reenactment_tensor = self.image_processor.pil2tensor(reenactment_result).view(1, 3, 512, 512)
        swap_tensor = self.image_processor.pil2tensor(swap_result).view(1, 3, 512, 512)

        im_rgb_pil = Image.fromarray(
            self.image_processor.convert_batch_to_nprgb(torch.cat([src_im_tensor, drive_im_tensor, reenactment_tensor, swap_tensor]), 4)
        )
        im_rgb_pil.save(os.path.join(save_concat_path, f"{src_im_name}_{drive_im_name}.jpg"), quality=100)


if __name__ == "__main__":
    config = Config()
    model_manager = ModelManager(config)
    image_processor = ImageProcessor(config)
    inference = Inference(config, model_manager, image_processor)
    inference.infer()
