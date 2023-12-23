import cv2
import argparse
import utils
from torch.utils.data import Dataset
import numpy as np
import torch
from collections import OrderedDict
from torchvision import transforms
import torchvision
from tqdm import tqdm
import os
import torch.nn.functional as F
import time
import pydensecrf.densecrf as dcrf
from sddnet import SDDNet

cv2.setNumThreads(0)
torch.cuda.set_device(0)


class ShadowDataset(Dataset):
    def __init__(self, data_root, im_size=512, normalize=False):
        
        self.root_dir = data_root
        self.img_names = [x for x in sorted(os.listdir(self.root_dir)) if '.png' in x]
        self.size = len(self.img_names)
        self.im_size = im_size
        
    def __getitem__(self, index):
        sample = OrderedDict()
        img_name = self.img_names[index]

        img_path = os.path.join(self.root_dir, img_name)
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ret_key = ['full_image']
        ret_val = [ img ]
        
        ret_key.append('image')
        ret_val.append(cv2.resize(img, (self.im_size, self.im_size), interpolation=cv2.INTER_LINEAR))

        ret_key.append('im_shape')
        ret_val.append(img.shape)
        
        ret_key.append('im_name')
        ret_val.append(img_name)

        return OrderedDict(zip(ret_key, ret_val))


    def __len__(self):
        return self.size


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def crf_refine(img, annos):
    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape

    # img and annos should be np array with data type uint8

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])
    return res.astype('uint8')



def get_shadow_models(shadowformer_weights='./checkpoints/ISTD_plus_model_latest.pth',
                      detector_weights='./checkpoints/detector_sbu.ckpt'):

    shadowformer_args = argparse.Namespace(
    tile=None, # можно тайлить не полный кадр
    gpus='0', arch='ShadowFormer', batch_size=1, save_images=False, cal_metrics=False, 
    embed_dim=32, win_size=10, token_projection='linear', token_mlp='leff', vit_dim=256, vit_depth=12, vit_nheads=8, 
    vit_mlp_dim=512, vit_patch_size=16, global_skip=False, local_skip=False, vit_share=False, train_ps=320,  tile_overlap=32)


    model_restoration = utils.get_arch(shadowformer_args)
    utils.load_checkpoint(model_restoration, shadowformer_weights)
    model_restoration.cuda()
    model_restoration = model_restoration.eval()
    img_multiple_of = 8 * shadowformer_args.win_size
    model_restoration.shadowformer_args = shadowformer_args

    model_detection = SDDNet(backbone='efficientnet-b3', proj_planes=16, pred_planes=32, use_pretrained=True,
                   fix_backbone=False, has_se=False, dropout_2d=0, normalize=True, mu_init=0.4, reweight_mode='manual')
    ckpt = torch.load(detector_weights)
    model_detection.load_state_dict(ckpt)
    model_detection.cuda()
    model_detection = model_detection.eval()

    return model_restoration, model_detection



def run_detector(model, data, refine=False):
    image = torch.Tensor(data['image']).permute(2, 0, 1)[None].cuda() / 255.
    ans = model(image)
    h,w,c = data['im_shape']
    pred = F.interpolate(torch.sigmoid(ans['logit']), 
                                           size=(h,w), align_corners=True, 
                                           mode='bilinear')[0][0].cpu().numpy() * 255
    if refine:
        pred = crf_refine(data['full_image'], pred.astype(np.uint8))
        
    return pred



def run_shadowformer(model, data, detector_prediction):
    rgb_noisy = torch.Tensor(data['full_image']).permute(2, 0, 1)[None].cuda() / 255.
    mask = torch.Tensor(detector_prediction)[None, None].cuda() / 255.
    
    # Pad the input if not_multiple_of win_size * 8
    img_multiple_of = 8 * model.shadowformer_args.win_size
    height, width = rgb_noisy.shape[2], rgb_noisy.shape[3]
    H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
                (width + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh = H - height if height % img_multiple_of != 0 else 0
    padw = W - width if width % img_multiple_of != 0 else 0
    rgb_noisy = F.pad(rgb_noisy, (0, padw, 0, padh), 'reflect')
    mask = F.pad(mask, (0, padw, 0, padh), 'reflect')

    if model.shadowformer_args.tile is None:
        rgb_restored = model(rgb_noisy, mask)
    else:
        # test the image tile by tile
        b, c, h, w = rgb_noisy.shape
        tile = min(model.shadowformer_args.tile, h, w)
        assert tile % 8 == 0, "tile size should be multiple of 8"
        tile_overlap = model.shadowformer_args.tile_overlap

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(b, c, h, w).type_as(rgb_noisy)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = rgb_noisy[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                mask_patch = mask[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                out_patch = model(in_patch, mask_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
                W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)
        rgb_restored = E.div_(W)

    rgb_restored = torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0)) * 255

    # Unpad the output
    rgb_restored = rgb_restored[:height, :width, :]
    rgb_restored = cv2.cvtColor(rgb_restored, cv2.COLOR_RGB2BGR)

    return rgb_restored