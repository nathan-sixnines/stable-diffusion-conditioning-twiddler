import argparse, os, sys, glob
from random import seed
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor

import pickle

torch.set_default_tensor_type(torch.HalfTensor)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

config = OmegaConf.load("configs/stable-diffusion/v1-inference.yaml")
model = load_model_from_config(config, "models/ldm/stable-diffusion-v1/model.ckpt")
model = model.half()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = model.to(device)

sampler = PLMSSampler(model)

def sample(c, filename):

    seed_everything(42)

    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():

                params = pickle.load(open('sample_args.pkl', 'rb'))

                for k, v in params.items():
                    print(k)
                    print(type(v))

                params['conditioning'] = c.half()

                params['unconditional_conditioning'] = params['unconditional_conditioning'].half()

                samples_ddim, _ = sampler.sample(**params)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img.save( f"{filename}.png")

#values = {(0, 1, 631): (1.5844104290008545, -1.9610470533370972), (0, 1, 662): (1.761114478111267, -1.7567311525344849), (0, 1, 594): (1.6169713735580444, -1.7154532670974731), (0, 1, 167): (2.6996772289276123, -0.6075859665870667), (0, 1, 361): (1.7535181045532227, -1.3490931987762451), (0, 1, 753): (0.16311094164848328, -2.924175500869751), (0, 1, 155): (0.1097082644701004, -2.9620752334594727), (0, 1, 184): (-0.4131699204444885, -3.3933939933776855), (0, 1, 582): (1.7582911252975464, -1.1144362688064575), (0, 1, 604): (0.8619845509529114, -2.005622625350952), (0, 1, 151): (2.5161654949188232, -0.32787466049194336), (0, 2, 79): (0.8848702907562256, -1.9535964727401733), (0, 1, 452): (1.2512072324752808, -1.5800508260726929), (0, 1, 156): (0.8811267614364624, -1.9181618690490723), (0, 1, 717): (0.6006028056144714, -2.16979718208313), (0, 1, 33): (1.62406587600708, -1.1343960762023926), (0, 1, 165): (1.3907853364944458, -1.3246147632598877), (0, 2, 491): (1.2030394077301025, -1.4970526695251465), (0, 1, 345): (1.679753065109253, -1.0141667127609253), (0, 1, 711): (1.3278063535690308, -1.3652255535125732)}
values = {(0, 14, 544): (2.1020896434783936, -2.293964147567749), (0, 6, 302): (1.2745475769042969, -2.8469276428222656), (0, 15, 544): (1.920946478843689, -2.014676332473755), (0, 14, 222): (2.3864591121673584, -1.4947199821472168), (0, 14, 398): (2.030101776123047, -1.3967053890228271), (0, 14, 33): (1.2458785772323608, -2.1167280673980713), (0, 6, 470): (1.7773962020874023, -1.470218539237976), (0, 6, 174): (1.0338664054870605, -2.181520462036133), (0, 7, 302): (0.1174919381737709, -3.09214186668396), (0, 7, 646): (1.1616055965423584, -1.9653006792068481), (0, 14, 204): (0.9736480116844177, -2.143932580947876), (0, 14, 322): (1.4684770107269287, -1.6042556762695312), (0, 6, 105): (1.9265365600585938, -1.0701429843902588), (0, 7, 76): (1.4183121919631958, -1.5087047815322876), (0, 14, 429): (0.9970932006835938, -1.862122893333435), (0, 6, 66): (0.6936299800872803, -2.162010669708252), (0, 14, 344): (1.7607487440109253, -1.0876396894454956), (0, 6, 257): (0.7429764270782471, -2.1023828983306885), (0, 6, 70): (0.9557280540466309, -1.8653149604797363), (0, 14, 370): (1.9010149240493774, -0.9125770330429077)}


frames = 100

for i in range(frames):

    c1 = torch.load("castle.npy")

    value = 40 - i/4

    #a = [(0, 1, 631), (0, 1, 662), (0, 1, 594), (0, 1, 167), (0, 1, 361), (0, 1, 753), (0, 1, 155), (0, 1, 184), (0, 1, 582), (0, 1, 604), (0, 1, 151), (0, 2, 79), (0, 1, 452), (0, 1, 156), (0, 1, 717), (0, 1, 33), (0, 1, 165), (0, 2, 491), (0, 1, 345), (0, 1, 711)]
    a = [(0, 14, 544), (0, 6, 302), (0, 15, 544), (0, 14, 222), (0, 14, 398), (0, 14, 33), (0, 6, 470), (0, 6, 174), (0, 7, 302), (0, 7, 646), (0, 14, 204), (0, 14, 322), (0, 6, 105), (0, 7, 76), (0, 14, 429), (0, 6, 66), (0, 14, 344), (0, 6, 257), (0, 6, 70), (0, 14, 370)]

    for i2 in a:

        value1 = values[i2][0]
        value2 = values[i2][1]

        scale = i/frames

        value = value1 * (scale*2) + value2 * (1 - scale*2) 

        c1[i2] = value

    sample(c1, f"castle-2-waterfall/{i}")