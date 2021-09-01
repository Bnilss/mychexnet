import requests
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as tfs
import matplotlib.pyplot as plt
from torchvision.models import densenet121
from PIL import Image

class DenseNet121(nn.Module):
    """
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.backbone = densenet121(pretrained=False)
        num_ftrs = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        return x

class ProcessImage:
    def __init__(self):
        self.tfms = self.__init_tfms()
    
    def __init_tfms(self):
        
        self.normal = tfs.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        stack_secnd = lambda crops: torch.stack([self.normal(crop) for crop in crops])
        tfms = tfs.Compose([
                tfs.ToTensor(),
                tfs.Resize(256),
                tfs.TenCrop(224),
                tfs.Lambda(stack_secnd)
        ])
        self.one_tfms = tfs.Compose([
                    tfs.ToTensor(),
                    tfs.Resize(224),
                    self.normal
        ])
        return tfms

    def read_img(self, src, size=None):
        if isinstance(src, (torch.Tensor, np.ndarray)):
            return src
        if src.startswith("http"):
            src = requests.get(src, stream=True).raw
        img = Image.open(src).convert("RGB")
        if size:
            img = img.resize(size)
        img = np.array(img)
        return img

    def prep_one(self, src):
        img = self.read_img(src)
        return self.one_tfms(img).unsqueeze(0)

    def process(self, src):
        img = self.read_img(src)
        tfm_img = self.tfms(img)
        return tfm_img.unsqueeze(0)

def _prep_input(input_bs):
    bs, n_crop, c, h, w = input_bs.size()
    input_bs = input_bs.view(-1, c, h, w)
    return input_bs, bs, n_crop

def predict(input_bs, model):
    input_bs, bs, n_crop = _prep_input(input_bs)
    with torch.no_grad():
        model.eval()
        output = model(input_bs)
        output = output.view(bs, n_crop, -1).mean(1)
    return output

class FwdHook:
    def __init__(self, m):
        self.hook = m.register_forward_hook(self.hook_func)   
    def hook_func(self, m, i, o): self.stored = o.detach().clone()
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.hook.remove()


class BwdHook:
    def __init__(self, m):
        self.hook = m.register_backward_hook(self.hook_func)   
    def hook_func(self, m, gi, go): self.stored = go[0].detach().clone()
    def __enter__(self, *args): return self
    def __exit__(self, *args): self.hook.remove()

class GradCam:
    """
    GradCam class, to use it initialize the class then call it on a batch_of_one
    with the target layer and class index
    """
    def __init__(self, model):
        self.model = model
        self.gard, self.act = None, None

    def __compute(self, batch_of_one, target_layer, cls_index):
        with BwdHook(target_layer) as bwh:
            with FwdHook(target_layer) as fwh:
                self.model.eval()
                out = self.model(batch_of_one)
                self.act = fwh.stored
            out[0, cls_index].backward()
            self.grad = bwh.stored

    def __call__(self, batch_of_one, target_layer, cls_index):
        if not isinstance(cls_index, int):
            return [self(batch_of_one, target_layer, i) for i in cls_index]
        self.__compute(batch_of_one, target_layer, cls_index)
        w = self.grad[0].mean(dim=(1,2), keepdim=True)
        self.cam_map = (w * self.act[0]).sum(0)
        return self.cam_map

    def show_one(self, img, cam_map):
        if isinstance(cam_map, torch.Tensor):
            cam_map = cam_map.detach().cpu()
        plt.imshow(img)
        return plt.imshow(cam_map, alpha=0.5, extent=(0,224,224,0),
              interpolation='bilinear', cmap='magma')
        
    def show_multi(self, img, cams_maps, clss=None):
        if len(cams_maps) > 25:
            raise ValueError(f"Too much gradcam maps ({len(cams_maps)}), they need to be <= 25")
        fig = plt.figure(figsize=(12, 12), dpi=100)
        for i, cam_map in enumerate(cams_maps, 1):
            fig.add_subplot(5, 5, i)
            self.show_on(img, cam_map)
            if clss is not None:
                plt.title(clss[i-1])
        return fig
