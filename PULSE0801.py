import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torchvision import transforms
from PIL import Image

from stylegan import G_synthesis

img_path = "img/selfie.png"
img_lr = transforms.ToTensor()(Image.open(img_path).convert('RGB')).unsqueeze(0).cuda()
img_lr_res = img_lr[0].size()[1:]

plt.imshow(np.array(transforms.ToPILImage()(img_lr[0].cpu())))
plt.title("Low resolution image ({} x {})".format(img_lr_res[0], img_lr_res[0]))
plt.axis('off');

generator = G_synthesis().cuda()
generator.load_state_dict(torch.load("synthesis.pt"));

gaussian_fit = torch.load("gaussian_fit.pt");
gaussian_fit.keys()

lrelu = torch.nn.LeakyReLU(negative_slope=0.025);

for param in generator.parameters():
    param.requires_grad = False

    noise_trainable = [torch.randn((1, 1, 2 ** (i // 2 + 2), 2 ** (i // 2 + 2)),
                                   dtype=torch.float, device='cuda', requires_grad=True)
                       for i in range(5)]

    noise_fixed = [torch.randn((1, 1, 2 ** (i // 2 + 2), 2 ** (i // 2 + 2)),
                               dtype=torch.float, device='cuda', requires_grad=False)
                   for i in range(5, 17)]

    noise_zero = [torch.zeros((1, 1, 2 ** (17 // 2 + 2), 2 ** (17 // 2 + 2)),
                              dtype=torch.float, device='cuda', requires_grad=False)]

    noise = noise_trainable + noise_fixed + noise_zero

    latent = torch.randn((1, 1, 512), dtype=torch.float, device='cuda', requires_grad=True)

radius = 0
with torch.no_grad():
    radius = latent.norm(dim=2, keepdim=True)

    print("| sqrt(511) - |z| | = {:.4}".format(np.abs(np.sqrt(511) - radius.cpu().item())))

optim = torch.optim.Adam([latent] + noise_trainable, lr=0.4)

mse = nn.MSELoss().cuda()

def ds(img_sr, target_res):
    """Downsamples """
    return nn.functional.interpolate(img_sr, target_res, mode='bicubic', align_corners=False)


best_imgs = []
min_mse = float("inf")
steps = 100

for step in range(steps):

    optim.zero_grad()

    #  1. Tile the latent vector
    z = latent.expand(-1, 18, -1)

    #  2. Fit z to gaussian
    z = lrelu(z * gaussian_fit["std"] + gaussian_fit["mean"])

    #  3. Generate super-resolution image
    img_sr = generator(z, noise)
    img_sr = ((img_sr + 1.0) / 2.0).clamp(0, 1)

    #  4. Downsample generated image
    img_sr_ds = ds(img_sr, img_lr_res)

    #  5. Get the loss
    loss = mse(img_lr, img_sr_ds)

    #  6. Save some images
    if loss < min_mse:
        best_imgs.append(img_sr.detach().clone())
        min_mse = loss.detach()

    #  7. Optimize
    loss.backward()
    optim.step()

    #  8. Project z back to the hypersphere
    with torch.no_grad():
        latent.div_(latent.norm(dim=2, keepdim=True))
        latent.mul_(radius)

        print("Loss: {:.4}. ".format(loss), end='')
        if loss >= 0.002:
            print("Generated image might not be satisfactory. Try running the PULSE loop again.")


        def to_numpy(img):
            """Converts a tensor image to numpy array."""
            return np.array(transforms.ToPILImage()(img.cpu()))


        before = to_numpy(img_lr[0])
        after = to_numpy(best_imgs[-1][0])
        after_lr = to_numpy(ds(best_imgs[-1], img_lr_res)[0])

        titles = ['Low-resolution (given)', 'Super-resolution (generated)', 'Super-resolution downsampled']
        imgs = [before, after, after_lr]

        fig, axs = plt.subplots(1, 3, figsize=(16, 12))

        for a, img, title in zip(axs, imgs, titles):
            a.imshow(img)
            a.axis('off')
            a.set_title(title)

        plt.subplots_adjust(wspace=0.01, hspace=0.0)

        best_imgs = best_imgs[3:]
        nimg = len(best_imgs)

        num_samples = 16 if nimg >= 16 else nimg
        idx = np.linspace(0, nimg - 1, num=num_samples, endpoint=True, dtype=int)

        disp_imgs = np.array(best_imgs)[idx]
        disp_imgs = [img.squeeze(0).cpu() for img in disp_imgs]

        fig, axs = plt.subplots(1, len(disp_imgs), figsize=(6 * len(disp_imgs), 6))

        for a, img in zip(axs, disp_imgs):
            a.imshow(to_numpy(img))
            a.axis('off')
        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        fig.savefig("img/out.png")