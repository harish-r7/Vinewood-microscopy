import argparse
import os
import random
from glob import glob
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import torchvision.transforms.functional as F


class FungiSegmentationDataset(Dataset):
    def __init__(self, root_dir, split="train", select="with", image_size=256):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / split / "images"
        self.mask_dir = self.root_dir / split / "masks"
        self.image_size = image_size

        image_paths = sorted(glob(str(self.image_dir / "*.png")))
        if select == "with":
            image_paths = [p for p in image_paths if "with" in os.path.basename(p).lower()]
        elif select == "wout":
            image_paths = [p for p in image_paths if "wout" in os.path.basename(p).lower()]

        self.image_paths = image_paths
        self.mask_paths = [str(self.mask_dir / os.path.basename(p)) for p in self.image_paths]

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.mask_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()

        return mask, image


class UNetSkipConnectionBlock(nn.Module):
    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        use_dropout=False,
    ):
        super().__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = nn.BatchNorm2d(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = nn.BatchNorm2d(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


class UNetGenerator(nn.Module):
    def __init__(self, input_nc=1, output_nc=3, num_downs=8, ngf=64):
        super().__init__()
        unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True)
        unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=unet_block, use_dropout=True)
        unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=unet_block, use_dropout=True)
        unet_block = UNetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=unet_block, use_dropout=True)
        unet_block = UNetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=unet_block)
        unet_block = UNetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet_block)
        unet_block = UNetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block)
        self.model = UNetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True)

    def forward(self, x):
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=4, ndf=64, n_layers=3):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def save_tensor_as_image(tensor, path):
    tensor = tensor.detach().cpu()
    tensor = (tensor + 1.0) / 2.0
    tensor = tensor.clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    image.save(path)


def save_mask_as_png(mask_tensor, path):
    mask = mask_tensor.detach().cpu().squeeze(0)
    mask = (mask > 0.5).float()
    mask = transforms.ToPILImage()(mask)
    mask.save(path)


def create_blank_mask(output_path, image_size=256):
    blank = Image.new("L", (image_size, image_size), 0)
    blank.save(output_path)


def augment_mask(mask_tensor):
    mask = mask_tensor.clone()
    if random.random() < 0.5:
        mask = torch.flip(mask, dims=[2])
    if random.random() < 0.5:
        mask = torch.flip(mask, dims=[1])
    if random.random() < 0.3:
        mask = mask.transpose(1, 2)
    mask = (mask > 0.5).float()
    return mask


def prepare_non_fungi_samples(source_image_paths, output_image_dir, output_mask_dir, target_count=1000):
    source_images = sorted(source_image_paths)
    if not source_images:
        raise ValueError("No source non-fungi images found for gan_wout generation")

    Path(output_image_dir).mkdir(parents=True, exist_ok=True)
    Path(output_mask_dir).mkdir(parents=True, exist_ok=True)
    count = 0
    while count < target_count:
        src = source_images[count % len(source_images)]
        image = Image.open(src).convert("RGB")
        if count >= len(source_images):
            if random.random() < 0.5:
                image = F.hflip(image)
            if random.random() < 0.5:
                image = F.vflip(image)
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                image = image.rotate(angle, expand=False)

        image_name = f"gan_wout_{count:04d}.png"
        image.save(Path(output_image_dir) / image_name)
        create_blank_mask(Path(output_mask_dir) / image_name)
        count += 1


def generate_gan_with_samples(
    generator,
    source_mask_paths,
    output_image_dir,
    output_mask_dir,
    device,
    target_count=174,
):
    Path(output_image_dir).mkdir(parents=True, exist_ok=True)
    Path(output_mask_dir).mkdir(parents=True, exist_ok=True)
    generator.eval()

    transforms_mask = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )

    selected_masks = []
    total_masks = len(source_mask_paths)
    for idx in range(target_count):
        mask_path = source_mask_paths[idx % total_masks]
        mask = Image.open(mask_path).convert("L")
        mask = transforms_mask(mask)
        mask = (mask > 0.5).float()
        mask = augment_mask(mask)
        selected_masks.append((mask_path, mask))

    with torch.no_grad():
        for idx, (mask_path, mask) in enumerate(selected_masks):
            mask_batch = mask.unsqueeze(0).to(device)
            fake_image = generator(mask_batch)
            fake_image = fake_image.squeeze(0)

            image_name = f"gan_with_{idx:04d}.png"
            save_tensor_as_image(fake_image, Path(output_image_dir) / image_name)
            save_mask_as_png(mask, Path(output_mask_dir) / image_name)

    generator.train()


def train_pix2pix(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FungiSegmentationDataset(args.data_root, split="train", select="with", image_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    netG = UNetGenerator(input_nc=1, output_nc=3).to(device)
    netD = NLayerDiscriminator(input_nc=4).to(device)
    netG.apply(weights_init)
    netD.apply(weights_init)

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_l1 = nn.L1Loss()

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.999))

    real_label = 1.0
    fake_label = 0.0

    sample_dir = Path(args.sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        for i, (mask, real_image) in enumerate(dataloader, start=1):
            mask = mask.to(device)
            real_image = real_image.to(device)
            input_concat = torch.cat((mask, real_image), dim=1)

            # Update discriminator
            optimizerD.zero_grad()
            output_real = netD(input_concat)
            label_real = torch.full_like(output_real, real_label, device=device)
            loss_D_real = criterion_gan(output_real, label_real)

            fake_image = netG(mask)
            fake_concat = torch.cat((mask, fake_image.detach()), dim=1)
            output_fake = netD(fake_concat)
            label_fake = torch.full_like(output_fake, fake_label, device=device)
            loss_D_fake = criterion_gan(output_fake, label_fake)

            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizerD.step()

            # Update generator
            optimizerG.zero_grad()
            fake_concat = torch.cat((mask, fake_image), dim=1)
            output_fake_for_G = netD(fake_concat)
            label_real_for_G = torch.full_like(output_fake_for_G, real_label, device=device)
            loss_G_gan = criterion_gan(output_fake_for_G, label_real_for_G)
            loss_G_l1 = criterion_l1(fake_image, real_image) * args.l1_lambda
            loss_G = loss_G_gan + loss_G_l1
            loss_G.backward()
            optimizerG.step()

            if i % args.log_interval == 0:
                print(
                    f"Epoch [{epoch}/{args.epochs}] Batch [{i}/{len(dataloader)}] "
                    f"Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f} "
                    f"Loss_G_GAN: {loss_G_gan.item():.4f} Loss_G_L1: {loss_G_l1.item():.4f}"
                )

        # Save sample output at epoch end
        sample_mask, sample_image = next(iter(dataloader))
        sample_mask = sample_mask.to(device)
        sample_image = sample_image.to(device)
        sample_fake = netG(sample_mask)
        grid = utils.make_grid(torch.cat([sample_mask.repeat(1, 3, 1, 1), sample_fake, sample_image], dim=0), nrow=sample_mask.size(0), normalize=True, value_range=(-1, 1))
        utils.save_image(grid, sample_dir / f"epoch_{epoch:03d}.png")

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "netG_state_dict": netG.state_dict(),
                "netD_state_dict": netD.state_dict(),
                "optimizerG_state_dict": optimizerG.state_dict(),
                "optimizerD_state_dict": optimizerD.state_dict(),
            },
            checkpoint_dir / f"pix2pix_fungi_epoch_{epoch:03d}.pth",
        )

    print("Training complete. Generating samples for GAN output directories.")

    source_mask_dir = Path(args.data_root) / "train" / "masks"
    fungus_masks = sorted([str(p) for p in source_mask_dir.glob("*with*.png")])
    generate_gan_with_samples(
        netG,
        fungus_masks,
        args.gan_with_dir + "/images",
        args.gan_with_dir + "/masks",
        device,
        target_count=args.gan_with_count,
    )

    nonfungi_source_dir = Path(args.data_root) / "train" / "images"
    nonfungi_source_paths = [p for p in sorted(glob(str(nonfungi_source_dir / "*.png"))) if "wout" in os.path.basename(p).lower()]
    prepare_non_fungi_samples(
        nonfungi_source_paths,
        args.gan_wout_dir + "/images",
        args.gan_wout_dir + "/masks",
        target_count=args.gan_wout_count,
    )

    print("GAN dataset generation complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Pix2Pix fungi segmentation dataset trainer and generator")
    parser.add_argument("--data-root", type=str, default="dataset", help="Root folder for dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--image-size", type=int, default=256, help="Target image size")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--l1-lambda", type=float, default=100.0, help="L1 loss weight")
    parser.add_argument("--log-interval", type=int, default=10, help="Logging interval for batches")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint save directory")
    parser.add_argument("--sample-dir", type=str, default="samples", help="Directory for generated sample images")
    parser.add_argument("--gan-with-dir", type=str, default="gan_with", help="Output folder for generated fungi samples")
    parser.add_argument("--gan-wout-dir", type=str, default="gan_wout", help="Output folder for non-fungi samples")
    parser.add_argument("--gan-with-count", type=int, default=174, help="Number of generated fungi samples to create")
    parser.add_argument("--gan-wout-count", type=int, default=1000, help="Number of non-fungi samples to prepare")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_pix2pix(args)
