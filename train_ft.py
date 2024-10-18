from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
import argparse
import sys
import os

sys.path.append(".")
from data import prepare_padding_data, prepare_watermarking_data, IMAGENETNORMALIZE
from reprogramming import *
from cfg import *
from mapping import FTlayer

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--reprogramming', choices=["padding", "watermarking"], default="padding")
    p.add_argument('--restrict', choices=["none", "nobias", "sigmoid"], default="none")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dataset', choices=["cifar10", "cifar100", "dtd", "flowers102", "ucf101", "food101", "gtsrb", "svhn", "eurosat", "oxfordpets", "stanfordcars", "sun397"], default="sun397")
    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    save_path = os.path.join(results_path, 'vmft_' + args.restrict +  '_' + args.reprogramming + '_' + args.dataset + '_' + str(args.seed))

    imgsize = 224
    padding_size = imgsize / 2

    # Data
    if args.reprogramming == "padding":
        loaders, configs = prepare_padding_data(args.dataset, data_path=data_path)
        class_names = configs['class_names']
        normalize = transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std'])
    elif args.reprogramming == "watermarking":
        train_preprocess = transforms.Compose([
            transforms.Resize((imgsize + 4, imgsize + 4)),
            transforms.RandomCrop(imgsize),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
        ])
        test_preprocess = transforms.Compose([
            transforms.Resize((imgsize, imgsize)),
            transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
        ])
        loaders, class_names = prepare_watermarking_data(args.dataset, data_path=data_path, preprocess=train_preprocess,
                                                     test_process=test_preprocess)

    # Network
    from torchvision.models import resnet18, ResNet18_Weights
    network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    network.requires_grad_(False)
    network.eval()
    ft_logits = FTlayer(class_num=len(class_names), norm=args.restrict).to(device)
    ft_logits.requires_grad_(True)

    # Visual Prompt
    if args.reprogramming == "padding":
        visual_prompt = PaddingVR(imgsize, mask=configs['mask'], normalize=normalize).to(device)
    elif args.reprogramming == "watermarking":
        visual_prompt = WatermarkingVR(imgsize, padding_size).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(visual_prompt.parameters(), lr=config_vm['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5 * config_vm['epoch']), int(0.72 * config_vm['epoch'])], gamma=0.1)

    ft_optimizer = torch.optim.Adam(ft_logits.parameters(), lr=config_vm['ft_lr'])
    ft_scheduler = torch.optim.lr_scheduler.MultiStepLR(ft_optimizer, milestones=[int(0.5 * config_vm['epoch']), int(0.72 * config_vm['epoch'])], gamma=0.1)

    os.makedirs(save_path, exist_ok=True)

    # Train
    best_acc = 0.
    scaler = GradScaler()


    for epoch in range(config_vm['epoch']):
        visual_prompt.train()
        ft_logits.train()
        total_num = 0
        true_num = 0
        loss_sum = 0
        pbar = tqdm(loaders['train'], total=len(loaders['train']), desc=f"Training Epo {epoch}", ncols=100)
        for x, y in pbar:
            pbar.set_description_str(f"Training Epo {epoch}", refresh=True)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            ft_optimizer.zero_grad()
            with autocast():
                fx = ft_logits(network(visual_prompt(x)))
                loss = F.cross_entropy(fx, y, reduction='mean')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.step(ft_optimizer)
            scaler.update()
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            loss_sum += loss.item() * fx.size(0)
            pbar.set_postfix_str(f"Training Acc {100 * true_num / total_num:.2f}%")
        scheduler.step()
        ft_scheduler.step()

        # Test
        visual_prompt.eval()
        ft_logits.eval()
        total_num = 0
        true_num = 0
        pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Testing Epo {epoch}", ncols=100)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                fx0 = network(visual_prompt(x))
                fx = ft_logits(fx0)
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            acc = true_num / total_num
            pbar.set_postfix_str(f"Testing Acc {100 * acc:.2f}%, Best Acc {100 * best_acc:.2f}%")

        # Save CKPT
        state_dict = {
            "visual_prompt_dict": visual_prompt.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "mapping_matrix": ft_logits.state_dict(),
        }
        if acc > best_acc:
            best_acc = acc
            state_dict['best_acc'] = best_acc
            torch.save(state_dict, os.path.join(save_path, 'best.pth'))
