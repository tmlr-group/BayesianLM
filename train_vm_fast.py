from functools import partial
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
import argparse
import sys
import os

sys.path.append(".")
from data import prepare_padding_data, prepare_watermarking_data, IMAGENETNORMALIZE
from reprogramming import *
from mapping import *
from cfg import *

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--reprogramming', choices=["padding", "watermarking"], default="padding")
    p.add_argument('--mapping', choices=["rlm", "flm", "ilm", "blm", "blmp"], default="blmp")
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--dataset', choices=["cifar10", "cifar100", "dtd", "flowers102", "ucf101", "food101", "gtsrb", "svhn", "eurosat", "oxfordpets", "stanfordcars", "sun397"], default="sun397")
    args = p.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    save_path = os.path.join(results_path,'vmfast_' + args.mapping + '_' + args.reprogramming + '_' + args.dataset + '_' + str(args.seed))

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
        loaders, class_names = prepare_watermarking_data(args.dataset, data_path=data_path, preprocess=train_preprocess, test_process=test_preprocess)

    # Network
    from torchvision.models import resnet18, ResNet18_Weights
    network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    network.requires_grad_(False)
    network.eval()

    # Visual Prompt
    if args.reprogramming == "padding":
        visual_prompt = PaddingVR(imgsize, mask=configs['mask'], normalize=normalize).to(device)
    elif args.reprogramming == "watermarking":
        visual_prompt = WatermarkingVR(imgsize, padding_size).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(visual_prompt.parameters(), lr=config_vm_fast['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5 * config_vm_fast['epoch']),
                                                                            int(0.72 * config_vm_fast['epoch'])], gamma=0.1)
    os.makedirs(save_path, exist_ok=True)

    # Train
    best_acc = 0.
    scaler = GradScaler()

    # Label Mapping for RLM, fLM, ILM, BLM, BLM++
    if args.mapping == 'rlm':
        mapping_matrix = torch.randperm(1000)[:len(class_names)]
    elif args.mapping == 'flm':
        mapping_matrix = one2one_mappnig_matrix(visual_prompt, network, loaders['train'])
    elif args.mapping == 'ilm':
        mapping_matrix = one2one_mappnig_matrix(visual_prompt, network, loaders['train'])
    elif args.mapping == 'blm':
        mapping_matrix = blm_reweight_matrix(visual_prompt, network, loaders['train'], lap=config_vm_fast['blm']['lap'])
    elif args.mapping == 'blmp':
        mapping_matrix = blmp_reweight_matrix(visual_prompt, network, loaders['train'], lap=config_vm_fast['blmp']['lap'], k=int(len(class_names) * config_vm_fast['blmp']['topk_ratio']))

    for epoch in range(config_vm_fast['epoch']):
        if args.mapping in ['rlm', 'flm', 'ilm']:
            label_mapping = partial(label_mapping_base, mapping_sequence=mapping_matrix)
        elif args.mapping in ['blm', 'blmp']:
            label_mapping = partial(label_mapping_calculation, mapping_matrix=mapping_matrix)

        visual_prompt.train()
        total_num = 0
        true_num = 0
        loss_sum = 0
        probs_list = []
        ys = []
        pbar = tqdm(loaders['train'], total=len(loaders['train']), desc=f"Training Epo {epoch}", ncols=100)
        for x, y in pbar:
            pbar.set_description_str(f"Training Epo {epoch}", refresh=True)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                fx0 = network(visual_prompt(x))
                fx = label_mapping(fx0)
                loss = F.cross_entropy(fx, y, reduction='mean')

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            loss_sum += loss.item() * fx.size(0)
            pbar.set_postfix_str(f"Training Acc {100 * true_num / total_num:.2f}%")

            with torch.no_grad():
                if args.mapping == 'blmp':
                    probabilities = F.softmax(fx0, dim=1)
                    top_values, top_indices = torch.topk(probabilities, int(len(class_names) * config_vm_fast['blmp']['topk_ratio']), dim=1)
                    result = torch.zeros_like(probabilities)
                    result.scatter_(1, top_indices, top_values)
                elif args.mapping in ['blm', 'ilm', 'flm', 'rlm']:
                    result = fx0

            probs_list.append(result.cpu().float())
            ys.append(y)

        probs = torch.cat(probs_list, dim=0)
        ys = torch.cat(ys).cpu().int()

        scheduler.step()

        # Test
        visual_prompt.eval()
        total_num = 0
        true_num = 0
        pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Testing Epo {epoch}", ncols=100)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                fx0 = network(visual_prompt(x))
                fx = label_mapping(fx0)
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            acc = true_num / total_num
            pbar.set_postfix_str(f"Testing Acc {100 * acc:.2f}%, Best Acc {100 * best_acc:.2f}%")

        # Save CKPT
        state_dict = {
            "visual_prompt_dict": visual_prompt.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
            "mapping_matrix": mapping_matrix,
        }
        if acc > best_acc:
            best_acc = acc
            state_dict['best_acc'] = best_acc
            torch.save(state_dict, os.path.join(save_path, 'best.pth'))

        # Update the mapping matrix at the end of the epoch
        if args.mapping == 'blmp':
            mapping_matrix = update_blmp_reweight_matrix(probs, ys, device, lap=config_vm_fast['blmp']['lap'])
        elif args.mapping == 'blm':
            mapping_matrix = update_blm_reweight_matrix(probs, ys, device, lap=config_vm_fast['blm']['lap'])
        elif args.mapping == 'ilm':
            mapping_matrix = update_one2one_mappnig_matrix(probs, ys)
