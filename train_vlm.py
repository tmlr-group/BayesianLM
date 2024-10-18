import argparse
from functools import partial
import os
from torch.cuda.amp import autocast, GradScaler
import clip

import sys
sys.path.append(".")
from data import prepare_watermarking_data
from reprogramming import *
from mapping import *
from cfg import *
from data import DEFAULT_TEMPLATE, ENSEMBLE_TEMPLATES, get_saparate_text_embedding

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--mapping', choices=["ilm", "blm", "blmp"], default="blmp")
    p.add_argument('--dataset', choices=["cifar10", "cifar100", "dtd", "flowers102", "ucf101", "food101", "gtsrb", "svhn", "eurosat", "oxfordpets", "stanfordcars", "sun397"], default="sun397")
    args = p.parse_args()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(args.seed)
    save_path = os.path.join(results_path, 'vlm_' + args.mapping + '_' + args.dataset + '_' + str(args.seed))

    model, preprocess = clip.load("ViT-B/32")
    convert_models_to_fp32(model)
    model.eval()
    model.requires_grad_(False)

    loaders, class_names = prepare_watermarking_data(dataset=args.dataset, data_path=data_path, preprocess=preprocess, test_process=preprocess)
    templates = [DEFAULT_TEMPLATE] + ENSEMBLE_TEMPLATES
    txt_emb = torch.cat(get_saparate_text_embedding(class_names, templates, model))
    emb_names = np.array([f"T{i // len(class_names)} {class_names[i % len(class_names)]}" for i in range(txt_emb.size(0))])

    def network(x):
        x_emb = model.encode_image(x)
        x_emb = x_emb / x_emb.norm(dim=-1, keepdim=True)
        logits = model.logit_scale.exp() * x_emb @ txt_emb.t()
        return logits
    mapping_network = network

    # Visual Prompt
    visual_prompt = WatermarkingVR(224, 30).to(device)

    # Optimizer
    optimizer = torch.optim.SGD(visual_prompt.parameters(), lr=config_vlm['lr'], momentum=0.9)
    t_max = config_vlm['epoch'] * len(loaders['train'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)

    os.makedirs(save_path, exist_ok=True)


    # Train
    best_acc = 0.
    scaler = GradScaler()
    for epoch in range(config_vlm['epoch']):
        # Label Mapping for ILM, BLM, BLM++
        if args.mapping == 'ilm':
            mapping_matrix = one2one_mappnig_matrix(visual_prompt, network, loaders['train'])
            label_mapping = partial(label_mapping_base, mapping_sequence=mapping_matrix)
        elif args.mapping == 'blm':
            mapping_matrix = blm_reweight_matrix(visual_prompt, network, loaders['train'], lap=config_vlm['blm']['lap'])
            label_mapping = partial(label_mapping_calculation, mapping_matrix=mapping_matrix)
        elif args.mapping == 'blmp':
            mapping_matrix = blmp_reweight_matrix(visual_prompt, network, loaders['train'], lap=config_vlm['blmp']['lap'], k=int(len(class_names) * config_vlm['blmp']['topk_ratio']))
            label_mapping = partial(label_mapping_calculation, mapping_matrix=mapping_matrix)

        visual_prompt.train()
        total_num = 0
        true_num = 0
        loss_sum = 0
        pbar = tqdm(loaders['train'], total=len(loaders['train']), desc=f"Training Epo {epoch}", ncols=100)

        for i, (x, y) in enumerate(pbar):
            pbar.set_description_str(f"Training Epo {epoch}", refresh=True)
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                fx = label_mapping(network(visual_prompt(x)))
                loss = F.cross_entropy(fx, y, reduction='mean')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            model.logit_scale.data = torch.clamp(model.logit_scale.data, 0, 4.6052)
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            loss_sum += loss.item() * fx.size(0)
            pbar.set_postfix_str(f"Training Acc {100 * true_num / total_num:.2f}%")
            scheduler.step()

        # Test
        visual_prompt.eval()
        total_num = 0
        true_num = 0
        pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Testing Epo {epoch}", ncols=100)
        fx0s = []
        ys = []
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            ys.append(y)
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
