import argparse
import os, cv2
import torch.nn as nn
import math
from functools import partial
import yaml
from dataset_g3 import Train_get_loader, Test_get_loader
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import datasets
import models
import utils
from evaluation.dataloader import EvalDataset
from evaluation.evaluator import Eval_thread
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchsummary import summary
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

def batched_predict(model, inp, coord, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred, preds


def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_psnr(test_loaders, model, testsets=None, test_root_dir=None):
    model.eval()
    root_path = './test_result/'
    testsets_ = [testsets[0]]
    for testset in testsets_:
        test_loader = test_loaders[testset]
        saved_root = os.path.join(root_path, testset)  #./val_result/
        if not os.path.exists(saved_root): os.makedirs(saved_root)

        pbar = tqdm(test_loader, leave=False, desc='val')
        for batch in pbar:
            inp = batch[0].cuda().squeeze(0)
            gts = batch[1].cuda().squeeze(0)
            subpaths = batch[2]
            ori_sizes = batch[3]

            score_prediction = torch.sigmoid(model.encoder.forward_dummy(inp, train=False)[0])

            valdata_class_path = os.path.join(saved_root, subpaths[0][0].split('/')[0])
            if not os.path.exists(valdata_class_path): os.makedirs(valdata_class_path)

            final_prediction = score_prediction  # N, 1, 256, 256
            one_calss_nums = final_prediction.size()[0]
            for inum in range(one_calss_nums):
                img = final_prediction[inum, :, :, :].unsqueeze(0)  # 1, 1, 256, 256
                ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
                name = subpaths[inum][0].split('/')[1]
                res_img = nn.functional.interpolate(img, size=ori_size, mode='bilinear', align_corners=True)
                res_img = res_img.data.cpu().numpy().squeeze()
                res_img = (res_img - res_img.min()) / (res_img.max() - res_img.min() + 1e-8)

                cv2.imwrite(valdata_class_path + '/' + name, res_img * 255)

        eval_loader = EvalDataset(saved_root, test_root_dir + testsets[0] + '/gt/')
        evaler = Eval_thread(eval_loader, cuda=True)
        s_m = evaler.Eval_Smeasure()

        Fm, prec, recall = evaler.Eval_fmeasure()
        max_f = Fm.max().item()

        mae = evaler.Eval_mae()

        # Em = evaler.Eval_Emeasure()
        # max_e = Em.max().item()             max_e, max_f,

    return max_f, s_m, mae

# def eval_psnr(test_loaders, model, testsets=None, test_root_dir=None):
#     model.eval()
#     root_path = './val_result/'
#     testsets_ = [testsets[0]]
#     for testset in testsets_:
#         test_loader = test_loaders[testset]
#         saved_root = os.path.join(root_path, testset)  #./val_result/
#         if not os.path.exists(saved_root): os.makedirs(saved_root)
#
#         pbar = tqdm(test_loader, leave=False, desc='val')
#         for batch in pbar:
#             inp = batch[0].cuda().squeeze(0)
#             gts = batch[1].cuda().squeeze(0)
#             subpaths = batch[2]
#             ori_sizes = batch[3]
#
#             score_prediction = torch.sigmoid(model.encoder.forward_dummy(inp, train=False)[0])
#
#             valdata_class_path = os.path.join(saved_root, subpaths[0][0].split('/')[0])
#             if not os.path.exists(valdata_class_path): os.makedirs(valdata_class_path)
#
#             final_prediction = score_prediction  # N, 1, 256, 256
#             one_calss_nums = final_prediction.size()[0]
#             for inum in range(one_calss_nums):
#                 img = final_prediction[inum, :, :, :].unsqueeze(0)  # 1, 1, 256, 256
#                 ori_size = (ori_sizes[inum][0].item(), ori_sizes[inum][1].item())
#                 name = subpaths[inum][0].split('/')[1]
#                 res_img = nn.functional.interpolate(img, size=ori_size, mode='bilinear', align_corners=True)
#                 res_img = res_img.data.cpu().numpy().squeeze()
#                 res_img = (res_img - res_img.min()) / (res_img.max() - res_img.min() + 1e-8)
#
#                 cv2.imwrite(valdata_class_path + '/' + name, res_img * 255)
#
#     eval_loader = EvalDataset(saved_root, test_root_dir + testsets[0] + '/gt/')
#     evaler = Eval_thread(eval_loader, cuda=True)
#     s_m = evaler.Eval_Smeasure()
#
#     Fm, prec, recall = evaler.Eval_fmeasure()
#     max_f = Fm.max().item()
#
#     mae = evaler.Eval_mae()
#
#     # Em = evaler.Eval_Emeasure()
#     # max_e = Em.max().item()             max_e, max_f,
#
#     return max_f, s_m, mae


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/test/test_cosod.yaml')
    parser.add_argument('--model', default='./mit_b4.pth')
    parser.add_argument('--prompt', default='./train_segformer_vcp_cosod/prompt_epoch_best.pth')
    parser.add_argument('--gpu', default='7')
    args = parser.parse_args()


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    test_config = config['val_dataset']
    test_root_dir = test_config['root_path']
    test_loaders = {}
    testsets = test_config['testsets']
    for testset in testsets:
        test_loader = Test_get_loader(
            os.path.join(test_root_dir, testset, 'img/'), os.path.join(test_root_dir, testset, 'gt/'), test_config['inp_size'], 1,
            shuffle=False, num_workers=8, pin=True)
        test_loaders[testset] = test_loader


    # spec = config['test_dataset']
    # dataset = datasets.make(spec['dataset'])
    # dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    # loader = DataLoader(dataset, batch_size=spec['batch_size'],
    #                     num_workers=8)

    # model = models.make(config['model']).cpu()
    # model.encoder.load_state_dict(torch.load(args.model, map_location='cpu'))
    model = models.make(config['model']).cuda()
    # model.encoder.load_state_dict(torch.load(args.model), strict=False)

    # if 'segformer' in args.model:
    if 'segformer' in config['model']['name']:
        print('loading public pretrain backbone...')
        checkpoint = load_checkpoint(model.encoder, args.model)
        model.encoder.PALETTE = checkpoint
        if args.prompt != 'none':
            print('loading prompt...')
            checkpoint = torch.load(args.prompt)
            model.encoder.backbone.prompt_generator.load_state_dict(checkpoint['prompt'])
            model.encoder.decode_head.load_state_dict(checkpoint['decode_head'])
    else:
        model.encoder.load_state_dict(torch.load(args.model), strict=False)

    # model.encoder.prompt_generator.pos_embed.load_state_dict(checkpoint_model['prompt_generator.pos_embed'], strict=False)

    metric2, metric3, metric4 = eval_psnr(test_loaders, model, testsets=test_config['testsets'],
                                          test_root_dir=test_config['root_path'])

    # metric2, metric3, metric4 = eval_psnr(test_loaders, model, data_norm=config.get('data_norm'),
    #                                                eval_type=config.get('eval_type'),
    #                                                eval_bsize=config.get('eval_bsize'),
    #                                                verbose=True)

    print('Max_Fm: {:.4f}'.format(metric2))
    print('Sm: {:.4f}'.format(metric3))
    print('MAE: {:.4f}'.format(metric4))
