import argparse
import os
import torch
import torch.nn as nn
from model import VGG16
from vis_flux import vis_flux
from datasets import FluxSegmentationDataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Edge import edge_pixel

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
INI_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-4
EPOCHES = 10000
DATASET = 'PascalContext'
SNAPSHOT_DIR = './snapshots/'
TRAIN_DEBUG_VIS_DIR = './train_debug_vis/'


def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Super-BPD Network")  # 创建一个解析对象
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="Dataset for training.")  # 向该对象中添加你要关注的命令行参数和选项
    parser.add_argument("--train-debug-vis-dir", type=str, default=TRAIN_DEBUG_VIS_DIR,
                        help="Directory for saving vis results during training.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    return parser.parse_args()  # 将解析返回


args = get_arguments()


def loss_calc(pred_flux, gt_flux, weight_matrix):
    device_id = pred_flux.device
    weight_matrix = weight_matrix.cpu()
    gt_flux = gt_flux.cpu()

    # mo_loss
    # mo_loss = 0.001 * ((pred_flux.norm(p=2, dim=1) - gt_flux.norm(p=2, dim=1)) ** 2)
    # mo_loss = mo_loss.sum()
    # print("==" * 60)
    # print(mo_loss)
    # print("==" * 60)

    # gt_flux = 0.999999 * gt_flux / (gt_flux.norm(p=2, dim=1) + 1e-9)   # 真实Dp：公式（1）
    # torch.norm(p=2, dim=1) 每行的每一列数据进行L2范数计算距离
    # torch.norm(p=2, dim=0) 每列的每一行数据进行L2范数计算距离
    """
    print("-----------------Bp_P--------------")
    Bp_P = (0.999999 * gt_flux)
    print((0.999999 * gt_flux).size())
    print(Bp_P)
    np.savetxt('D_1.csv', Bp_P[0][0])
    np.savetxt('D_2.csv', Bp_P[0][1])
    print("-----------------mo--------------")
    mo = (gt_flux.norm(p=2, dim=1) + 1e-9)
    print((gt_flux.norm(p=2, dim=1) + 1e-9).size())
    print(mo)
    np.savetxt('mo.csv', mo[0])
    print('over')
    """

    # norm loss
    norm_loss = weight_matrix * ((pred_flux - gt_flux) ** 2 )
    norm_loss = norm_loss.sum()

    # print((pred_flux.norm(p=2, dim=1) + 1e-9).shape)
    # print(pred_flux.shape)
    gt_flux_temp = 0.999999 * gt_flux / (gt_flux.norm(p=2, dim=1) + 1e-9)
    pred_flux_temp = 0.999999 * pred_flux / (pred_flux.norm(p=2, dim=1) + 1e-9)
    # print("--------gt_flux.norm-----")
    # print((gt_flux.norm(p=2, dim=1) + 1e-9))
    # print("-----" * 50)

    # angle loss
    angle_loss = weight_matrix * (torch.acos(torch.sum(pred_flux_temp * gt_flux_temp, dim=1))) ** 2
    angle_loss = angle_loss.sum()

    # 梯度损失
    # import torch.nn.functional as F
    # loss = 500*F.l1_loss(pred_flux, gt_flux)
    # loss = loss.sum()
    # print(loss)
    # print(norm_loss)
    # print(angle_loss)

    return norm_loss, angle_loss


def get_params(model, key, bias=False):
    # for backbone
    if key == "backbone":
        for m in model.named_modules():
            if "backbone" in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    if not bias:
                        yield m[1].weight
                    else:
                        yield m[1].bias
    # for added layer
    if key == "added":
        for m in model.named_modules():
            if "backbone" not in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    if not bias:
                        yield m[1].weight
                    else:
                        yield m[1].bias


def adjust_learning_rate(optimizer, step):
    if step == 8e4:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1


def main():
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    if not os.path.exists(args.train_debug_vis_dir + args.dataset):
        os.makedirs(args.train_debug_vis_dir + args.dataset)

    model = VGG16()

    saved_dict = torch.load('D://A-work//our_SuperBPD//PascalContext_400000.pth', map_location=torch.device('cpu'))
    model_dict = model.state_dict()
    saved_key = list(saved_dict.keys())
    model_key = list(model_dict.keys())
    #("=====" * 50)
    #print(len(saved_key))
    #print(len(model_key))


    for i in range(26):
        model_dict[model_key[i]] = saved_dict[saved_key[i]]
        #print("--" * 50)
        #print(saved_key[i])
        #print(model_key[i])

    model.load_state_dict(model_dict)

    model.train()
    model.cpu()

    optimizer = torch.optim.Adam(
        params=[
            {
                "params": get_params(model, key="backbone", bias=False),
                "lr": INI_LEARNING_RATE  # 后面对学习率进行衰减
            },
            {
                "params": get_params(model, key="backbone", bias=True),
                "lr": 2 * INI_LEARNING_RATE
            },
            {
                "params": get_params(model, key="added", bias=False),
                "lr": 10 * INI_LEARNING_RATE
            },
            {
                "params": get_params(model, key="added", bias=True),
                "lr": 20 * INI_LEARNING_RATE
            },
        ],
        weight_decay=WEIGHT_DECAY
    )

    dataloader = DataLoader(FluxSegmentationDataset(dataset=args.dataset, mode='train'), batch_size=1, shuffle=True,
                            num_workers=4)

    global_step = 0

    for epoch in range(1, EPOCHES):

        for i_iter, batch_data in enumerate(dataloader):

            global_step += 1

            Input_image, vis_image, gt_mask, gt_flux, weight_matrix, dataset_lendth, image_name = batch_data

            optimizer.zero_grad()
            # print(Input_image)
            # print(Input_image.cpu())
            pred_flux = model(Input_image.cpu())

            norm_loss, angle_loss = loss_calc(pred_flux, gt_flux, weight_matrix)

            total_loss = norm_loss + angle_loss

            total_loss.backward()

            optimizer.step()

            edge_pixel(image_name, gt_flux)

            vis_flux(vis_image, pred_flux, gt_flux, gt_mask, image_name,
                     args.train_debug_vis_dir + args.dataset + '/')

            return

            if global_step % 100 == 0:
                print('epoche {} i_iter/total {}/{} norm_loss {:.2f} angle_loss {:.2f}'
                      .format(epoch, i_iter, int(dataset_lendth.data), norm_loss, angle_loss))

            if global_step % 500 == 0:
                vis_flux(vis_image, pred_flux, gt_flux, gt_mask, image_name,
                         args.train_debug_vis_dir + args.dataset + '/')

            if global_step % 1e4 == 0:
                torch.save(model.state_dict(), args.snapshot_dir + args.dataset + '_' + str(global_step) + '.pth')

            if global_step % 4e5 == 0:
                return


if __name__ == '__main__':
    main()



