import argparse
import os
import torch
import time
from model import VGG16
from vis_flux import vis_flux
from test_datasets import FluxSegmentationDataset
from torch.autograd import Variable
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from draw_test import draw_direction
from Edge import edge_pixel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
DATASET = 'PascalContext'
TEST_VIS_DIR = './test_pred_flux/'
SNAPSHOT_DIR = './snapshots/'

def get_arguments():
    """Parse all the arguments provided fromthe CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Super-BPD Network")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help="Dataset for training.")
    parser.add_argument("--test-vis-dir", type=str, default=TEST_VIS_DIR,
                        help="Directory for saving vis results during testing.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    return parser.parse_args()

args = get_arguments()

def main():

    if not os.path.exists(args.test_vis_dir + args.dataset):
        os.makedirs(args.test_vis_dir + args.dataset)

    model = VGG16()

    model.load_state_dict(torch.load(args.snapshot_dir + 'PascalContext' + '_410000.pth',map_location=torch.device('cpu')))

    model.eval()
    model.to(device)

    print(args.dataset)
    dataloader = DataLoader(FluxSegmentationDataset(dataset=args.dataset, mode='test'), batch_size=1, shuffle=False, num_workers=4)

    for i_iter, batch_data in enumerate(dataloader):

        Input_image, vis_image, dataset_lendth, image_name = batch_data

        start = time.time()
        pred_flux = model(Input_image.to(device))
        end = time.time()
        print("=" * 60)
        print("total time is %s second" % (end - start))
        print("=" * 60)

        # vis_flux(vis_image, pred_flux, image_name, args.test_vis_dir + args.dataset + '/')
        # draw_direction(vis_image.numpy()[0], gt_flux.numpy()[0], 160, 250, 40, 40)
        # draw_direction(vis_image.numpy()[0], gt_flux.numpy()[0], 100, 150, 30, 30)
        edge_pixel(image_name, pred_flux)

        pred_flux = pred_flux.data.cpu().numpy()[0, ...]
        #sio.savemat(args.test_vis_dir + args.dataset + '/' + image_name[0] + '.mat', {'flux': pred_flux})
        sio.savemat('D:\\A-work\\our_SuperBPD\\test_pred_flux\\' + args.dataset + '\\' + image_name[0] + '.mat', {'flux': pred_flux})
        print(image_name)

if __name__ == '__main__':
    main()





