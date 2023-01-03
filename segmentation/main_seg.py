import argparse
import logging
import torch
import numpy as np
import simplecv as sc
from data.thyroid import COLOR_MAP
from data.thyroid import ImageFolderDataset
from concurrent.futures import ProcessPoolExecutor
from tensorboardX import SummaryWriter
from module import foroptFPN
from torch.utils.data.dataloader import DataLoader
from simplecv.api.preprocess import comm
from simplecv.api.preprocess import segm
from tqdm import tqdm
from simplecv.data.preprocess import sliding_window
from skimage import io
from skimage import transform
import cv2

class SegmSlidingWinInference(object):
    def __init__(self):
        super(SegmSlidingWinInference, self).__init__()
        self._h = None
        self._w = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def patch(self, input_size, patch_size, stride, transforms=None):
        self.wins = sliding_window(input_size, patch_size, stride)
        self.transforms = transforms
        return self

    def merge(self, out_list):
        pred_list, win_list = list(zip(*out_list))
        num_classes = pred_list[0].size(1)
        res_img = torch.zeros(pred_list[0].size(0), num_classes, self._h, self._w, dtype=torch.float32)
        res_count = torch.zeros(self._h, self._w, dtype=torch.float32)

        for pred, win in zip(pred_list, win_list):
            res_count[win[1]:win[3], win[0]: win[2]] += 1
            res_img[:, :, win[1]:win[3], win[0]: win[2]] += pred.cpu()

        avg_res_img = res_img / res_count

        return avg_res_img

    def forward(self, model, image_np, **kwargs):
        assert self.wins is not None, 'patch must be performed before forward.'
        self._h, self._w, _ = image_np.shape
        return self._forward(model, image_np, **kwargs)

    def _forward(self, model, image_np, **kwargs):
        self.device = kwargs.get('device', self.device)
        size_divisor = kwargs.get('size_divisor', None)
        assert self.wins is not None, 'patch must be performed before forward.'
        out_list = []
        for win in tqdm(self.wins):
            x1, y1, x2, y2 = win
            image = image_np[y1:y2, x1:x2, :].astype(np.float32)
            if self.transforms is not None:
                image = self.transforms(image)
            h, w = image.shape[2:4]
            if size_divisor is not None:
                image = sc.preprocess.function.th_divisible_pad(image, size_divisor)
            image = image.to(self.device)
            with torch.no_grad():
                out = model(image)
            if size_divisor is not None:
                out = out[:, :, :h, :w]
            out_list.append((out.cpu(), win))
            torch.cuda.empty_cache()
        self.wins = None

        return self.merge(out_list)


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', default=None, type=str,
                    help='path to config file')
parser.add_argument('--ckpt_path', default=None, type=str,
                    help='path to model directory')
parser.add_argument('--image_dir', default=None, type=str,
                    help='path to image dir')
parser.add_argument('--mask_dir', default=None, type=str,
                    help='path to mask dir')
parser.add_argument('--vis_dir', default=None, type=str,
                    help='path to vis_dir')
parser.add_argument('--log_dir', default=None, type=str,
                    help='path to log')
parser.add_argument('--patch_size', default=896, type=int,
                    help='patch size')

logger = logging.getLogger('SW-Infer')
logger.setLevel(logging.INFO)

def FillHole(mask):
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    if len_contour==0:
        return mask
    contour_list = []
    contour_sum = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)  # create a black image
        img_contour = cv2.drawContours(drawing, contours, i, (255, 255, 255), -1)
        print(contours)
        contour_sum.append(np.sum(img_contour))
        # img_contour is a single contour.
        contour_list.append(img_contour)
    max_contour=max(contour_sum)
    out = np.zeros_like(mask, np.uint8) # original out image
    for i in range(len(contour_list)):
        if contour_sum[i]/max_contour>0.5:
            out+=contour_list[i]
    #out = sum(contour_list)
    return out

def FloodFill(mask):
    holes=mask.copy()
    cv2.floodFill(holes,None,(0,0),1)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if holes[i,j]==0:
                mask[i,j]=255
    return mask

def run():
    args = parser.parse_args()

    model, global_step = sc.infer_tool.build_and_load_from_file(args.config_path, args.ckpt_path)
    model.to(torch.device('cuda'))
    segm_helper = SegmSlidingWinInference()

    ppe = ProcessPoolExecutor(max_workers=4)

    dataset = ImageFolderDataset(image_dir=args.image_dir, mask_dir=args.mask_dir)

    palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()

    viz_op = sc.viz.VisualizeSegmm(args.vis_dir, palette=palette)
    miou_op = sc.metric.NPmIoU(num_classes=2, logdir=args.log_dir)

    image_trans = comm.Compose([
        segm.ToTensor(True),
        comm.THMeanStdNormalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375)),
        comm.CustomOp(lambda x: x.unsqueeze(0))
    ])

    for idx, blob in enumerate(
            DataLoader(dataset, 1, shuffle=False, pin_memory=True, num_workers=4, collate_fn=lambda x: x)):
        image, mask, filename = blob[0]
        print(filename)

        h, w = image.shape[:2]
        logging.info('Progress - [{} / {}] size = ({}, {})'.format(idx + 1, len(dataset), h, w))
        seg_helper = segm_helper.patch((h, w), patch_size=(args.patch_size, args.patch_size), stride=512,
                                       transforms=image_trans)

        out = seg_helper.forward(model, image, size_divisor=32)

        out = out.argmax(dim=1)

        if mask is not None:
            miou_op.forward(mask, out)
        ppe.submit(viz_op, out.numpy(), filename)
        
        io.imsave(arr=out.numpy()[0,:,:],fname=r'./thyroid/val/temp/temp.jpg')
        img_cv=cv2.imread(r'./thyroid/val/temp/temp.jpg', 0)
        mask_out = FloodFill(img_cv)
        cv2.imwrite(r'./thyroid/val/mask_pred/'+filename, img_cv)
        img_s=transform.resize(image=img_cv,output_shape=(256,256))
        print(np.max(img_s*255))
        cv2.imwrite(r'./thyroid/val/mask_pred/'+filename, img_s*255)

    ppe.shutdown()
    ious, miou = miou_op.summary()

    # tensorboard
    sw = SummaryWriter(logdir=args.log_dir)

    sw.add_scalar('eval-miou/miou', miou, global_step=global_step)
    sw.add_scalar('eval-miou/miou-fg', ious[1:].mean(), global_step=global_step)
    for name, iou in zip(list(COLOR_MAP.keys()), ious):
        sw.add_scalar('eval-ious/{}'.format(name), iou, global_step=global_step)

    sw.close()


if __name__ == '__main__':
    run()
