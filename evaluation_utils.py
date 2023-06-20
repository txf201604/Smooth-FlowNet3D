"""
Evaluation metrics
Borrowed from HPLFlowNet
Date: May 2020

@inproceedings{HPLFlowNet,
  title={HPLFlowNet: Hierarchical Permutohedral Lattice FlowNet for
Scene Flow Estimation on Large-scale Point Clouds},
  author={Gu, Xiuye and Wang, Yijie and Wu, Chongruo and Lee, Yong Jae and Wang, Panqu},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2019 IEEE International Conference on},
  year={2019}
}
"""

import numpy as np
import os.path as osp
import sys

def evaluate_3d(sf_pred, sf_gt):
    """
    sf_pred: (N, 3)
    sf_gt: (N, 3)
    """
    l2_norm = np.linalg.norm(sf_gt - sf_pred, axis=-1)
    EPE3D = l2_norm.mean()

    sf_norm = np.linalg.norm(sf_gt, axis=-1)
    relative_err = l2_norm / (sf_norm + 1e-4)

    acc3d_strict = (np.logical_or(l2_norm < 0.05, relative_err < 0.05)).astype(np.float).mean()
    acc3d_relax = (np.logical_or(l2_norm < 0.1, relative_err < 0.1)).astype(np.float).mean()
    outlier = (np.logical_or(l2_norm > 0.3, relative_err > 0.1)).astype(np.float).mean()

    return EPE3D, acc3d_strict, acc3d_relax, outlier


def evaluate_2d(flow_pred, flow_gt):
    """
    flow_pred: (N, 2)
    flow_gt: (N, 2)
    """

    epe2d = np.linalg.norm(flow_gt - flow_pred, axis=-1)
    epe2d_mean = epe2d.mean()

    flow_gt_norm = np.linalg.norm(flow_gt, axis=-1)
    relative_err = epe2d / (flow_gt_norm + 1e-5)

    acc2d = (np.logical_or(epe2d < 3., relative_err < 0.05)).astype(np.float).mean()

    return epe2d_mean, acc2d

class Logger(object):
    def __init__(self, out_fname):
        self.out_fd = open(out_fname, 'w')

    def log(self, out_str, end='\n'):
        """
        out_str: single object now
        """
        self.out_fd.write(str(out_str) + end)
        self.out_fd.flush()
        print(out_str, end=end, flush=True)

    def close(self):
        self.out_fd.close()

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_batch_2d_flow(pc1, pc2, predicted_pc2, paths):
    if 'KITTI' in paths[0] or 'kitti' in paths[0]:
        focallengths = []
        cxs = []
        cys = []
        constx = []
        consty = []
        constz = []
        for path in paths:
            fname = osp.split(path)[-1]
            calib_path = osp.join(
                osp.dirname(__file__),
                'calib_cam_to_cam',
                fname + '.txt')
            with open(calib_path) as fd:
                lines = fd.readlines()
                P_rect_left = \
                    np.array([float(item) for item in
                              [line for line in lines if line.startswith('P_rect_02')][0].split()[1:]],
                             dtype=np.float32).reshape(3, 4)
                focallengths.append(-P_rect_left[0, 0])
                cxs.append(P_rect_left[0, 2])
                cys.append(P_rect_left[1, 2])
                constx.append(P_rect_left[0, 3])
                consty.append(P_rect_left[1, 3])
                constz.append(P_rect_left[2, 3])
        focallengths = np.array(focallengths)[:, None, None]
        cxs = np.array(cxs)[:, None, None]
        cys = np.array(cys)[:, None, None]
        constx = np.array(constx)[:, None, None]
        consty = np.array(consty)[:, None, None]
        constz = np.array(constz)[:, None, None]

        px1, py1 = project_3d_to_2d(pc1, f=focallengths, cx=cxs, cy=cys,
                                    constx=constx, consty=consty, constz=constz)
        px2, py2 = project_3d_to_2d(predicted_pc2, f=focallengths, cx=cxs, cy=cys,
                                    constx=constx, consty=consty, constz=constz)
        px2_gt, py2_gt = project_3d_to_2d(pc2, f=focallengths, cx=cxs, cy=cys,
                                          constx=constx, consty=consty, constz=constz)
    else:
        px1, py1 = project_3d_to_2d(pc1)
        px2, py2 = project_3d_to_2d(predicted_pc2)
        px2_gt, py2_gt = project_3d_to_2d(pc2)

    flow_x = px2 - px1
    flow_y = py2 - py1

    flow_x_gt = px2_gt - px1
    flow_y_gt = py2_gt - py1

    flow_pred = np.concatenate((flow_x[..., None], flow_y[..., None]), axis=-1)
    flow_gt = np.concatenate((flow_x_gt[..., None], flow_y_gt[..., None]), axis=-1)
    return flow_pred, flow_gt


def project_3d_to_2d(pc, f=-1050., cx=479.5, cy=269.5, constx=0, consty=0, constz=0):
    x = (pc[..., 0] * f + cx * pc[..., 2] + constx) / (pc[..., 2] + constz)
    y = (pc[..., 1] * f + cy * pc[..., 2] + consty) / (pc[..., 2] + constz)

    return x, y

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")