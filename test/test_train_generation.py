import numpy as np

from utils.file_utils import *
from train_generation import (
    parse_args,
    get_dataset,
    get_dataloader,
    get_betas,
    GaussianDiffusion,
    PVCNN2,
    Model
)
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def test_dir_preparation():
    print(f"__file__ = {__file__}")
    basename = os.path.basename(__file__)
    print(f"basename = {basename}")
    exp_id = os.path.splitext(os.path.basename(__file__))[0]
    print(f"exp_id = {exp_id}")
    dir_id = os.path.dirname(__file__)
    print(f"dir_id = {dir_id}")
    output_dir = get_output_dir(dir_id, exp_id)
    print(f"output_dir = {output_dir}")

def test_dataset_preparation():
    dataroot = "../data/ShapeNetCore.v2.PC15k/"
    npoints = 2048
    category = "airplane"
    # dataset
    train_dataset, test_dataset = get_dataset(dataroot, npoints, category)
    # dataset loader
    opt = parse_args()
    dataloader, _, train_sampler, _ = get_dataloader(opt, train_dataset, None)

def test_gaussian_diffusion():
    opt = parse_args()
    betas = get_betas(opt.schedule_type, opt.beta_start, opt.beta_end, opt.time_num)
    diffusion = GaussianDiffusion(betas, opt.loss_type, opt.model_mean_type, opt.model_var_type)

def test_pvcnn():
    opt = parse_args()
    model = PVCNN2(num_classes=opt.nc, embed_dim=opt.embed_dim, use_att=opt.attention,
                        dropout=opt.dropout, extra_feature_channels=0)

def test_train_generation_process():

    class TestDataset(Dataset):     # to build a small dataset for practicable debug
        def __init__(self, data):
            self.train_points = data

        def __len__(self):
            return len(self.train_points)

        def __getitem__(self, idx):
            out = {
                "idx": idx,
                "train_points": self.train_points[idx]
            }
            return out

    train_dataset = TestDataset(np.random.randn(32, 8, 3).astype(np.float32))
    dataloader = DataLoader(train_dataset, batch_size=16)
    noises_init = torch.randn(len(train_dataset), 8, 3)

    for i, data in enumerate(dataloader):
        x = data['train_points'].transpose(1, 2)    # x = {Tensor:(16,3,8)}
        noises_batch = noises_init[data['idx']].transpose(1, 2) # noises_batch = {Tensor:(16,3,8)}

def test_pvcnn():
    args = parse_args()
    model = PVCNN2(num_classes=args.nc, embed_dim=args.embed_dim, use_att=args.attention,
                   dropout=args.dropout, extra_feature_channels=0).cuda()
    device = torch.device("cuda")
    B, D, N = 16, 3, 2048
    data = torch.randn([B, D, N], device=device)
    t = torch.randint(0, 1000, size=(B,), device=data.device)
    out = model(data, t)


def test_pvcnn_2():
    device = torch.device("cuda")

    args = parse_args()
    betas = get_betas(args.schedule_type, args.beta_start, args.beta_end, args.time_num)
    model = Model(args, betas, args.loss_type, args.model_mean_type, args.model_var_type)
    model = model.cuda()

    dataroot = "../data/ShapeNetCore.v2.PC15k/"
    category = "chair"
    train_dataset, _ = get_dataset(dataroot, args.npoints, category)

    noises_init = torch.randn(len(train_dataset), args.npoints, args.nc, device=device)

    # one point cloud each iteration
    # for i in range(len(train_dataset)):
    #     point_idx = np.random.randint(0, 9999, size=2048)
    #     data = np.expand_dims(train_dataset.train_points[i, point_idx, :], axis=0)
    #     data = torch.tensor(data, device=device).transpose(1, 2)
    #     noises_batch = noises_init[i, :, :].unsqueeze(dim=0).transpose(1, 2)
    #     loss = model.get_loss_iter(data, noises_batch).mean()

    for i in range(len(train_dataset) // 16):
        point_idx = np.random.randint(0, 9999, size=2048)
        start_idx = i*16
        end_idx = min((i+1)*16, len(train_dataset))
        data = train_dataset.train_points[start_idx:end_idx, point_idx, :]
        data = torch.tensor(data, device=device).transpose(1, 2)
        noises_batch = noises_init[start_idx:end_idx, :, :].transpose(1, 2)
        loss = model.get_loss_iter(data, noises_batch).mean()


if __name__ == "__main__":
    # test_dir_preparation()
    # test_dataset_preparation()
    # test_gaussian_diffusion()
    # test_pvcnn()
    # test_train_generation_process()
    # test_pvcnn()
    test_pvcnn_2()
