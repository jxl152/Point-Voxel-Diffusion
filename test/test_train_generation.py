from utils.file_utils import *
from train_generation import get_dataset


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
    dataroot = "data/ShapeNetCore.v2.PC15k/"
    npoints = 2048
    category = "airplane"
    train_dataset, test_dataset = get_dataset(dataroot, npoints, category)


if __name__ == "__main__":
    # test_dir_preparation()
    test_dataset_preparation()
