import argparse
#from model import DCGAN
from model import DCGAN


parser = argparse.ArgumentParser()
parser.add_argument('--z_dim', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--checkpoint_dir', type=str, default='./save_model' )
parser.add_argument('--result_dir', type=str, default='./result')
args = parser.parse_args()

def main():
    print(args)
    dcgan = DCGAN(args)
    dcgan.train()

if __name__ == "__main__":
    main()



