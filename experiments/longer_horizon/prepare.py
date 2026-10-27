import argparse
from .data import download_auctions

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--prior',required=True)
    p.add_argument('--output',required=True);p.add_argument('--env-file',default='.env');a=p.parse_args()
    download_auctions(a.reference,a.prior,a.output,a.env_file)

if __name__=='__main__':main()
