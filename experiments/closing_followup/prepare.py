"""Download reusable ETF inputs without computing strategy performance."""
import argparse
from pathlib import Path
from experiments.strategy_search.storage import read
from .config import SCOPES
from .data import stock_market, prepare_etfs, prepare_etf_quotes


def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--prior',required=True)
    p.add_argument('--output',required=True);p.add_argument('--env-file',default='.env');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    protected=read(Path(a.prior)/'protected_sessions.json')
    protocol={'binding':dict(partitions=read(Path(a.reference)/'protocol.json')['partitions'],
        reserved_session_dates=protected['reserved_session_dates'])}
    for scope in SCOPES:
        m,_=stock_market(a.reference,scope,protocol)
        etf=prepare_etfs(m,a.output,a.env_file,a.workers)
        prepare_etf_quotes(etf,a.output,a.env_file,a.workers)
        print('INPUTS_READY '+scope,flush=True)


if __name__=='__main__':main()
