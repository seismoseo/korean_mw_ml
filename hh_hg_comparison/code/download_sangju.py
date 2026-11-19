"""Obtain both original NECIS sensor families for a co-located comparison."""
from pathlib import Path
import asyncio,logging,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parents[1]))
from necis.config import NECISConfig
from necis.events import run_events

async def main():
    logging.basicConfig(level=logging.WARNING)
    inputs=ROOT/'inputs';inputs.mkdir(exist_ok=True)
    catalog=inputs/'Sangju_download_catalog.csv'
    catalog.write_text('event_id,datetime,magnitude\n20261117141017,2026-11-17 14:10:17,3.6\n')
    cfg=NECISConfig.from_env(download_dir=inputs/'necis',capture_api=False,headless=True)
    print('Requesting original Sangju acceleration and velocity records',flush=True)
    await run_events(cfg,catalog,['SAJB','OKCB','JECB'],components=['Z','N','E'],
        data_types=['a','v'],convert_sac=False,out_root=inputs/'events')
    files=list((inputs/'events').rglob('*'))
    print('Finished; waveform files',sum(p.is_file() for p in files),flush=True)

if __name__=='__main__':asyncio.run(main())
