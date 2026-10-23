#!/usr/bin/env python3
"""Compare the low blocker's exact umbra with and without the upper blocker."""
import csv
import json
import math
from pathlib import Path
from analyze_results import read_pfm

def compare(mixed, low):
    mixed,low=Path(mixed),Path(low)
    manifests=[json.loads((p/'run_manifest.json').read_text()) for p in (mixed,low)]
    for key in ('custom_camera','camera_yaw','camera_pitch','camera_distance','camera_target','warp','map_backend'):
        if manifests[0][key]!=manifests[1][key]:raise ValueError('Cameras/maps differ')
    with (mixed/'raw.csv').open() as f: rows=list(csv.DictReader(f))
    if any(r['scene']!='mixed-overlap' for r in rows):raise ValueError('Expected mixed-overlap scene')
    with (low/'raw.csv').open() as f: low_rows=list(csv.DictReader(f))
    if any(r['scene']!='single-low' for r in low_rows):raise ValueError('Expected single-low scene')
    w,h,a=read_pfm(mixed/'captures/c0-arr.pfm');lw,lh,b=read_pfm(low/'captures/c0-arr.pfm')
    if (w,h)!=(lw,lh):raise ValueError('Capture dimensions differ')
    _,_,m=read_pfm(mixed/'captures/c0-receiver.pfm');_,_,n=read_pfm(low/'captures/c0-receiver.pfm')
    core=[i for i in range(len(a)) if m[i]>.5 and n[i]>.5 and a[i]<=.01 and b[i]<=.01]
    if not core:raise ValueError('No common near-blocker umbra')
    _,_,global_mixed=read_pfm(mixed/'captures/c0-geometric.pfm')
    _,_,global_low=read_pfm(low/'captures/c0-geometric.pfm')
    affected=[i for i in core if global_mixed[i]>global_low[i]+.05]
    result=[]
    for row in rows:
        method=row['method'];_,_,x=read_pfm(mixed/f'captures/c0-{method}.pfm');_,_,y=read_pfm(low/f'captures/c0-{method}.pfm')
        result.append({'method':method,'near_core_pixels':len(core),
                       'near_core_rmse':math.sqrt(sum((x[i]-a[i])**2 for i in core)/len(core)),
                       'brightening_after_upper_mean':sum(max(0,x[i]-y[i]) for i in core)/len(core),
                       'affected_pixels':len(affected),
                       'affected_mean_visibility':sum(x[i] for i in affected)/len(affected) if affected else 0.0})
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mixed');p.add_argument('low');args=p.parse_args()
    print(json.dumps(compare(args.mixed,args.low),indent=2))
