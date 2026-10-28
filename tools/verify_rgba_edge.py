#!/usr/bin/env python3
"""Independent disk-CDF reference for isolated and overlapping infinite strips."""
import csv
import math
from pathlib import Path


def cdf(t):
    t=max(-1.0,min(1.0,t))
    return .5+(math.asin(t)+t*math.sqrt(max(0,1-t*t)))/math.pi


def reference(fixture,x):
    angle=math.tan(math.radians(1.5))
    blockers=[(-math.inf,0,4)] if fixture==0 else [(-.03,.03,.1)] if fixture==1 else [(-.15,0,4),(-.03,.03,.1)]
    intervals=[]
    for lo,hi,gap in blockers:
        a,b=max(-1,(lo-x)/(gap*angle)),min(1,(hi-x)/(gap*angle))
        if b>a:intervals.append((a,b))
    merged=[]
    for a,b in sorted(intervals):
        if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
        else:merged.append((a,b))
    return 1-sum(cdf(b)-cdf(a) for a,b in merged)


def verify(path):
    rows=list(csv.DictReader(Path(path).open(newline='')))
    if len(rows)!=384:raise ValueError('Missing edge reconstruction fixtures')
    report={}
    for fixture in range(3):
        subset=[r for r in rows if int(r['fixture'])==fixture]
        if len(subset)!=128:raise ValueError('Missing edge fixture samples')
        errors={'raw':[],'reconstructed':[]};core=[]
        for r in subset:
            x=float(r['x']);ref=reference(fixture,x)
            for key in errors:
                value=float(r[key])
                if not math.isfinite(value) or not 0<=value<=1:raise ValueError('Invalid edge visibility')
                errors[key].append((value-ref)**2)
            if fixture>0 and abs(x)<.03-.1*math.tan(math.radians(1.5)):
                core.append(float(r['reconstructed']))
        report[str(fixture)]={key+'_rmse':math.sqrt(sum(vals)/len(vals)) for key,vals in errors.items()}
        report[str(fixture)]['near_core_max_visibility']=max(core,default=0)
    if report['0']['reconstructed_rmse']>report['0']['raw_rmse']*.25:
        raise ValueError('Continuous single-edge reconstruction regressed')
    if any(report[str(i)]['near_core_max_visibility']>.001 for i in (1,2)):
        raise ValueError('Near shadow core was brightened')
    return {'cases':384,'fixtures':report}
