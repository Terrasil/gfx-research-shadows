#!/usr/bin/env python3
"""Independent CPU coordinates and domain checks for the GPU PCSS samplers."""
import csv
import json
import math
from pathlib import Path
import statistics
import struct


def f32(x):
    return struct.unpack('f',struct.pack('f',x))[0]


def pcg(x):
    state=(x*747796405+2891336453)&0xffffffff
    word=(((state>>((state>>28)+4))^state)*277803737)&0xffffffff
    return (word>>22)^word


def disk_key(pixel,seed,stream):
    return pcg(pixel[0]^pcg(pixel[1])^pcg(seed)^(((stream+1)*0x85ebca6b)&0xffffffff))


def context(case):
    if case<768:
        count=case%256+1
        return case//256,count,(case*37+3)%count,((case*13)%1024,(case*7)%1024),(case%11)*17,f32(f32(.21)*(case%7))
    group,k=divmod(case-768,400)
    count,offset=(48,0) if k<48 else (96,48) if k<144 else (256,144)
    return group+1,count,k-offset,(145,87),19,f32(.37)


def strata(count):
    rings=max(1,math.floor(math.sqrt(count/math.pi)+.5))
    # Integer nearest-area boundaries; no GPU loop or float boundaries.
    boundaries=[(2*count*r*r+rings*rings)//(2*rings*rings) for r in range(rings+1)]
    return list(zip(boundaries,boundaries[1:]))


def point(index,count,angle,pattern,pixel,seed,stream):
    if pattern==0:
        r=math.sqrt((index+.5)/count);a=index*2.39996322972865+angle
    else:
        key=disk_key(pixel,seed,stream)
        value=(key+index*0x9e3779b9)&0xffffffff
        xi=(pcg(value)>>8)/16777216
        eta=(pcg((value+0x68bc21eb)&0xffffffff)>>8)/16777216
        if pattern==1:
            area,azimuth=xi,eta
        else:
            begin,end=next((a,b) for a,b in strata(count) if a<=index<b)
            area=(begin+(end-begin)*xi)/count
            azimuth=(index-begin+eta)/(end-begin)
        r=math.sqrt(area);a=2*math.pi*azimuth+angle
    return r*math.cos(a),r*math.sin(a)


def verify(path):
    with Path(path).open(newline='') as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=1568 or sorted(int(r['case_id']) for r in rows)!=list(range(1568)):
        raise ValueError('Incomplete GPU sampling validation')
    errors=dict.fromkeys(('vogel','random','stratified'),0.0)
    sets={}
    for row in rows:
        case=int(row['case_id'])
        pattern,count,index,pixel,seed,angle=context(case)
        name=('vogel','random','stratified')[pattern]
        for stream,fields in enumerate((('blocker_x','blocker_y'),('filter_x','filter_y'))):
            actual=tuple(float(row[k]) for k in fields)
            expected=point(index,count,angle,pattern,pixel,seed,stream)
            error=max(abs(x-y) for x,y in zip(actual,expected))
            # Vogel's phase reaches ~612 radians at N=256, so FP32 phase rounding
            # dominates the mathematical double-precision oracle at high indices.
            # Allow separate FP32 multiply/add rounding as well as fused evaluation.
            limit=0.0001 if pattern==0 else 0.000003
            if any(not math.isfinite(v) for v in actual) or error>limit or sum(v*v for v in actual)>1.000002:
                raise ValueError(f'{name} GPU/CPU mismatch in sampling case {case}: {actual} vs {expected}')
            errors[name]=max(errors[name],error)
            if case>=768:
                sets.setdefault((name,count,stream),[]).append(actual)
    statistics_by_set={f'{name}-{count}-stream{stream}':{
        'points':len(values),'mean_radius_squared':statistics.mean(x*x+y*y for x,y in values),
        'centroid': [statistics.mean(v[c] for v in values) for c in range(2)]}
        for (name,count,stream),values in sets.items()}
    return {'cases':len(rows),'max_abs_coordinate_error':errors,'disk_sets':statistics_by_set}


if __name__=='__main__':
    import sys
    print(json.dumps(verify(sys.argv[1]),indent=2))
