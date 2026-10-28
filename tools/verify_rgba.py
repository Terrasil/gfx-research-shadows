#!/usr/bin/env python3
"""CPU oracle for normalized angular mixtures and cascade edge transitions."""
import csv
import math
from pathlib import Path


def expected(case,blend=0.0,extended=False):
    if extended:
        case%=1312
    if extended and case>=1280:
        fixture=(case-1280)%8;mode=(0,2,3,4)[(case-1280)//8]
        values=[(0,1,1,1),(0,1,1,1),(0,0,0,0),(1,1,1,1),(.37,)*4,
                (1e-9,.5,.8,1),(.3,0,0,0),(.2,.4,.6,.8)][fixture]
        available=[] if fixture==7 else list(range(4))
        center=0.0 if fixture==1 else 1.0 if fixture==6 else .25
        margins=[1.0]*4;cascades=False
    else:
        values=[((case*7+k*5)%17)/16 for k in range(4)]
        margins=[((case*3+k*7)%11)/10 for k in range(4)]
        available=[k for k in range(4) if case%16&(1<<k)]
        center=((case//7)%3)*.5;mode=case//256;cascades=(case//128)%2!=0
    if not available:
        return 1.0
    weights=[center if k==0 else (1-center)/3 for k in available]
    total=sum(weights)
    weights=[v/total for v in weights] if total>1e-7 else [1/len(available)]*len(available)
    selected=[values[k] for k,w in zip(available,weights) if w>0]
    arithmetic=sum(values[k]*w for k,w in zip(available,weights))
    geometric=0.0 if any(v==0 for v in selected) else math.exp(sum(w*math.log(values[k]) for k,w in zip(available,weights) if w>0))
    angular=(arithmetic,min(selected),math.prod(selected),geometric,(1-blend)*arithmetic+blend*geometric)[mode]
    if not cascades:
        return angular
    width=0.0 if (case//64)%2==0 else .3
    residual=1.0
    cascade=0.0
    for index,k in enumerate(available):
        fraction=1.0
        if index<len(available)-1 and width>0:
            t=max(0.0,min(1.0,margins[k]/width))
            fraction=t*t*(3-2*t)
        cascade+=residual*fraction*values[k]
        residual*=1-fraction
    cascade+=residual
    mix=(0,.25,.75,1)[(case//16)%4]
    return cascade*(1-mix)+angular*mix


def verify(path):
    with Path(path).open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    extended=bool(rows and 'geometric_blend' in rows[0]);count=3936 if extended else 768
    if len(rows)!=count or {int(r['case_id']) for r in rows}!=set(range(count)):
        raise ValueError('Missing RGBA mixture cases')
    error=0.0
    for row in rows:
        blend=float(row.get('geometric_blend',0));case=int(row['case_id'])
        if extended and blend!=(0,.5,1)[case//1312]:raise ValueError('Wrong geometric blend fixture')
        value=float(row['gpu']);delta=abs(value-expected(case,blend,extended))
        if not math.isfinite(value) or not 0<=value<=1 or delta>2e-6:
            raise ValueError('RGBA mixture differs from CPU oracle')
        error=max(error,delta)
    return dict(cases=count,max_abs_error=error)
