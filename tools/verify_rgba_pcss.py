#!/usr/bin/env python3
"""Independent depth-map oracle for per-channel directional PCSS."""
import csv
import math
import struct
from pathlib import Path


def f32(value):
    return struct.unpack('f',struct.pack('f',value))[0]


def expected(row):
    x,y=int(row['x']),int(row['y']);channel=(x+y)%4
    fixture=int(row['fixture']);warp=float(row['warp']);extent=4**channel if int(row['layout']) else 1
    strength=float(row['strength']);cap=float(row['cap_texels']);limit=float(row['search_limit_m'])
    origin=((x+f32(.37))/8-1,(y+f32(.63))/8-1,f32(.5+((x+y)%3)*f32(.001)))
    gradient=((x%3-1)*f32(.0003),(y%3-1)*f32(.0003))
    def project(p):return tuple(v*(1+warp)/(1+warp*abs(v)) for v in p)
    def unproject(p):return tuple(v/(1+warp-warp*abs(v)) for v in p)
    def covered(p):
        u=project(p[:2]);return all(-1<=v<1 for v in u) and 0<=p[2]<=1
    def stored(tx,ty):
        d=1.0 if (tx+ty+channel)%11==0 else ((tx*13+ty*7+tx*ty*3+channel*11)%37+25)/100
        if fixture==1:d=1
        if fixture==2:d=.31
        if fixture==3:d=.45 if tx<7 else .4985 if tx==8 else 1
        if fixture==4:
            if channel==0:d=1
            if channel==2:d=.4
            if channel==3:d=.4985 if tx==8 else 1
        return f32(d)
    def texel_point(tx,ty):return unproject(((tx+.5)/8-1,(ty+.5)/8-1))
    def gap(p):
        u=project(p)
        if any(v< -1 or v>=1 for v in u):return 0.0
        tx,ty=(math.floor((v*.5+.5)*16) for v in u);d=stored(tx,ty)
        if d>=1:return 0.0
        point=texel_point(tx,ty)
        receiver=origin[2]+sum(a*(b-o) for a,b,o in zip(gradient,point,origin))
        return max(receiver-d-f32(.00015),0)*80
    def visibility(p):
        if not covered(p):return 1.0
        def compare(tx,ty):
            if not 0<=tx<16 or not 0<=ty<16:return 1.0
            d=stored(tx,ty);point=texel_point(tx,ty)
            receiver=p[2]+sum(a*(b-o) for a,b,o in zip(gradient,point,p))-f32(.00015)
            return float(d>=1 or receiver<=d)
        u=tuple((v*.5+.5)*16 for v in project(p[:2]))
        if int(row['smoothing'])==0:return compare(*(math.floor(v) for v in u))
        t=tuple(v-.5 for v in u);base=tuple(math.floor(v) for v in t);fraction=tuple(v-b for v,b in zip(t,base))
        return sum((fraction[0] if dx else 1-fraction[0])*(fraction[1] if dy else 1-fraction[1])*compare(base[0]+dx,base[1]+dy) for dy in range(2) for dx in range(2))
    def disk(i,n):
        radius=math.sqrt((i+.5)/n);angle=i*2.39996322973
        return radius*math.cos(angle),radius*math.sin(angle)
    center_gap=gap(origin[:2]);base=visibility(origin)
    mean_gap=radius=search=0.0;count=0
    active=strength>0 and cap>0
    if active:
        search=min(max(origin[2],0)*80*f32(.026),limit)/extent
        n=int(row['blocker_samples']) if search>1e-8 else 1
        offsets=[(0,0)]+[tuple(v*search for v in disk(i,n-1)) for i in range(n-1)]
        gaps=[gap((origin[0]+dx,origin[1]+dy)) for dx,dy in offsets]
        gaps=[g for g in gaps if g>0];count=len(gaps)
        if count:
            mean_gap=math.exp(sum(math.log(max(g,1e-8)) for g in gaps)/count) if int(row['mean']) else sum(gaps)/count
        u=project(origin[:2]);a=unproject((u[0]+.125,u[1]));b=unproject((u[0],u[1]+.125))
        footprint=min(math.dist(a,origin[:2]),math.dist(b,origin[:2]))
        radius=min(mean_gap*f32(.026)/extent*strength,footprint*cap)
    result=base
    if active and not count:result=1.0
    elif radius>1e-8:
        samples=[]
        for i in range(int(row['filter_samples'])):
            dx,dy=disk(i,int(row['filter_samples']));dx*=radius;dy*=radius
            point=(origin[0]+dx,origin[1]+dy,origin[2]+gradient[0]*dx+gradient[1]*dy)
            if covered(point):samples.append(visibility(point))
        result=sum(samples)/len(samples) if samples else base
    sharp=0.0
    if active and center_gap>0:
        support=center_gap*f32(.026)/extent/(f32(.02)+(y%3)*f32(.01))
        t=max(0.0,min(1.0,(support-.75)/.75));sharp=(1-base)*(1-t*t*(3-2*t))
    return result,mean_gap,radius,count,search,center_gap,sharp


def verify(path):
    with Path(path).open(newline='') as stream:rows=list(csv.DictReader(stream))
    count=72*256
    if len(rows)!=count or {(int(r['config']),int(r['x']),int(r['y'])) for r in rows}!={(c,x,y) for c in range(72) for x in range(16) for y in range(16)}:
        raise ValueError('Incomplete per-channel PCSS validation')
    keys=('visibility','mean_gap','radius','blockers','search_radius','center_gap','sharp_support')
    errors=dict.fromkeys(keys,0.0);mixed=0
    for row in rows:
        reference=expected(row)
        for key,ref in zip(keys,reference):
            value=float(row[key]);delta=abs(value-ref)
            tolerance=0 if key=='blockers' else 5e-4 if key in ('mean_gap','center_gap') else 1e-4
            if not math.isfinite(value) or value<0 or (key in ('visibility','sharp_support') and value>1.000001) or delta>tolerance:
                raise ValueError(f'Per-channel PCSS differs from CPU: config={row["config"]} xy={row["x"]},{row["y"]} {key} error={delta}')
            errors[key]=max(errors[key],delta)
        if reference[3]>1 and reference[5]>0 and reference[1]>reference[5]*1.5:mixed+=1
    if not mixed:raise ValueError('PCSS validation never averaged beyond a center blocker')
    return {'cases':count,'max_abs_error':errors,'mixed_center_blocker_cases':mixed}
