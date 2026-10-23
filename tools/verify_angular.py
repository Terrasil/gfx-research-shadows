#!/usr/bin/env python3
"""Independent circle-segment, ray-box, and separable convolution GPU oracles."""
import csv
import math
from pathlib import Path

def dot(a,b):return sum(x*y for x,y in zip(a,b))
def norm(v):
    s=math.sqrt(dot(v,v));return tuple(x/s for x in v)
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def ray_box(origin,direction,lo,hi):
    near,far=0.0,math.inf
    for o,d,a,b in zip(origin,direction,lo,hi):
        if abs(d)<1e-12:
            if o<a or o>b:return False
        else:
            t0,t1=sorted(((a-o)/d,(b-o)/d));near=max(near,t0);far=min(far,t1)
            if near>far:return False
    return far>1e-6

def verify(path):
    with Path(path).open(newline='') as stream:rows=list(csv.DictReader(stream))
    if len(rows) not in (256,416):raise ValueError('Incomplete angular fixtures')
    planes=0;margins=0;masks=0;max_error=0.0;max_mask_error=0.0;seen=set();mask_levels=set()
    sun=norm((.55,1,.35));u=norm(cross((0,0,1),sun));v=cross(sun,u)
    boxes=[((-1.8,4.5,-1.6),(1.8,5,1.6)),((-5.2,.15,-3.15),(-.6,.7,-2.75))]
    for row in rows:
        key=(row['kind'],int(row['queries']),int(row['case_id']))
        if key in seen:raise ValueError('Duplicate angular fixture')
        seen.add(key);actual=float(row['value'])
        if not math.isfinite(actual):raise ValueError('Non-finite angular result')
        if row['kind']=='plane':
            a,b,c=(float(row[k]) for k in ('a','b','c'))
            d=max(-1,min(1,c/math.hypot(a,b)))
            expected=.5+(math.asin(d)+d*math.sqrt(max(0,1-d*d)))/math.pi
            error=abs(actual-expected);max_error=max(max_error,error);planes+=1
            if error>8e-5:raise ValueError(f'Circular coverage mismatch: {key}, {actual} vs {expected}')
        elif row['kind']=='margin':
            origin=tuple(float(row[k]) for k in ('ox','oy','oz'));p=tuple(float(row[k]) for k in ('px','py'))
            direction=norm(tuple(sun[k]+math.tan(math.radians(1.5))*(u[k]*p[0]+v[k]*p[1]) for k in range(3)))
            blocked=any(ray_box(origin,direction,lo,hi) for lo,hi in boxes)
            if abs(actual)>1e-5 and (actual<0)!=blocked:raise ValueError(f'Ray-box classification mismatch: {key}')
            margins+=1
        elif row['kind']=='masks':
            count=int(row['queries']);ring=count if count==3 else count-1
            radius,rotation=(float(row[k]) for k in ('px','py'))
            points=[] if count==3 else [(0,0)]
            points += [(radius*math.cos(rotation+2*math.pi*k/ring),radius*math.sin(rotation+2*math.pi*k/ring)) for k in range(ring)]
            origin=tuple(float(row[k]) for k in ('ox','oy','oz'))
            normal=norm(tuple(float(row[k]) for k in ('a','b','c')))
            lit=0.0;total=0.0;hits=0;tangent=math.tan(math.radians(1.5))
            for p in points:
                direction=norm(tuple(sun[k]+tangent*(u[k]*p[0]+v[k]*p[1]) for k in range(3)))
                weight=max(0,dot(normal,direction))*(1+tangent*tangent*dot(p,p))**-1.5
                hit=any(ray_box(origin,direction,lo,hi) for lo,hi in boxes)
                lit += 0 if hit else weight;total += weight;hits += int(hit)
            expected=lit/total if total>1e-12 else 1.0
            error=abs(actual-expected);max_mask_error=max(max_mask_error,error)
            if error>2e-5:raise ValueError(f'Ray-mask blend mismatch: {key}, {actual} vs {expected}')
            mask_levels.add((hits==0,hits==count));masks+=1
        else:raise ValueError('Unknown angular fixture')
    if planes!=160 or margins!=96:raise ValueError('Incomplete angular fixture kinds')
    if masks not in (0,160) or len(rows)!=256+masks:raise ValueError('Incomplete ray-mask fixtures')
    if masks and len(mask_levels)!=3:raise ValueError('Ray-mask fixtures need lit, shadowed, and penumbra cases')
    return {'cases':len(rows),'circle_planes':planes,'ray_box_signs':margins,'max_coverage_error':max_error,
            'ray_mask_cases':masks,'max_mask_error':max_mask_error}

def kernel(x,mode):
    x=abs(x)
    if mode in (1,3):
        if x>=1:return 0.0
        return 1-x if mode==1 else 1-3*x*x+2*x*x*x
    if mode==4:
        return (4-6*x*x+3*x**3)/6 if x<1 else (2-x)**3/6 if x<2 else 0.0
    return 1.5*x**3-2.5*x*x+1 if x<1 else -.5*x**3+2.5*x*x-4*x+2 if x<2 else 0.0

def verify_filter(path):
    with Path(path).open(newline='') as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=1024:raise ValueError('Incomplete angular filter fixtures')
    max_error=0.0;seen=set();h=.375
    for row in rows:
        mode,x,y=(int(row[k]) for k in ('mode','x','y'));key=(mode,x,y)
        if mode not in (1,3,4,5) or x not in range(16) or y not in range(16) or key in seen:raise ValueError('Invalid angular filter case')
        seen.add(key)
        if y==15:expected=1.0
        else:
            weights={i:.5*(kernel(i-h,mode)+kernel(i+h,mode)) for i in range(-3,4)}
            total=0.0;weight=0.0
            # Direct 2D convolution with the analytically combined 1D kernels,
            # independent of the GPU's four reconstruction locations.
            for dy,wy in weights.items():
                for dx,wx in weights.items():
                    xx=max(0,min(15,x+dx));yy=max(0,min(15,y+dy))
                    if yy==15 or (xx<8)!=(x<8):continue
                    w=wx*wy;total+=w*((xx*13+yy*7+xx*yy*3)%97)/96;weight+=w
            expected=max(0,min(1,total/weight))
        actual=float(row['value']);error=abs(actual-expected);max_error=max(max_error,error)
        if not math.isfinite(actual) or error>5e-5:raise ValueError(f'Angular reconstruction filter mismatch: {key}: {actual} vs {expected}')
    return {'cases':len(rows),'max_abs_error':max_error,'depth_step_guard':True}

if __name__=='__main__':
    import json,sys
    print(json.dumps({'angular':verify(sys.argv[1]),'filter':verify_filter(sys.argv[2])},indent=2))
