#!/usr/bin/env python3
"""Independent CPU oracle for channel PCF, blocker search and receiver-plane correction."""
import csv
import math
import struct
from pathlib import Path


def f32(v):return struct.unpack('f',struct.pack('f',v))[0]
def disk(i,n):
    r=math.sqrt((i+.5)/n);phi=i*2.39996322973
    return (r*math.cos(phi),r*math.sin(phi))
def depth(x,y,c):return 1.0 if (x+y+c)%11==0 else f32(((x*13+y*7+x*y*3+c*11)%37+25)/100)
def covered(p):return -1<=p[0]<1 and -1<=p[1]<1 and 0<=p[2]<=1


def expected(row):
    x,y=int(row['x']),int(row['y']);mode=int(row['mode']);kernel=int(row['smoothing']);n=int(row['samples']);strength=float(row['strength'])
    extent=float(row.get('extent',1));world=int(row.get('world_units',0));moment_gap=int(row.get('moment_gap',0))
    origin=((x+f32(.37))/8-1,(y+f32(.63))/8-1,f32(.5+((x+y)%3)*f32(.001)))
    gradient=((x%3-1)*f32(.003),(y%3-1)*f32(.003));c=(x+y)%4
    def gap(p):
        if not -1<=p[0]<1 or not -1<=p[1]<1:return 0.0
        tx,ty=math.floor((p[0]*.5+.5)*16),math.floor((p[1]*.5+.5)*16);d=depth(tx,ty,c)
        if d>=1:return 0.0
        center=((tx+.5)/8-1,(ty+.5)/8-1)
        g=origin[2]+sum(a*(b-o) for a,b,o in zip(gradient,center,origin))-d-f32(.00015)
        return max(g,0)*80
    def visibility(p):
        if not covered(p):return 1.0
        def compare(tx,ty):
            if not 0<=tx<16 or not 0<=ty<16:return 1.0
            d=depth(tx,ty,c);center=((tx+.5)/8-1,(ty+.5)/8-1)
            receiver=p[2]+gradient[0]*(center[0]-p[0])+gradient[1]*(center[1]-p[1])-f32(.00015)
            return float(d>=1 or receiver<=d)
        uv=((p[0]*.5+.5)*16,(p[1]*.5+.5)*16)
        if kernel==0:return compare(math.floor(uv[0]),math.floor(uv[1]))
        t=(uv[0]-.5,uv[1]-.5);base=(math.floor(t[0]),math.floor(t[1]));f=(t[0]-base[0],t[1]-base[1])
        return sum((f[0] if dx else 1-f[0])*(f[1] if dy else 1-f[1])*compare(base[0]+dx,base[1]+dy) for dy in range(2) for dx in range(2))
    def estimate_gap():
        g=gap(origin)
        if g<=0:
            search=min(max(origin[2],0)*80*f32(.026)/extent,.5)
            gs=[gap((origin[0]+dx*search,origin[1]+dy*search)) for dx,dy in (disk(i,8) for i in range(8))]
            gs=[v for v in gs if v>0]
            g=math.exp(sum(math.log(max(v,1e-8)) for v in gs)/len(gs)) if gs else 0
        return g
    radius=0.0;blocker_gap=0.0
    if strength>0:
        if mode==1:
            radius=(f32(float(row['world_radius_m']))/extent if world else .125*2)*strength
            if moment_gap:blocker_gap=estimate_gap()
        else:
            blocker_gap=estimate_gap()
            radius=min(blocker_gap*f32(.026)/extent*strength,.5)
    g=gap(origin);support=0.0
    if mode==2 and strength>0 and g>0:
        ratio=g*f32(.026)/(f32(.02)+(y%3)*f32(.01));t=max(0.0,min(1.0,(ratio-.75)/.75))
        support=(1-visibility(origin))*(1-t*t*(3-2*t))
    extra=(blocker_gap,) if 'blocker_gap' in row else ()
    if radius<=1e-8:return (visibility(origin),radius,support)+extra
    vals=[]
    for i in range(n):
        dx,dy=disk(i,n);dx*=radius;dy*=radius;p=(origin[0]+dx,origin[1]+dy,origin[2]+gradient[0]*dx+gradient[1]*dy)
        if covered(p):vals.append(visibility(p))
    return (sum(vals)/len(vals) if vals else visibility(origin),radius,support)+extra


def verify(path):
    rows=list(csv.DictReader(Path(path).open(newline='')))
    count=18432 if rows and 'blocker_gap' in rows[0] else 6144
    if len(rows)!=count or len({(r['config'],r['x'],r['y']) for r in rows})!=count:raise ValueError('Incomplete RGBA PCF validation')
    keys=('visibility','radius','sharp_support','blocker_gap') if count==18432 else ('visibility','radius','sharp_support')
    errors=[0.0]*len(keys)
    for row in rows:
        ref=expected(row)
        for k,key in enumerate(keys):
            value=float(row[key]);delta=abs(value-ref[k])
            # The geometric mean is sensitive to nearly zero positive gaps.
            # Gap is reported in meters (80 m depth range); radius is normalized.
            tolerance=5e-4 if key=='blocker_gap' else 1e-4
            if not math.isfinite(value) or value<0 or (key in ('visibility','sharp_support') and value>1) or delta>tolerance:raise ValueError(f'RGBA PCF differs from CPU: config={row["config"]} xy={row["x"]},{row["y"]} {key} error={delta}')
            errors[k]=max(errors[k],delta)
    result={'cases':len(rows),'max_visibility_error':errors[0],'max_radius_error':errors[1],'max_sharp_support_error':errors[2]}
    if len(errors)==4:result['max_blocker_gap_error']=errors[3]
    return result
