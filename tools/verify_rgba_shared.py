#!/usr/bin/env python3
"""CPU projection and quadrature oracle for the shared world-space RGBA PCF."""
import csv
import math
from pathlib import Path
from verify_rgba_pcf import f32, disk, depth


EXTENTS = [f32(v) for v in (1, 1.3, 1.8, 2.5)]
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def unit(v):
    length=math.sqrt(dot(v,v))
    return tuple(x/length for x in v)
def cross(a,b):return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def rotate(v,c):
    a=f32(c*f32(.21));b=f32(c*f32(.1))
    x=math.cos(b)*v[0]+math.sin(b)*v[2];y=v[1];z=-math.sin(b)*v[0]+math.cos(b)*v[2]
    return (math.cos(a)*x-math.sin(a)*y,math.sin(a)*x+math.cos(a)*y,z)
def smooth(a,b,v):
    t=max(0,min(1,(v-a)/(b-a)))
    return t*t*(3-2*t)


def expected(row):
    x,y=int(row['x']),int(row['y']);adaptive=int(row['adaptive']);samples=int(row['samples'])
    mix,cascades,kernel=(int(row[k]) for k in ('mix','cascades','smoothing'))
    strength,limit=(float(row[k]) for k in ('strength','limit'))
    world=(f32(f32(x+f32(.37))/4-2),f32(f32(y+f32(.63))/4-2),f32(f32(.1)+(x+y)%3*f32(.04)))
    n=unit(((x%3-1)*f32(.15),(y%3-1)*f32(.2),1))
    if y==14:n=unit((1,-f32(2/3),0))
    if y==15:n=(0,1,0)
    gradients=[]
    for c,e in enumerate(EXTENTS):
        rn=rotate(n,c)
        gradients.append((e*rn[0]/(8*rn[2]),e*rn[1]/(8*rn[2])) if abs(-4*rn[2])>1e-7 else (0,0))
    def project(p,c):
        q=rotate(p,c);e=EXTENTS[c]
        return (q[0]/e+f32(c*f32(.15)),q[1]/e,.5-q[2]/8)
    def compare(tx,ty,c,light):
        if not 0<=tx<16 or not 0<=ty<16:return 1.0
        d=depth(tx,ty,c);g=gradients[c]
        receiver=light[2]+g[0]*((tx+.5)/8-1-light[0])+g[1]*((ty+.5)/8-1-light[1])-f32(.00015)
        return float(d>=1 or receiver<=d)
    def visibility(light,c):
        u,v=((q*.5+.5)*16 for q in light[:2])
        if kernel==0:return compare(math.floor(u),math.floor(v),c,light)
        bx,by=math.floor(u-.5),math.floor(v-.5);fx,fy=u-.5-bx,v-.5-by
        return sum((fx if dx else 1-fx)*(fy if dy else 1-fy)*compare(bx+dx,by+dy,c,light) for dy in (0,1) for dx in (0,1))
    def query(p):
        lights=[project(p,c) for c in range(4)]
        margins=[1-max(abs(l[0]),abs(l[1])) for l in lights]
        valid=[margin>0 and 0<=l[2]<=1 for margin,l in zip(margins,lights)]
        if not any(valid):return 1.0,valid
        values=[visibility(l,c) if valid[c] else 1.0 for c,l in enumerate(lights)]
        w=[f32(.35)]+[(1-f32(.35))/3]*3
        total=sum(a for a,b in zip(w,valid) if b)
        weights=[a/total if b else 0 for a,b in zip(w,valid)]
        active=[v for v,w in zip(values,weights) if w>0]
        angular=sum(w*v for w,v in zip(weights,values)) if mix==0 else min(active) if mix==1 else math.prod(active)
        if not cascades:return angular,valid
        remaining=1.0;cascade=0
        for c in range(4):
            if not valid[c]:continue
            take=smooth(0,f32(.2),margins[c]) if any(valid[c+1:]) else 1
            cascade+=remaining*take*values[c];remaining*=1-take
        cascade+=remaining
        return (1-f32(.6))*cascade+f32(.6)*angular,valid
    base,valid=query(world)
    radius=0.0
    if strength>0 and limit>0 and any(valid):
        radius=limit*strength
        if adaptive:
            r=[f32(v)*((x+y)%3+1)*e for v,e in zip((.03,.06,.09,.12),EXTENTS)]
            radius=min(radius,min(v for v,ok in zip(r,valid) if ok))
    if radius<=1e-8:return base,radius,base
    sun=unit((f32(.2),f32(.3),1));den=dot(n,sun)
    if abs(den)>=.1:
        tangent=unit(cross((0,1,0),sun));bitangent=cross(sun,tangent)
    else:
        helper=(0,0,1) if abs(n[2])<.9 else (0,1,0)
        tangent=unit(cross(helper,n));bitangent=cross(n,tangent)
    results=[]
    for j in range(samples):
        dx,dy=disk(j,samples)
        delta=tuple(radius*(dx*a+dy*b) for a,b in zip(tangent,bitangent))
        if abs(den)>=.1:
            correction=dot(n,delta)/den
            delta=tuple(a-b*correction for a,b in zip(delta,sun))
        value,coverage=query(tuple(a+b for a,b in zip(world,delta)))
        if any(coverage):results.append(value)
    return (sum(results)/len(results) if results else base),radius,base


def verify(path):
    rows=list(csv.DictReader(Path(path).open(newline='')))
    keys={(int(r['config']),int(r['x']),int(r['y'])) for r in rows}
    if len(rows)!=6656 or keys!={(c,x,y) for c in range(26) for x in range(16) for y in range(16)}:
        raise ValueError('Incomplete shared PCF fixture matrix')
    errors=[0.0,0.0,0.0];plane_error=0.0
    for row in rows:
        reference=expected(row)
        for k,key in enumerate(('visibility','radius','base')):
            value=float(row[key]);error=abs(value-reference[k])
            if not math.isfinite(value) or value<0 or (key!='radius' and value>1.000001) or error>2e-4:
                raise ValueError(f'Shared PCF differs from CPU: config={row["config"]} xy={row["x"]},{row["y"]} {key} error={error}')
            errors[k]=max(errors[k],error)
        p=float(row['plane_error']);plane_error=max(plane_error,p)
        if not math.isfinite(p) or p>1e-6:raise ValueError('Shared sample left the receiver plane')
    return {'cases':len(rows),'max_visibility_error':errors[0],'max_radius_error':errors[1],
            'max_center_error':errors[2],'max_receiver_plane_error':plane_error}


if __name__=='__main__':
    import json
    import sys
    print(json.dumps(verify(sys.argv[1]),indent=2))
