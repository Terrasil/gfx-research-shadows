#!/usr/bin/env python3
"""Independent scalar 2D Gaussian oracle for encoded bases and reconstruction."""
import csv
import math
from functools import lru_cache
from pathlib import Path
import struct

def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]

@lru_cache(None)
def expected_basis(method, x, y, radius=3):
    sigma = max(radius*.5, .5)
    weights = [math.exp(-.5*i*i/(sigma*sigma)) for i in range(-radius,radius+1)]
    norm = sum(weights)**2
    result = [0.0]*16
    # Direct 7x7 integration, independent of the GPU's two 1D passes.
    for row,wy in zip(range(-radius,radius+1),weights):
        for col,wx in zip(range(-radius,radius+1),weights):
            a,b = min(31,max(0,x+col)),min(31,max(0,y+row))
            d = f32(((a*13+b*7+a*b*3)%89+5)/100)
            if method=='gsm':
                if (a+b)%11==0:d=1.0
                elif (a+b)%13==0:d=0.0
            w = wx*wy/norm
            if method == 'gsm':
                result[0] += w*math.log1p(d)
            elif method == 'vsm':
                result[0] += w*d; result[1] += w*d*d
            elif method == 'esm':
                result[0] += w*math.exp(80*(d-1))
            else:
                for k in range(8):
                    q = math.pi*(2*k+1)*(.05+.9*d)
                    result[2*k] += w*math.cos(q)
                    result[2*k+1] += w*math.sin(q)
    return result

def verify(path):
    with Path(path).open(newline='') as f: rows = list(csv.DictReader(f))
    methods=('vsm','csm','esm','gsm') if len(rows)==1536 else ('vsm','csm','esm')
    expected={(m,i,3,0) for m in ('vsm','csm','esm') for i in range(128)}
    if 'gsm' in methods:
        expected |= {('gsm',i,radius,power) for i in range(128) for radius in (0,3,16) for power in (1,64,128)}
    actual={(r['method'],int(r['case_id']),int(r.get('radius',3)),float(r.get('sharpness',0))) for r in rows}
    if len(rows)!=len(expected) or actual!=expected:
        raise ValueError('Incomplete prefilter fixtures')
    errors = {m:0.0 for m in methods}
    visibility_error = dict(errors)
    for row in rows:
        method,i = row['method'],int(row['case_id'])
        x,y,t = (i*7)%32,(i*13+i//32)%32,(i%17)/16
        if (int(row['x']),int(row['y'])) != (x,y) or abs(float(row['t'])-t)>1e-7:
            raise ValueError('Incorrect fixture coordinate')
        radius=int(row.get("radius",3))
        basis = expected_basis(method,x,y,radius)
        for c,want in enumerate(basis):
            actual = float(row[f'b{c}'])
            tolerance = max(1e-36,abs(want)*1e-4) if method=='esm' else 2e-5
            if not math.isfinite(actual) or abs(actual-want)>tolerance:
                raise ValueError(f'{method} encoded/blurred basis mismatch at {i}/{c}: {actual} vs {want}')
            errors[method] = max(errors[method],abs(actual-want))
        if method == 'gsm':
            # Independent weighted product, rather than the shader's log response.
            sigma=max(radius*.5,.5)
            weights=[math.exp(-.5*j*j/(sigma*sigma)) for j in range(-radius,radius+1)]
            norm=sum(weights)**2
            geometric_mean=1.0
            for dy,wy in zip(range(-radius,radius+1),weights):
                for dx,wx in zip(range(-radius,radius+1),weights):
                    a,b=min(31,max(0,x+dx)),min(31,max(0,y+dy))
                    depth=f32(((a*13+b*7+a*b*3)%89+5)/100)
                    if (a+b)%11==0:depth=1.0
                    elif (a+b)%13==0:depth=0.0
                    geometric_mean *= (1.0+depth)**(wx*wy/norm)
            value=min(1.0,geometric_mean/(1.0+t))**float(row['sharpness'])
        elif method == 'vsm':
            variance = max(basis[1]-basis[0]**2,1e-7)
            value = 1 if t<=basis[0] else variance/(variance+(t-basis[0])**2)
        elif method == 'esm':
            value = min(1,basis[0]*math.exp(80*(1-t)))
        else:
            # Direct scalar expansion at each source depth, then 2D integration.
            weights=[math.exp(-.5*j*j/2.25) for j in range(-3,4)]
            value=0.0
            for dy,wy in zip(range(-3,4),weights):
                for dx,wx in zip(range(-3,4),weights):
                    a,b=min(31,max(0,x+dx)),min(31,max(0,y+dy))
                    depth=f32(((a*13+b*7+a*b*3)%89+5)/100)
                    delta=.9*(depth-t+.015)
                    sample=.5+sum(2/(math.pi*(2*k+1))*math.sin(math.pi*(2*k+1)*delta) for k in range(8))
                    value+=wx*wy*sample/sum(weights)**2
            value=min(1,max(0,value))
        actual = float(row['visibility'])
        if not math.isfinite(actual) or abs(actual-value)>2e-4:
            raise ValueError(f'{method} reconstruction mismatch at {i}: {actual} vs {value}')
        visibility_error[method]=max(visibility_error[method],abs(actual-value))
    return {'cases':len(rows),'basis_max_abs_error':errors,'visibility_max_abs_error':visibility_error}

if __name__=='__main__':
    import json,sys
    print(json.dumps(verify(sys.argv[1]),indent=2))
