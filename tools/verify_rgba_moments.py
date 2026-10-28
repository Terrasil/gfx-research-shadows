#!/usr/bin/env python3
"""Compare moment reconstruction with independent analytic disk/strip areas."""
import csv
import math
from pathlib import Path


def coverage(x):
    x=max(-1.0,min(1.0,x))
    return (math.acos(-x)+x*math.sqrt(max(0.0,1-x*x)))/math.pi


def verify(path):
    rows=list(csv.DictReader(Path(path).open(newline='')))
    keys={(int(r['fixture']),int(r['samples']),int(r['index'])) for r in rows}
    if len(rows)!=3120 or keys!={(f,n,i) for f in range(4) for n in (16,32,64) for i in range(260)}:
        raise ValueError('Incomplete PCF moment fixture matrix')
    errors={};reference_error=0.0;fallback_error=0.0;active=0
    for row in rows:
        f,n=int(row['fixture']),int(row['samples']);margin=float(row['margin'])
        raw,model,confidence,reference=(float(row[k]) for k in ('raw','model','confidence','reference'))
        if not all(math.isfinite(v) and -1e-6<=v<=1.000001 for v in (raw,model,confidence,reference)):
            raise ValueError('Invalid PCF moment value')
        expected=float(margin>=0) if f==2 else 1-(coverage(.15-margin)-coverage(-.15-margin)) if f==3 else coverage(margin)
        reference_error=max(reference_error,abs(expected-reference))
        result=raw+(model-raw)*confidence*.5
        a=errors.setdefault((f,n),[0.0,0.0,0])
        a[0]+=(raw-expected)**2;a[1]+=(result-expected)**2;a[2]+=1
        active+=confidence>0
        if f in (1,2):fallback_error=max(fallback_error,abs(result-raw))
    if reference_error>2e-6 or fallback_error>1e-7:
        raise ValueError('PCF moment analytic reference or required fallback failed')
    metrics={f'{f}-{n}':{'raw_rmse':math.sqrt(a/count),'reconstructed_rmse':math.sqrt(b/count)}
             for (f,n),(a,b,count) in errors.items()}
    if not active:raise ValueError('Moment reconstruction never activated')
    for n in (32,64):
        r=metrics[f'0-{n}']
        if r['reconstructed_rmse']>=r['raw_rmse']:
            raise ValueError('Moment reconstruction did not improve the analytic straight edge')
    return {'cases':len(rows),'reference_max_abs_error':reference_error,'fallback_max_abs_error':fallback_error,
            'active_cases':active,'fixture_rmse':metrics}


if __name__=='__main__':
    import json
    import sys
    print(json.dumps(verify(sys.argv[1]),indent=2))
