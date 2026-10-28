#!/usr/bin/env python3
"""Check physical PCSS radii and invariance to affine depth encodings."""
import csv
import math
import struct
from pathlib import Path


def verify(path):
    with Path(path).open(newline='') as stream:rows=list(csv.DictReader(stream))
    key=lambda r:(int(r['fixture']),float(r['extent']),int(r['mean']),int(r['bounds']),int(r['encoding']),int(r['index']))
    expected={(f,e,m,b,c,i) for f in range(3) for e in (1.,4.) for m in (0,1) for b in (0,1) for c in range(4) for i in range(64)}
    if len(rows)!=6144 or {key(r) for r in rows}!=expected:raise ValueError('Incomplete PCSS depth-encoding cases')
    tangent=struct.unpack('f',struct.pack('f',.026))[0]
    lookup={key(r):r for r in rows};search_error=radius_error=plane_gap_error=0.0
    changes={k:0.0 for k in ('visibility','mean_gap','filter_radius_m','search_radius_m')}
    legacy_visibility_change=0.0;legacy_search_change=0.0
    for r in rows:
        f,e,m,b,c,i=key(r);distance=8*4**c
        for field in changes:
            v=float(r[field])
            if not math.isfinite(v) or v<0 or (field=='visibility' and v>1.000001):raise ValueError('Invalid depth-scale output')
        reference_search=min((4 if b else distance)*tangent,3)
        search_error=max(search_error,abs(float(r['search_radius_m'])-reference_search))
        radius_error=max(radius_error,abs(float(r['filter_radius_m'])-float(r['mean_gap'])*tangent*.5))
        if f==0 and float(r['mean_gap'])>0:
            plane_gap_error=max(plane_gap_error,abs(float(r['mean_gap'])-(4-.00015*80)))
        base=lookup[(f,e,m,b,0,i)]
        if b:
            for field in changes:changes[field]=max(changes[field],abs(float(r[field])-float(base[field])))
        else:
            legacy_visibility_change=max(legacy_visibility_change,abs(float(r['visibility'])-float(base['visibility'])))
            legacy_search_change=max(legacy_search_change,abs(float(r['search_radius_m'])-float(base['search_radius_m'])))
    if search_error>1e-6 or radius_error>1e-6 or plane_gap_error>1e-4:raise ValueError('PCSS world radius differs from physical reference')
    if changes['search_radius_m']>1e-6 or changes['mean_gap']>1e-4 or changes['filter_radius_m']>5e-6 or changes['visibility']>2e-4:
        raise ValueError('Bounded PCSS changed with camera distance: '+str(changes))
    if legacy_search_change<2.5:raise ValueError('Legacy fixture did not expose camera-dependent search')
    return dict(cases=len(rows),max_search_reference_error=search_error,max_radius_reference_error=radius_error,
                max_plane_gap_reference_error=plane_gap_error,bounded_max_camera_changes=changes,
                legacy_max_camera_visibility_change=legacy_visibility_change,legacy_max_camera_search_change_m=legacy_search_change)
