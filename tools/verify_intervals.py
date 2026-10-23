#!/usr/bin/env python3
"""Check concurrent GPU endpoint reduction against an independent CPU reduction."""
import csv
from pathlib import Path

def verify(path):
    with Path(path).open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    if len(rows)!=256 or sorted(int(r['pixel']) for r in rows)!=list(range(256)):
        raise ValueError('Incomplete interval fixtures')
    expected=[]
    for pixel in range(256):
        inputs=[1000+(i*137+(i//128)**2*31+pixel*19)%62000 for i in range(pixel,4096,128)] if pixel<128 else []
        expected.append((min(inputs)<<16)|max(inputs) if inputs else 0xffff0000)
    for row in rows:
        pixel=int(row['pixel'])
        if int(row['packed'])!=expected[pixel]:
            raise ValueError(f'Atomic endpoint mismatch at pixel {pixel}')
    return {'cases':256,'concurrent_updates':4096,'exact':True}

if __name__=='__main__':
    import json,sys
    print(json.dumps(verify(sys.argv[1]),indent=2))
