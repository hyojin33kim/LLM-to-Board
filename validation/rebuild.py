"""Rebuild repeat-token-1 C fixtures; run from repository root (little-endian host)."""
from pathlib import Path
import hashlib, json, subprocess, sys, tempfile
assert sys.byteorder == 'little'
p=Path('validation/run.c')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='9c4f2d5c6ae01b71726d1cc37530d71e60bff0ec7cc012565f16a43c1ca658bd'
weights=Path('dist/fixtures/weights.bin')
assert hashlib.sha256(weights.read_bytes()).hexdigest()=='b0a507e7ad0f626624f17112325e66691f9076d622e1d3274d103d00299f2696'
s=p.read_text();needle='x[i] += s->xb[i];\n        }\n    }';assert s.count(needle)==1
Path('validation/golden_run.c').write_text(s.replace(needle,'x[i] += s->xb[i];\n        }\n        capture_layer(l,x,dim);\n    }'))
with tempfile.TemporaryDirectory() as tmp:
    exe=str(Path(tmp)/'golden')
    subprocess.run(['cc','-O2','-ffp-contract=off','-Wno-unknown-pragmas','validation/golden.c','-lm','-o',exe],check=True)
    subprocess.run([exe],check=True)
manifest={'source_commit':'661292a84ec02d5fda4ec5f61ea03f3a8e4a23cf','run_c_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'input':'token ID 1 repeated','contexts':[1,128,512],'compiler':subprocess.check_output(['cc','--version'],text=True).splitlines()[0],'flags':['-O2','-ffp-contract=off'],'files':{}}
for p in sorted(Path('dist/fixtures').glob('*.bin')):
    manifest['files'][p.name]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
Path('dist/fixtures/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
