"""Isolated prior phase main contract and frozen-input preparation."""
from __future__ import annotations
import copy, json, os
from pathlib import Path
import yaml
from core.common import file_sha256, fingerprint, read_jsonl, write_jsonl
from adversarial.protocol import seal, verify_record
from core.expansion import validate_rewrites
from adversarial.prepare import build_poison


def load(path):
    p=Path(path).resolve(); c=yaml.safe_load(p.read_text(encoding='utf-8')); root=(p.parent/c['project_root']).resolve(); c['project_root']=str(root); c['config_path']=str(p)
    for k in ('data','cache','results','frozen_query_views','repaired_poisons','frozen_protected'):
        c['paths'][k]=str((root/c['paths'][k]).resolve())
    c['frozen_config']=str((root/c['frozen_config']).resolve()); c['frozen_manifest']=str((root/c['frozen_manifest']).resolve())
    c['targets']['path']=str((root/c['targets']['path']).resolve())
    for sec in ('retrieval','generator'):
        c[sec]['model_path']=str((root/c[sec]['model_path']).resolve())
    for k in ('data','cache','results'):
        Path(c['paths'][k]).mkdir(parents=True,exist_ok=True)
    os.environ['OMP_NUM_THREADS']='8'; os.environ['MKL_NUM_THREADS']='8'; os.environ['TOKENIZERS_PARALLELISM']='false'; os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8'); os.environ['PYTHONDONTWRITEBYTECODE']='1'
    if c['experiment'] != {'query_counts':[1,2,4,8],'per_view_top_k':10,'final_top_k':5,'k_rrf':60,'poison_docs_per_target':1,'run_clean_controls':True}:
        raise ValueError('prior phase main experiment settings changed')
    return c


def source_hashes(c):
    root=Path(c['project_root']); paths=sorted((root/'src/runtime').glob('*.py'))+[root/'scripts/run_frozen_retrieval_main.py',root/'configs/frozen_retrieval_main.yaml']
    return {str(p.relative_to(root)):file_sha256(p) for p in paths if p.exists()}


def _historical_files(c):
    root=Path(c['project_root']); old=json.loads((root/'results/frozen_queries/main/protected_before_main.json').read_text())['files_sha256']; extra={}
    for p in [root/'data/frozen_queries/main/query_views.jsonl',root/'data/frozen_retrieval/main/poisons_repaired.jsonl',root/'configs/frozen_retrieval_poison.yaml',root/'results/frozen_retrieval/tier3_repair/repair_summary.json',Path(c['config_path'])]:
        extra[str(p.relative_to(root))]=file_sha256(p)
    return {**old,**extra}


def preflight(c):
    out=Path(c['paths']['results']); snap=out/'protected_before_main.json'
    if not snap.exists():
        files=_historical_files(c); snap.write_text(json.dumps({'files_sha256':files,'expected_count':len(files),'source':'Step7.2 guard plus Step7.3 repaired inputs'},indent=2)+'\n')
    guard(c); return len(json.loads(snap.read_text())['files_sha256'])


def guard(c):
    root=Path(c['project_root']); snap=json.loads((Path(c['paths']['results'])/'protected_before_main.json').read_text())
    bad=[]
    for name,h in snap['files_sha256'].items():
        p=root/name
        if not p.is_file() or file_sha256(p)!=h: bad.append(name)
    if bad: raise ValueError('Protected historical artifact changed: '+bad[0])
    return len(snap['files_sha256'])


def identity(c):
    return fingerprint({'config_sha256':file_sha256(c['config_path']),'source_sha256':source_hashes(c),'protected_sha256':file_sha256(Path(c['paths']['results'])/'protected_before_main.json')})


def initialize(c):
    guard(c); key=identity(c); p=Path(c['paths']['results'])/'main_manifest.json'; value={'protocol_fingerprint':key,'config_sha256':file_sha256(c['config_path']),'source_sha256':source_hashes(c)}
    if p.exists() and json.loads(p.read_text())!=value: raise ValueError('prior phase main manifest mismatch')
    if not p.exists(): p.write_text(json.dumps(value,indent=2)+'\n')
    return key


def targets(c):
    p=Path(c['targets']['path']); rows=read_jsonl(p)
    if len(rows)!=100 or len({r['sample_id'] for r in rows})!=100: raise ValueError('Fixed 100 targets required')
    return rows


def prepare_inputs(c):
    key=initialize(c); root=Path(c['project_root']); data=Path(c['paths']['data']); target_rows=targets(c)
    old_views=read_jsonl(c['paths']['frozen_query_views']); old_poisons=read_jsonl(c['paths']['repaired_poisons'])
    if len(old_views)!=100 or len(old_poisons)!=100: raise ValueError('Expected 100 frozen views and repaired poisons')
    vmap={r['sample_id']:r for r in old_views}; pmap={r['sample_id']:r for r in old_poisons}; views=[]; poisons=[]
    for t in target_rows:
        sid=t['sample_id']; oldv=verify_record(vmap[sid]); oldp=pmap[sid]
        if oldv['original_query']!=t['question'] or oldv['query_views'][0]!=t['question']: raise ValueError('Rewrite pool question mismatch')
        validate_rewrites(t['question'],oldv['rewrites'],7)
        v={k:v for k,v in oldv.items() if k!='record_sha256'}; v.update(protocol_fingerprint=key,source_protocol_fingerprint=oldv.get('protocol_fingerprint'),source_record_sha256=oldv['record_sha256'])
        views.append(seal(v))
        if oldp['sample_id']!=sid or 'poison_document' not in oldp: raise ValueError('Poison input mismatch')
        p={'sample_id':sid,'question':t['question'],'reference_answer':t['answer'],'attack_id':oldp['attack_id'],'target_answer':oldp['target_answer'],'poison_document':oldp['poison_document'],'construction':oldp.get('construction'),'protocol_fingerprint':key,'source_record_sha256':fingerprint(oldp),'fresh_frozen_main':True}
        poisons.append(seal(p))
    write_jsonl(data/'query_views.jsonl',views); write_jsonl(data/'poisons.jsonl',poisons)
    validation={'passed':True,'protocol_fingerprint':key,'n_targets':100,'n_rewrites':700,'n_query_views':800,'n_poisons':100,'fixed_target_order':True,'query_views_sha256':file_sha256(data / 'query_views.jsonl'),'poisons_sha256':file_sha256(data / 'poisons.jsonl'),'repaired_ordinals':[29,46,48,98]}
    (Path(c['paths']['results'])/'construction_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    (Path(c['paths']['results'])/'frozen_inputs.json').write_text(json.dumps({'protocol_fingerprint':key,'query_views_sha256':validation['query_views_sha256'],'poisons_sha256':validation['poisons_sha256'],'n_targets':100,'n_views':800,'n_poisons':100,'source_repaired_collection_sha256':file_sha256(c['paths']['repaired_poisons'])},indent=2)+'\n')
    return target_rows,views,poisons


def construction_gate(c):
    key=initialize(c); out=Path(c['paths']['results']); v=json.loads((out/'construction_validation.json').read_text());
    if not v.get('passed') or v['protocol_fingerprint']!=key or v['n_poisons']!=100 or v['n_rewrites']!=700: raise ValueError('Construction gate failed')
    if file_sha256(Path(c['paths']['data'])/'query_views.jsonl')!=v['query_views_sha256'] or file_sha256(Path(c['paths']['data'])/'poisons.jsonl')!=v['poisons_sha256']: raise ValueError('Prepared input changed after validation')
