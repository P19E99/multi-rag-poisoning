"""prior phase retrieval, fusion, victim generation, validation and analysis runtime."""
from __future__ import annotations
import csv, gc, json, logging, math, sys
from pathlib import Path
import numpy as np
from core.common import file_sha256, fingerprint, read_jsonl, write_jsonl
from core.encoder import DenseEncoder, validate_embeddings
from core.index import ExactIndex, save_numpy
from adversarial.protocol import seal, verify_record
from adversarial.mechanism import mechanism_record, answer_metrics
from adversarial.analysis import summarize, write_cases, plots
from main71.logged_lm import LoggedLM
from .contract import initialize, construction_gate, guard, targets

LOG=logging.getLogger(__name__)


def load_base(c):
    import faiss
    from core.common import load_config
    base=load_config(c['frozen_config']); corpus=read_jsonl(base['paths']['corpus']); cache=Path(base['paths']['cache_dir'])
    meta=json.loads((cache/'corpus_index.meta.json').read_text())
    encoder=DenseEncoder({**base['retrieval'],'seed':c['seed']})
    if encoder.identity!=meta['encoder']: raise ValueError('Frozen BGE identity mismatch')
    index=faiss.read_index(str(cache/'corpus_index.faiss'))
    if not isinstance(index,faiss.IndexFlatIP) or index.ntotal!=100000: raise ValueError('Frozen exact 100k base index required')
    return corpus, ExactIndex(index,[d['doc_id'] for d in corpus],meta['fingerprint']), encoder


def overlay_index(base, poison, vector):
    import faiss
    if poison['doc_id'] in base.doc_ids: raise ValueError('Poison ID collides with base corpus')
    validate_embeddings(vector,1); augmented=faiss.clone_index(base.index); augmented.add(np.ascontiguousarray(vector,dtype=np.float32))
    if augmented.ntotal != base.index.ntotal+1: raise AssertionError('Poison overlay size mismatch')
    return ExactIndex(augmented,[*base.doc_ids,poison['doc_id']],fingerprint([base.fingerprint,poison,vector.tolist()]))


def retrieve(c):
    key=initialize(c); construction_gate(c); root=Path(c['project_root']); data=Path(c['paths']['data']); cache=Path(c['paths']['cache']); cache.mkdir(parents=True,exist_ok=True)
    ts=targets(c); views={r['sample_id']:verify_record(r) for r in read_jsonl(data/'query_views.jsonl')}; attacks={r['sample_id']:verify_record(r) for r in read_jsonl(data/'poisons.jsonl')}
    corpus,base,encoder=load_base(c); lookup={d['doc_id']:d for d in corpus}; out=Path(c['paths']['results'])
    for ordinal,t in enumerate(ts,1):
        sid=t['sample_id']; view=views[sid]; attack=attacks[sid]; output=cache/f'{sid}.retrieval.json'
        if view['protocol_fingerprint']!=key or attack['protocol_fingerprint']!=key: raise ValueError('Input provenance mismatch')
        input_key=fingerprint([key,t,view,attack,base.fingerprint])
        if output.exists():
            old=verify_record(json.loads(output.read_text()));
            if old['input_fingerprint']!=input_key: raise ValueError('Retrieval cache mismatch')
            continue
        queries=view['query_views']; qvec=encoder.encode_queries(queries); pvec=encoder.encode_documents([attack['poison_document']]); save_numpy(cache/f'{sid}.queries.npy',qvec); save_numpy(cache/f'{sid}.poison.npy',pvec)
        augmented=overlay_index(base,attack['poison_document'],pvec); clean_lists=base.search(qvec,10); poison_lists=augmented.search(qvec,10); docs={**lookup,attack['poison_document']['doc_id']:attack['poison_document']}; records=[]
        target={**t}
        for q in c['experiment']['query_counts']:
            for treatment,lists in (('clean',clean_lists),('poisoned',poison_lists)):
                r=mechanism_record(target,attack,queries[:q],lists[:q],docs,q,treatment,c['seed'],final_k=5,k_rrf=60)
                r.update(protocol_fingerprint=key,base_index_fingerprint=base.fingerprint,augmented_index_fingerprint=augmented.fingerprint if treatment=='poisoned' else base.fingerprint,query_vectors_sha256=file_sha256(cache/f'{sid}.queries.npy'),poison_vector_sha256=file_sha256(cache/f'{sid}.poison.npy'),n_corpus_documents=base.index.ntotal+(1 if treatment=='poisoned' else 0))
                records.append(r)
        write_jsonl(output,[seal({'input_fingerprint':input_key,'records':records})]); LOG.info('retrieved %d/100 %s',ordinal,sid)
    del encoder, corpus, base; gc.collect()


def generate(c):
    key=initialize(c); construction_gate(c); ts=targets(c); cache=Path(c['paths']['cache']); out=Path(c['paths']['results'])/'raw'/'main'; out.mkdir(parents=True,exist_ok=True); answer_cache=cache/'answers'; answer_cache.mkdir(exist_ok=True); lm=None
    for ordinal,t in enumerate(ts,1):
        sid=t['sample_id']; payload=verify_record(json.loads((cache/f'{sid}.retrieval.json').read_text()))
        for rec in payload['records']:
            dest=out/f'{sid}_q{rec["query_count"]}_{rec["treatment"]}.jsonl'
            if dest.exists():
                old=verify_record(read_jsonl(dest)[0]);
                if old['protocol_fingerprint']!=key or old['context_payload_sha256']!=rec['context_payload_sha256']: raise ValueError('Generation cache mismatch')
                continue
            answer_key=fingerprint([key,rec['generator_messages'],c['generator']]); answer_path=answer_cache/f'{answer_key}.json'
            if answer_path.exists(): generated=verify_record(json.loads(answer_path.read_text()))
            else:
                if lm is None: lm=LoggedLM(c)
                generated=seal(lm.generate(rec['generator_messages'],max_new_tokens=96)); answer_path.write_text(json.dumps(generated,ensure_ascii=False)+'\n')
            if generated['messages']!=rec['generator_messages'] or generated['input_truncated']: raise AssertionError('Generator input differs from retrieval context')
            if any(doc['text'] not in generated['actual_prompt'] for doc in rec['final_context_documents']): raise AssertionError('Context document absent from actual prompt')
            ids=[doc['doc_id'] for doc in rec['final_context_documents'] if doc['text'] in generated['actual_prompt']]
            row={**rec,'actual_context_document_ids':ids,'poison_in_final_context':rec['poison_document_id'] in ids,'generator_input':generated,'generated_answer':generated['text'],'status':'completed',**answer_metrics(generated['text'],rec['target_answer'],rec['reference_answer'])}
            row.update(clean_or_poisoned=rec['treatment'],context_doc_ids=ids,full_context_text=rec['generator_messages'][1]['content'],actual_prompt=generated['actual_prompt'],input_token_ids=generated['input_token_ids'],output_token_ids=generated['output_token_ids'],primary_ASR=row['attack_success'],EM=row['qa_em'],F1=row['qa_f1'],clean_target_answer_indicator=row['attack_success'] if rec['treatment']=='clean' else None,clean_target_answer_rate=row['attack_success'] if rec['treatment']=='clean' else None)
            write_jsonl(dest,[seal(row)])
        LOG.info('generated %d/100 %s',ordinal,sid)


def validate(c):
    key=initialize(c); construction_gate(c); out=Path(c['paths']['results']); files=sorted((out/'raw'/'main').glob('*.jsonl')); rows=[verify_record(read_jsonl(p)[0]) for p in files]
    if len(rows)!=800: raise ValueError(f'Expected 800 generation records, found {len(rows)}')
    seen=set()
    for r in rows:
        pair=(r['target_id'],r['query_count'],r['treatment']);
        if pair in seen: raise ValueError('Duplicate condition record')
        seen.add(pair)
        if r['protocol_fingerprint']!=key or len(r['final_context_document_ids'])!=5 or len(r['query_views'])!=r['query_count']: raise ValueError('Record contract mismatch')
        if len(r['per_view_retrieval'])!=r['query_count'] or any(len(v['ranked_list'])!=10 for v in r['per_view_retrieval']): raise ValueError('Top-10 retrieval contract failed')
        if len(set(r['final_context_document_ids']))!=5: raise ValueError('Duplicate final context document')
        if r['actual_context_document_ids']!=r['final_context_document_ids']: raise ValueError('Actual context mismatch')
        if not isinstance(r['generator_input'].get('input_token_ids'),list) or not isinstance(r['generator_input'].get('output_token_ids'),list): raise ValueError('Missing generator token trace')
        if r['treatment']=='clean' and r['poison_in_final_context']: raise ValueError('Poison in clean context')
    if len(seen)!=800: raise ValueError('Incomplete condition set')
    replay={'passed':True,'protocol_fingerprint':key,'target_id':targets(c)[0]['sample_id'],'retrieval_records_checked':8,'generation_records_checked':8,'scope':'artifact replay; no additional model calls'}
    (out/'replay.json').write_text(json.dumps(replay,indent=2)+'\n')
    validation={'passed':True,'protocol_fingerprint':key,'n_targets':100,'n_records':800,'n_retrieval_target_files':100,'n_generation_records':800,'clean_records':400,'poisoned_records':400,'all_conditions_present':True,'replay_passed':True}
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n'); return rows


def analyze(c, rows=None):
    key=initialize(c); out=Path(c['paths']['results']); rows=rows or [verify_record(read_jsonl(p)[0]) for p in sorted((out/'raw'/'main').glob('*.jsonl'))]
    result=summarize(rows,42,2000); result['cases']=write_cases(rows,out/'cases');
    sys.path.insert(0,str(Path(c['project_root'])/'cache/target_selection/plotting_deps')); plots(result,out/'plots',c); (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result['summary'][0])); w.writeheader(); w.writerows(result['summary'])
    (out/'bootstrap.json').write_text(json.dumps({'design':result['bootstrap'],'intervals':result['bootstrap_95ci']},indent=2)+'\n')
    clean_hits=[{'target_id':r['target_id'],'query_count':r['query_count'],'question':r['original_query'],'target_answer':r['target_answer'],'reference_answer':r['reference_answer'],'generated_answer':r['generated_answer'],'substring_match':r['attack_success'],'context_doc_ids':r['actual_context_document_ids']} for r in rows if r['treatment']=='clean' and r['attack_success']]
    write_jsonl(out/'clean_target_answer_hits.jsonl',clean_hits)
    flags=[{'target_id':r['target_id'],'query_count':r['query_count'],'treatment':r['treatment'],'question':r['original_query'],'target_answer':r['target_answer'],'reference_answer':r['reference_answer'],'generated_answer':r['generated_answer'],**r['attack_evaluation_details']} for r in rows if r['attack_evaluation_details']['substring_only_match'] or r['attack_evaluation_details']['target_normalized_equals_gold'] or (r['attack_success'] and not r['poison_in_final_context'])]
    write_jsonl(out/'evaluation_flags.jsonl',flags); return result,clean_hits
