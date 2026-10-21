"""Independent actual-artifact checks and full first-target deterministic replay."""
import json,math
from pathlib import Path
import numpy as np
from core.common import read_jsonl,write_jsonl,fingerprint,file_sha256
from core.expansion import validate_rewrites
from .protocol import initialize,selected_targets,verify_frozen,verify_record
from .mechanism import answer_metrics,context_messages

def validate(c,phase):
    key=initialize(c);targets=selected_targets(c,phase);root=Path(c['paths']['results']);directory=root/'raw'/phase
    rows=[verify_record(read_jsonl(p)[0]) for p in sorted(directory.glob('*.jsonl'))]
    assert len(rows)==len(targets)*8
    target_map={r['sample_id']:r for r in targets};seen=set();views={r['sample_id']:verify_record(r) for r in read_jsonl(Path(c['paths']['data'])/'query_views.jsonl')}
    attacks={r['sample_id']:verify_record(r) for r in read_jsonl(Path(c['paths']['data'])/'poisons.jsonl')}
    for r in rows:
        pair=(r['target_id'],r['query_count'],r['treatment']);assert pair not in seen;seen.add(pair)
        target=target_map[r['target_id']];count=r['query_count'];poison=r['poison_document_id']
        assert r['protocol_fingerprint']==key and r['seed']==42
        assert r['original_query']==target['question'] and r['query_views']==views[r['target_id']]['query_views'][:count]
        validate_rewrites(target['question'],views[r['target_id']]['rewrites'],7)
        completion=views[r['target_id']].get('completion')
        if completion:
            raw=root/'construction_raw';sid=r['target_id']
            assert file_sha256(raw/f'{sid}_dedup.json')==completion['dedup_trace_sha256']
            if completion['topup_raw_sha256']:
                assert file_sha256(raw/f'{sid}_topup.jsonl')==completion['topup_raw_sha256']
            assert completion['full_attempts_used']==3 and 0<=completion['topup_attempts_used']<=3
        assert r['poison_document']==attacks[r['target_id']]['poison_document']
        assert r['n_corpus_documents']==100000+(r['treatment']=='poisoned')
        assert len(r['per_view_retrieval'])==count and len(r['final_context_document_ids'])==5
        assert len(set(r['final_context_document_ids']))==5
        scores={};votes={};flat=[]
        for v in r['per_view_retrieval']:
            ranked=v['ranked_list'];assert len(ranked)==10 and [h['rank'] for h in ranked]==list(range(1,11))
            assert len({h['doc_id'] for h in ranked})==10
            assert all(math.isfinite(h['raw_cosine_score']) for h in ranked)
            assert ranked==sorted(ranked,key=lambda h:(-h['raw_cosine_score'],h['doc_id']))
            hit=next((h for h in ranked if h['doc_id']==poison),None)
            assert v['poison_retrieved']==(hit is not None)
            assert v['poison_rank']==(hit['rank'] if hit else None)
            assert v['poison_retrieval_score']==(hit['raw_cosine_score'] if hit else None)
            for h in ranked:
                scores[h['doc_id']]=scores.get(h['doc_id'],0)+(h['raw_cosine_score'] if count==1 else 1/(60+h['rank']))
                votes[h['doc_id']]=votes.get(h['doc_id'],0)+1
                flat.append({'view_id':v['query_view_id'],**h})
        order=sorted(scores,key=lambda d:(-scores[d],d))
        assert r['pre_dedup_candidates']==flat and r['candidate_total']==count*10
        assert r['deduplicated_candidate_ids']==sorted(scores) and r['candidate_unique']==len(scores)
        assert [h['doc_id'] for h in r['fused_candidates']]==order
        for i,h in enumerate(r['fused_candidates'],1):
            assert h['fused_rank']==i and math.isclose(h['fused_score'],scores[h['doc_id']],abs_tol=1e-12)
            assert h['num_supporting_views']==votes[h['doc_id']]
        assert r['final_context_document_ids']==order[:5]
        coverage=votes.get(poison,0);assert r['exposure']==bool(coverage) and r['view_coverage']==coverage
        assert r['normalized_view_coverage']==coverage/count
        assert r['fusion_survival']==(poison in order[:5])
        assert r['poison_fused_rank']==(order.index(poison)+1 if poison in order else None)
        assert r['generator_messages']==context_messages(target['question'],r['final_context_documents'])
        g=r['generator_input'];verify_record(g)
        assert g['messages']==r['generator_messages'] and not g['input_truncated']
        assert len(g['prompt_token_ids'])==g['prompt_tokens']
        assert fingerprint(g['actual_prompt'])==g['prompt_sha256']
        assert all(d['text'] in g['actual_prompt'] for d in r['final_context_documents'])
        assert r['actual_context_document_ids']==r['final_context_document_ids']
        assert r['poison_in_final_context']==(poison in r['actual_context_document_ids'])==r['fusion_survival']
        assert r['poison_context_position']==(order[:5].index(poison)+1 if poison in order[:5] else None)
        for k,v in answer_metrics(r['generated_answer'],r['target_answer'],r['reference_answer']).items():assert r[k]==v
        gold=set(target['supporting_doc_ids']);recall=len(gold&set(order[:5]))/len(gold)
        assert r['benign_recall_at_5']==recall and r['hit_any_gold_at_5']==bool(recall)
    assert len(seen)==len(targets)*8
    log=(root/'unit_tests.log').read_text();assert log.rstrip().endswith('OK')
    verify_frozen(c);out=root/phase;out.mkdir(exist_ok=True)
    replay_path=(out/'replay.json') if phase=='smoke' else (root/'smoke'/'replay.json');replay=json.loads(replay_path.read_text()) if replay_path.exists() else {}
    report={'passed':True,'protocol_fingerprint':key,'n_targets':len(targets),'n_records':len(rows),'n_conditions_per_target':8,
            'failed_samples':0,'skipped_samples':0,'frozen_unchanged':True,'unit_tests_passed':True,
            'single_sample_replay_passed':replay.get('passed',False) and replay.get('protocol_fingerprint')==key,
            'raw_files_sha256':{p.name:file_sha256(p) for p in sorted(directory.glob('*.jsonl'))}}
    write_jsonl(out/'validation.json',[report]);print(json.dumps({k:v for k,v in report.items() if k!='raw_files_sha256'},indent=2))
    return report

def replay(c,phase):
    from .run import load_base,overlay_index
    from .lm import LocalLM
    key=initialize(c);target=selected_targets(c,phase)[0];sid=target['sample_id'];cache=Path(c['paths']['cache']);root=Path(c['paths']['results'])
    prepared=verify_record(json.loads((cache/f'{sid}.retrieval.json').read_text()))
    view=next(r for r in read_jsonl(Path(c['paths']['data'])/'query_views.jsonl') if r['sample_id']==sid)
    attack=next(r for r in read_jsonl(Path(c['paths']['data'])/'poisons.jsonl') if r['sample_id']==sid)
    corpus,base,encoder=load_base(c);q=encoder.encode_queries(view['query_views']);p=encoder.encode_documents([attack['poison_document']])
    assert np.allclose(q,np.load(cache/f'{sid}.queries.npy'),rtol=0,atol=1e-7)
    assert np.allclose(p,np.load(cache/f'{sid}.poison.npy'),rtol=0,atol=1e-7)
    augmented=overlay_index(base,attack['poison_document'],p)
    lists={'clean':base.search(q,10),'poisoned':augmented.search(q,10)}
    for record in prepared['records']:
        assert [v['ranked_list'] for v in record['per_view_retrieval']]==lists[record['treatment']][:record['query_count']]
    # Recreate original construction responses, not only the cached retrieval world.
    lm=LocalLM(c);construction=[]
    for kind in ('expansion','attack'):
        schema=({'type':'array','minItems':7,'maxItems':7,'items':{'type':'string','minLength':1}} if kind=='expansion' else
                {'type':'object','properties':{k:{'type':'string','minLength':1} for k in ('incorrect_answer','corpus1')},'required':['incorrect_answer','corpus1'],'additionalProperties':False})
        for original in read_jsonl(root/'construction_raw'/f'{sid}_{kind}.jsonl'):
            repeated=lm.generate(original['messages'],max_new_tokens=c[kind]['max_new_tokens'],schema=schema)
            assert repeated['text']==original['text'] and repeated['prompt_token_ids']==original['prompt_token_ids']
            construction.append({'kind':kind,'attempt':original['attempt'],'exact_match':True})
    topup_path=root/'construction_raw'/f'{sid}_topup.jsonl'
    if topup_path.exists():
        for original in read_jsonl(topup_path):
            missing=json.loads(original['messages'][1]['content'])['number_of_new_rewrites']
            schema={'type':'array','minItems':missing,'maxItems':missing,'items':{'type':'string','minLength':1}}
            repeated=lm.generate(original['messages'],max_new_tokens=c['expansion']['topup']['max_new_tokens'],schema=schema)
            assert repeated['text']==original['text'] and repeated['prompt_token_ids']==original['prompt_token_ids']
            construction.append({'kind':'topup','attempt':original['attempt'],'exact_match':True})
    outputs=[]
    for record in prepared['records']:
        path=root/'raw'/phase/f'{sid}_q{record["query_count"]}_{record["treatment"]}.jsonl';original=verify_record(read_jsonl(path)[0])
        repeated=lm.generate(record['generator_messages'],max_new_tokens=c['generator']['max_new_tokens'])
        assert repeated['text']==original['generated_answer'] and repeated['prompt_token_ids']==original['generator_input']['prompt_token_ids']
        outputs.append({'query_count':record['query_count'],'treatment':record['treatment'],'answer':repeated['text'],'exact_match':True})
    verify_frozen(c);out=root/phase;out.mkdir(exist_ok=True)
    report={'passed':True,'protocol_fingerprint':key,'target_id':sid,'construction':construction,'retrieval_exact_match':True,'answers':outputs,'frozen_unchanged':True}
    write_jsonl(out/'replay.json',[report]);print(json.dumps(report,indent=2))
