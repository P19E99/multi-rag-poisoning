"""Second-model Llama confirmation generation over frozen prior phase traces."""
from __future__ import annotations
import json, re, time
from pathlib import Path
import torch
from core.common import fingerprint

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'experiments/consensus'; MODEL=ROOT.parent/'models/Meta-Llama-3.1-8B-Instruct'
BASE={'do_sample':False,'num_beams':1,'temperature':1.0,'top_p':1.0,'top_k':0,'repetition_penalty':1.0,'no_repeat_ngram_size':0,'use_cache':True}
def rows(p): return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def marker_text(raw):
 raw=str(raw or ''); pos=raw.lower().rfind('final answer'); return raw[pos+len('final answer'):] if pos>=0 else raw
def group_sources():
 return {
  'C1_baseline_poisoned': ('baseline', 'poisoned', ROOT/'results/frozen_retrieval/main/raw/main', 'C1'),
  'C2_agree_v2_poisoned': ('agree_v2', 'poisoned', ROOT/'consensus_probes/fix/traces/agree_v2.jsonl', 'C2'),
  'C3_baseline_poisoned': ('baseline', 'poisoned', ROOT/'experiments/baseline/traces/C3_baseline.jsonl', 'C3'),
  'C4_agree_v2_poisoned': ('agree_v2', 'poisoned', ROOT/'experiments/baseline/traces/C4_agree_v2.jsonl', 'C4'),
  'clean_baseline': ('baseline', 'clean', ROOT/'results/frozen_retrieval/main/raw/main', 'clean_baseline'),
  'clean_agree_v2': ('agree_v2', 'clean', ROOT/'consensus_probes/fix/traces/agree_v2.jsonl', 'clean_agree_v2'),
 }
def load_group(prompt,treatment,path):
 if path.is_dir():
  out=[]
  for p in sorted(path.glob('*_q*_*.jsonl')):
   rs=rows(p)
   if rs and rs[0].get('treatment')==treatment: out.extend(rs)
  return out
 return [r for r in rows(path) if r.get('treatment')==treatment]
class Victim:
 def __init__(self):
  from transformers import AutoModelForCausalLM,AutoTokenizer
  self.tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
  self.model=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').to('cuda').eval()
 def call(self,messages,max_new_tokens):
  prompt=self.tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
  inp=self.tok(prompt,return_tensors='pt',add_special_tokens=False,truncation=False).to('cuda')
  kwargs=dict(BASE); kwargs['max_new_tokens']=max_new_tokens; kwargs['pad_token_id']=self.tok.pad_token_id; kwargs['eos_token_id']=self.tok.eos_token_id
  torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
  with torch.inference_mode(): out=self.model.generate(**inp,**kwargs)
  torch.cuda.synchronize(); ids=out[0,inp['input_ids'].shape[1]:].tolist(); raw=self.tok.decode(ids,skip_special_tokens=True).strip()
  return {'messages':messages,'actual_prompt':prompt,'input_token_ids':inp['input_ids'][0].tolist(),'output_token_ids':ids,'raw_text':raw,'decoding_kwargs':kwargs,'seed':42,'thinking_parameter_passed':False,'runtime_seconds':time.perf_counter()-started,'gpu_memory':{'allocated':int(torch.cuda.max_memory_allocated()),'reserved':int(torch.cuda.max_memory_reserved())},'prompt_sha256':fingerprint(prompt),'raw_sha256':fingerprint(raw)}
def main():
 (OUT/'traces').mkdir(exist_ok=True); sources=group_sources(); lm=Victim(); total=0
 for name,(prompt,treatment,path,tag) in sources.items():
  dest=OUT/'traces'/f'{name}.jsonl'; done={}
  if dest.exists():
   for r in rows(dest): done[(r['target_id'],r['query_count'])]=r
  recs=sorted(load_group(prompt,treatment,path),key=lambda r:(r['target_id'],r['query_count']))
  if len(recs)!=400: raise RuntimeError(f'{name}: expected 400 rows, got {len(recs)}')
  with dest.open('a',encoding='utf-8') as h:
   for i,r in enumerate(recs,1):
    key=(r['target_id'],r['query_count'])
    if key in done: continue
    msgs=r.get('generator_messages') or r.get('generator_input',{}).get('messages')
    if not msgs: raise RuntimeError(f'missing messages {name} {key}')
    g=lm.call(msgs,96 if prompt=='baseline' else 512); raw=g['raw_text']; sc=raw if prompt=='baseline' else marker_text(raw)
    out={'condition':tag,'prompt_variant':prompt,'treatment':treatment,'target_id':r['target_id'],'query_count':r['query_count'],'original_query':r.get('original_query'),'target_answer':r['target_answer'],'reference_answer':r['reference_answer'],'final_context_document_ids':r.get('final_context_document_ids'),'source_trace_sha256':fingerprint(r),'source_trace_file':str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),'generator_input':g,'generated_answer':raw,'scored_answer':sc,'parse_success':True if prompt=='baseline' else ('final answer' in raw.lower()),'decoding_protocol':'explicit do_sample=false; temperature=1.0; top_p=1.0; top_k=0','record_sha256':fingerprint({'condition':tag,'target_id':r['target_id'],'q':r['query_count'],'raw_sha256':g['raw_sha256']})}
    h.write(json.dumps(out,ensure_ascii=False)+'\n'); h.flush(); done[key]=out; total+=1
    if i%25==0: print(name,i,flush=True)
  if len(done)!=400: raise RuntimeError(f'{name}: output count {len(done)}')
 del lm
 print(json.dumps({'generated_new':total,'groups':{k:sum(1 for _ in rows(OUT/'traces'/f'{k}.jsonl')) for k in sources}},indent=2))
if __name__=='__main__': main()
