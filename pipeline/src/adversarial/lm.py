"""Local deterministic LM with exact model-input logging; no context truncation."""
from core.common import fingerprint
from copy import deepcopy
class LocalLM:
    def __init__(self,config):
        from core.expansion import QwenExpansionBackend
        from core.json_constraints import JsonListConstraint
        self.backend=QwenExpansionBackend({**config['generator'],'seed':config['seed'],'constrain_json':False})
        self.constraint=JsonListConstraint(self.backend.tokenizer,7)
        self.limit=config['generator']['max_input_tokens']
    def generate(self,messages,*,max_new_tokens,schema=None):
        b=self.backend
        prompt=b.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=b.tokenizer(prompt,return_tensors='pt',add_special_tokens=False,truncation=False).to(b.device)
        n=inputs['input_ids'].shape[1]
        if n>self.limit or n+max_new_tokens>b.model.config.max_position_embeddings:raise ValueError(f'Full context exceeds window: {n}; cannot drop documents')
        kwargs={}
        if schema:
            self.constraint.schema=schema;kwargs['prefix_allowed_tokens_fn']=self.constraint.prefix_function()
        with b.torch.inference_mode():
            tokens=b.model.generate(**inputs,**kwargs,do_sample=False,num_beams=1,max_new_tokens=max_new_tokens,pad_token_id=b.tokenizer.pad_token_id)
        output=b.tokenizer.decode(tokens[0,n:],skip_special_tokens=True)
        return {'text':output.strip(),'messages':deepcopy(messages),'actual_prompt':prompt,'prompt_token_ids':inputs['input_ids'][0].tolist(),
                'prompt_tokens':n,'generated_tokens':tokens.shape[1]-n,'prompt_sha256':fingerprint(prompt),
                'input_truncated':False,'do_sample':False,'temperature':0,'enable_thinking':False}
