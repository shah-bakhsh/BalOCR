"""Bounded, reproducible LoRA adaptation; full labels, single FP32 GPU."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

from .model import BASE_MODEL, BASE_REVISION, load_model, encode, assistant_labels

def main():
    import torch
    from PIL import Image
    from peft import LoraConfig, get_peft_model
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True);p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--allow-experimental-pascal',action='store_true')
    p.add_argument('--max-steps',type=int,default=100);p.add_argument('--accumulation',type=int,default=4)
    p.add_argument('--max-length',type=int,default=4096);p.add_argument('--max-pixels',type=int,default=401408)
    p.add_argument('--learning-rate',type=float,default=5e-5);p.add_argument('--seed',type=int,default=20261110)
    a=p.parse_args()
    if a.max_steps<1 or a.accumulation<1:raise ValueError('Positive training steps and accumulation required')
    output=Path(a.output)
    if (output/'adapter'/'adapter_model.safetensors').exists():raise ValueError('Refusing to overwrite completed adapter')
    output.mkdir(parents=True,exist_ok=True)
    root=Path(a.data_root).resolve()
    all_rows=[json.loads(s) for s in Path(a.manifest).read_text(encoding='utf8').splitlines() if s.strip()]
    if any(r.get('split')!='train' for r in all_rows):raise ValueError('Training manifest contains a held-out record')
    if len({r['id'] for r in all_rows})!=len(all_rows):raise ValueError('Duplicate training record IDs')
    torch.manual_seed(a.seed);random.seed(a.seed)
    model,processor=load_model(a.device,allow_pascal=a.allow_experimental_pascal)
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,bias='none',
        target_modules=r'.*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)$',task_type='CAUSAL_LM'))
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    model.train()
    optimizer=torch.optim.AdamW([w for w in model.parameters() if w.requires_grad],lr=a.learning_rate,weight_decay=0.01)
    ids=[];rejected={};history=[];micro=0;step=0;started=time.time()
    status=dict(phase='training',base_model=BASE_MODEL,base_revision=BASE_REVISION,args=vars(a),
                manifest_sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),
                input_records=len(all_rows),input_by_script=dict(Counter(r['script'] for r in all_rows)),
                trainable_parameters=sum(w.numel() for w in model.parameters() if w.requires_grad),
                total_parameters=sum(w.numel() for w in model.parameters()),dtype='float32',attention='eager',
                experimental_pascal=a.allow_experimental_pascal)
    def save_status(phase):
        record={**status,'phase':phase,'optimizer_steps':step,'microbatches':micro,'elapsed_seconds':time.time()-started,
                'unique_seen_records':len(set(ids)),'seen_ids':sorted(set(ids)),'rejected_records':rejected,'history':history}
        temp=output/'status.tmp.json';temp.write_text(json.dumps(record,indent=2),encoding='utf8');temp.replace(output/'status.json')
    save_status('training')
    try:
        while step<a.max_steps:
            rows=all_rows.copy();random.shuffle(rows)
            usable=0
            for row in rows:
                if row['id'] in rejected:continue
                image_path=(root/row['image']).resolve()
                if not image_path.is_relative_to(root):raise ValueError('Image escapes data root')
                with Image.open(image_path) as im:image=im.convert('RGB')
                batch=encode(processor,image,row['text'],a.max_pixels)
                prompt=encode(processor,image,max_pixels=a.max_pixels)
                try:labels=assistant_labels(batch['input_ids'][0].tolist(),prompt['input_ids'][0].tolist(),a.max_length)
                except ValueError as exc:
                    rejected[row['id']]=str(exc);continue
                usable+=1
                batch['labels']=torch.tensor([labels],dtype=torch.long)
                batch=batch.to(model.device)
                loss=model(**batch,use_cache=False).loss
                if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
                (loss/a.accumulation).backward()
                ids.append(row['id']);micro+=1
                if micro%a.accumulation==0:
                    torch.nn.utils.clip_grad_norm_([w for w in model.parameters() if w.requires_grad],1.0,error_if_nonfinite=True)
                    optimizer.step();optimizer.zero_grad(set_to_none=True);step+=1
                    history.append(dict(step=step,last_microbatch_loss=float(loss.detach().cpu()),seconds=time.time()-started))
                    save_status('training')
                    print(json.dumps(history[-1]),flush=True)
                    if step%50==0:model.save_pretrained(output/'checkpoint',safe_serialization=True)
                    if step>=a.max_steps:break
                del batch,prompt,loss
            if not usable:raise ValueError('No usable complete-label training examples')
        model.save_pretrained(output/'adapter',safe_serialization=True)
        processor.save_pretrained(output/'adapter')
        save_status('complete')
    except BaseException:
        save_status('failed')
        raise

if __name__=='__main__':main()
