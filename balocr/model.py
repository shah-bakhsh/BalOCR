"""Native Transformers model interface. No repository custom code executes."""
from pathlib import Path
import json

BASE_MODEL='PaddlePaddle/PaddleOCR-VL-1.6'
BASE_REVISION='132c45f52c0a26ed5a6b2f6bd8e3985794cc27dd'

def assistant_labels(full_ids,prompt_ids,max_length=4096):
    if len(full_ids)>max_length:
        raise ValueError(f'Example length {len(full_ids)} exceeds {max_length}; not truncated')
    if full_ids[:len(prompt_ids)]!=prompt_ids:
        raise ValueError('Prompt is not an exact token prefix of full conversation')
    if len(full_ids)<=len(prompt_ids):
        raise ValueError('No assistant target tokens')
    return [-100]*len(prompt_ids)+full_ids[len(prompt_ids):]

def conversation(image,text=None):
    result=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':'OCR:'}]}]
    if text is not None:result.append({'role':'assistant','content':[{'type':'text','text':text}]})
    return result

def load_model(device='cpu',adapter=None,allow_pascal=False):
    import torch
    from transformers import AutoConfig,AutoModelForImageTextToText,AutoProcessor
    if device.startswith('cuda'):
        if not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        capability=torch.cuda.get_device_capability(torch.device(device))
        if capability<(7,0) and not allow_pascal:
            raise RuntimeError('Official PaddleOCR-VL requirements are CC>=7.0. Pascal requires an explicit experimental preflight flag.')
    processor=AutoProcessor.from_pretrained(BASE_MODEL,revision=BASE_REVISION,trust_remote_code=False)
    config=AutoConfig.from_pretrained(BASE_MODEL,revision=BASE_REVISION,trust_remote_code=False)
    # The pinned original checkpoint explicitly has untied embeddings. Preserve
    # that setting across the Transformers 5 flat-to-nested config migration.
    config.tie_word_embeddings=False
    config.text_config.tie_word_embeddings=False
    model,loading=AutoModelForImageTextToText.from_pretrained(BASE_MODEL,revision=BASE_REVISION,
                config=config,trust_remote_code=False,dtype=torch.float32,attn_implementation='eager',
                output_loading_info=True)
    if loading.get('missing_keys') or loading.get('unexpected_keys') or loading.get('mismatched_keys'):
        raise RuntimeError('Checkpoint loading mismatch: '+repr(loading))
    model._balocr_loading_info={k:sorted(v) if isinstance(v,set) else v for k,v in loading.items()}
    if adapter:
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,adapter)
    return model.to(device),processor

def encode(processor,image,text=None,max_pixels=401408):
    return processor.apply_chat_template(conversation(image,text),add_generation_prompt=text is None,
               tokenize=True,return_dict=True,return_tensors='pt',
               processor_kwargs={'images_kwargs':{'size':{'shortest_edge':112896,'longest_edge':max_pixels}}})

def predict(model,processor,image,max_new_tokens=1024,max_pixels=401408):
    import torch
    inputs=encode(processor,image,max_pixels=max_pixels).to(model.device)
    with torch.inference_mode():
        output=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,use_cache=True)
    ids=output[0,inputs['input_ids'].shape[-1]:].tolist()
    eos=model.generation_config.eos_token_id
    eos={eos} if isinstance(eos,int) else set(eos or [])
    return dict(text=processor.tokenizer.decode(ids,skip_special_tokens=True),finished=bool(ids and ids[-1] in eos),
                generated_tokens=len(ids),max_new_tokens=max_new_tokens,max_pixels=max_pixels)

def predict_manifest(model,processor,rows,data_root,output,max_new_tokens=1024,max_pixels=401408):
    from PIL import Image
    data_root=Path(data_root).resolve()
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate prediction record IDs')
    with Path(output).open('w',encoding='utf8',newline='\n') as f:
        for row in rows:
            path=(data_root/row['image']).resolve()
            if not path.is_relative_to(data_root):raise ValueError('Image escapes data root')
            with Image.open(path) as im:result=predict(model,processor,im.convert('RGB'),max_new_tokens,max_pixels)
            f.write(json.dumps(dict(id=row['id'],**result),ensure_ascii=False)+'\n')
            f.flush()

def main():
    import argparse
    from PIL import Image
    p=argparse.ArgumentParser()
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--image');mode.add_argument('--manifest')
    p.add_argument('--data-root');p.add_argument('--output',required=True)
    p.add_argument('--adapter');p.add_argument('--device',default='cpu');p.add_argument('--allow-experimental-pascal',action='store_true')
    p.add_argument('--max-new-tokens',type=int,default=1024)
    a=p.parse_args()
    if a.manifest and not a.data_root:p.error('--manifest requires --data-root')
    model,processor=load_model(a.device,a.adapter,a.allow_experimental_pascal)
    model.eval()
    if a.manifest:
        rows=[json.loads(s) for s in Path(a.manifest).read_text(encoding='utf8').splitlines() if s.strip()]
        predict_manifest(model,processor,rows,a.data_root,a.output,a.max_new_tokens)
    else:
        with Image.open(a.image) as image:result=predict(model,processor,image.convert('RGB'),a.max_new_tokens)
        result.update(id=Path(a.image).stem,base_model=BASE_MODEL,base_revision=BASE_REVISION,adapter=a.adapter)
        Path(a.output).write_text(json.dumps(result,ensure_ascii=False)+'\n',encoding='utf8')

if __name__=='__main__':main()
