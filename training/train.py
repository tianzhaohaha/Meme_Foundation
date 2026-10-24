"""Legacy binary entry point and shared data/evaluation helpers."""
import argparse, collections, csv, hashlib, io, json, os, random, math
from pathlib import Path
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
import numpy as np
import torch
from PIL import Image, ImageOps
import zipfile, re
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from model import TEXT_MODEL, TEXT_MODEL_REVISION
from tracking import add_tracking_arguments
from experiment_data import write_json as dump

def save_checkpoint(path, state):
    temporary = path.with_suffix('.pt.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)

def make_scheduler(optimizer, total_steps, warmup_ratio):
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    def factor(step):
        if step < warmup_steps:
            return float(step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)

def rng_state(generator):
    numpy_state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': {
                'generator': numpy_state[0], 'keys': numpy_state[1].tolist(), 'position': numpy_state[2],
                'has_gauss': numpy_state[3], 'cached_gaussian': numpy_state[4]},
            'torch': torch.get_rng_state(), 'loader': generator.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng(state, generator):
    random.setstate(state['python'])
    numpy_state = state['numpy']
    if isinstance(numpy_state, dict):
        numpy_state = (numpy_state['generator'], np.asarray(numpy_state['keys'], dtype=np.uint32),
                       numpy_state['position'], numpy_state['has_gauss'], numpy_state['cached_gaussian'])
    np.random.set_state(numpy_state)
    torch.set_rng_state(state['torch']); generator.set_state(state['loader'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])

LABELS = ['negative', 'positive']
MAP = {'negative':0, 'positive':1}

def prepare(data, cache):
    manifest=cache/'manifest.json'
    if manifest.exists(): return json.loads(manifest.read_text())
    selected=list(csv.DictReader((Path(__file__).resolve().parent.parent/'data/manifests/source_binary_20000.tsv').open(),delimiter='\t'))
    assert len(selected)==20000 and len({r['id'] for r in selected})==20000
    train_ids,val_ids=train_test_split([r['id'] for r in selected],test_size=2000,random_state=42,stratify=[r['label'] for r in selected])
    validation=set(val_ids); records=[];invalid=[]
    pattern=r'(?i)#(?:angry|anger|bored|boredom|calm|fear|fearful|happy|happiness|love|loving|sad|sadness)\b'
    with zipfile.ZipFile(data) as z:
        for i,row in enumerate(selected):
            raw=z.read(f"subset_50000/all_data/{row['id']}.jpg");digest=hashlib.sha256(raw).hexdigest()
            path=cache/'images'/f'{digest}.jpg';path.parent.mkdir(parents=True,exist_ok=True)
            try:
                if not path.exists():
                    with Image.open(io.BytesIO(raw)) as im:
                        im=ImageOps.exif_transpose(im).convert('RGBA');bg=Image.new('RGBA',im.size,(128,128,128,255));bg.alpha_composite(im)
                        ImageOps.pad(bg.convert('RGB'),(224,224),method=Image.Resampling.BILINEAR,color=(128,128,128)).save(path,quality=95)
            except (OSError,ValueError) as e:
                invalid.append({'id':row['id'],'error':str(e)});continue
            text=z.read(f"subset_50000/all_data/{row['id']}.txt").decode('utf-8-sig',errors='replace')
            text=re.sub(pattern,' ',text);text=re.sub(r'https?://\S+',' ',text);text=' '.join(text.split())
            records.append({'id':row['id'],'emotion':row['label'],'split':'validation' if row['id'] in validation else 'train','image_sha256':digest,'image':str(path),'text':text,'label':MAP[row['polarity']]})
            if (i+1)%2000==0:print(json.dumps({'event':'preparing','done':i+1,'total':20000}),flush=True)
    val_hashes={r['image_sha256'] for r in records if r['split']=='validation'}
    removed=[r['id'] for r in records if r['split']=='train' and r['image_sha256'] in val_hashes]
    records=[r for r in records if r['id'] not in set(removed)]
    dump(cache/'audit.json',{'invalid':invalid,'removed_source_train_validation_exact_duplicates':removed,'counts':dict(collections.Counter(r['split'] for r in records)),'text_cleaning':'remove emotion hashtags and URLs; no label fields as model input','scope':'Only selected 20000 IDs accessed. Remaining 30000 reserved; cross-task duplicate audit required before future transfer.'})
    dump(manifest,records);return records

class Memes(Dataset):
    def __init__(self, rows, tokenizer, length):
        self.rows=rows
        self.tokens=tokenizer([r['text'] for r in rows],padding='max_length',truncation=True,max_length=length,return_tensors='pt') if tokenizer else {'input_ids':torch.zeros((len(rows),1),dtype=torch.long),'attention_mask':torch.ones((len(rows),1),dtype=torch.long)}
        self.transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        with Image.open(self.rows[i]['image']) as im: image=self.transform(im.convert('RGB'))
        return image,self.tokens['input_ids'][i],self.tokens['attention_mask'][i],self.rows[i]['label']

def evaluate(model,loader,device,labels=None):
    labels = labels or LABELS
    indices = list(range(len(labels)))
    model.eval();truth=[];pred=[];probs=[];loss_sum=0.0;count=0
    with torch.inference_mode():
        for image,ids,mask,y in loader:
            logits=model(image.to(device),ids.to(device),mask.to(device))
            loss_sum += torch.nn.functional.cross_entropy(logits, y.to(device), reduction='sum').item(); count += len(y)
            p=logits.softmax(-1).cpu();probs.extend(p.tolist());pred.extend(p.argmax(-1).tolist());truth.extend(y.tolist())
    return {'loss':loss_sum / max(1,count), 'macro_f1':float(f1_score(truth,pred,labels=indices,average='macro',zero_division=0)),
        'accuracy':float(accuracy_score(truth,pred)),
        'report':classification_report(truth,pred,labels=indices,target_names=labels,output_dict=True,zero_division=0),
        'confusion_matrix':confusion_matrix(truth,pred,labels=indices).tolist()},pred,probs

def run(args,rows,tokenizer,seed):
    from training_engine import run_stage
    return run_stage(args,rows,tokenizer,seed)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('runs_three_class'));p.add_argument('--cache',type=Path,default=Path('prepared'));p.add_argument('--seeds',type=int,nargs='+',default=[42,123,2026]);p.add_argument('--epochs',type=int,default=20);p.add_argument('--patience',type=int,default=4);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--max-length',type=int,default=96);p.add_argument('--encoder-lr',type=float,default=5e-6);p.add_argument('--head-lr',type=float,default=1e-4)
    p.add_argument('--dropout',type=float,default=0.5);p.add_argument('--weight-decay',type=float,default=0.01);p.add_argument('--class-weight-power',type=float,default=0.0);p.add_argument('--label-smoothing',type=float,default=0.0)
    p.add_argument('--warmup-ratio',type=float,default=0.05);p.add_argument('--resume',type=Path)
    p.add_argument('--trust-legacy-checkpoint', action='store_true', help='Allow pickle loading ONLY for a trusted old checkpoint')
    add_tracking_arguments(p)
    args=p.parse_args();
    if not 0 <= args.warmup_ratio < 1: p.error('--warmup-ratio must be in [0,1)')
    if args.resume and len(args.seeds)!=1: p.error('--resume requires exactly one seed')
    args.data=args.data.resolve();args.cache=args.cache.resolve();args.output=args.output.resolve()
    if not args.data.is_file():raise FileNotFoundError(args.data)
    rows=prepare(args.data,args.cache);tokenizer=AutoTokenizer.from_pretrained(TEXT_MODEL,revision=TEXT_MODEL_REVISION)
    results=[run(args,rows,tokenizer,s) for s in args.seeds]
    dump(args.output/'summary.json',{'results':results})
if __name__=='__main__': main()
