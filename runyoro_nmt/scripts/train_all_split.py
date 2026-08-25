#!/usr/bin/env python3
"""train_all_split.py - Runs incremental pipeline using subprocess calls to avoid OOM."""
import subprocess, sys, json, os, re, random, unicodedata, shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT.parent / "raw"
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "models" / "checkpoints"
SEED = 42
random.seed(SEED)

RAW_FILES = ["100 sentence pairs 01.xlsx","sentence variations (2).xlsx","sentence pairs (3).xlsx","sentence pair (4).xlsx"]

def clean_text(t):
    t = str(t)
    if t.lower()=="nan" or not t.strip(): return ""
    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"[\u200b\u200c\u200d\ufeff]","",t)
    t = re.sub(r"\s+"," ",t).strip()
    t = re.sub(r"^[-\u2013\u2014]+\s*","",t).strip()
    return t

def is_valid(eng, rny):
    if len(eng)<5 or len(rny)<5: return False
    if eng.lower()=="nan" or rny.lower()=="nan": return False
    if re.search(r"\(v\.\w+\)",eng) or re.search(r"\(v\.\w+\)",rny): return False
    if max(len(eng),len(rny))/max(min(len(eng),len(rny)),1)>8: return False
    return True

def extract_pairs(filepath):
    df = pd.read_excel(filepath, header=None); pairs = []
    if len(df.columns)>=8:
        for _,row in df.iterrows():
            oe=clean_text(row.iloc[2]) if pd.notna(row.iloc[2]) else ""
            or_=clean_text(row.iloc[4]) if pd.notna(row.iloc[4]) else ""
            ve=clean_text(row.iloc[6]) if pd.notna(row.iloc[6]) else ""
            vr=clean_text(row.iloc[7]) if pd.notna(row.iloc[7]) else ""
            if oe and or_ and is_valid(oe,or_): pairs.append((or_,oe))
            if ve and vr and is_valid(ve,vr): pairs.append((vr,ve))
    else:
        df2=pd.read_excel(filepath)
        for _,row in df2.iterrows():
            eng=clean_text(row.get("English",row.get("Original English","")))
            rny=clean_text(row.get("Runyoro-Rutooro (to fill)",row.get("Reference (Runyoro-Rutooro, original tense)","")))
            if eng and rny and is_valid(eng,rny): pairs.append((rny,eng))
    return list(set(pairs))

def dedup(pairs):
    seen=set(); out=[]
    for r,e in pairs:
        k=(str(r).lower().strip(),str(e).lower().strip())
        if k not in seen: seen.add(k); out.append((r,e))
    return out

def split_data(pairs):
    rng=random.Random(SEED); s=pairs.copy(); rng.shuffle(s)
    n=len(s); nt=max(1,int(n*0.05)); nv=max(1,int(n*0.05))
    return s[nt+nv:], s[nt:nt+nv], s[:nt]

def augment(pairs):
    aug=[]
    for rny,eng in pairs:
        wr=rny.split(); we=eng.split()
        if len(wr)>4:
            dr=" ".join(w for w in wr if random.random()>0.05)
            de=" ".join(w for w in we if random.random()>0.05)
            if len(dr.split())>=3 and len(de.split())>=3: aug.append((dr,de))
        if len(wr)>3:
            i=random.randint(0,len(wr)-2); sw=wr.copy()
            sw[i],sw[i+1]=sw[i+1],sw[i]; aug.append((" ".join(sw),eng))
    return aug

def run_phase(script_code):
    """Run a training/BT phase as a separate process."""
    tmp = ROOT / "scripts" / "_phase_tmp.py"
    tmp.write_text(script_code, encoding="utf-8")
    env = os.environ.copy(); env["USE_TF"]="0"; env["TF_CPP_MIN_LOG_LEVEL"]="3"; env["TOKENIZERS_PARALLELISM"]="false"; env["CUDA_VISIBLE_DEVICES"]="0,1"; env["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"; result = subprocess.run([sys.executable, str(tmp)], cwd=str(ROOT.parent), capture_output=False, env=env)
    tmp.unlink()
    return result.returncode == 0

TRAIN_TEMPLATE = '''
import os, sys, torch, gc
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
os.environ["USE_TF"]="0"
os.environ["TOKENIZERS_PARALLELISM"]="false"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
from pathlib import Path
import pandas as pd
from datasets import Dataset as HFDataset, concatenate_datasets
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)
set_seed(42)
pairs_df = pd.read_csv("{pairs_csv}")
pairs = list(zip(pairs_df["Runyoro"].tolist(), pairs_df["English"].tolist()))
print(f"Training {{len(pairs)}} pairs from {src_model}")
tok = AutoTokenizer.from_pretrained("{src_model}")
def tok_fn(ex):
    s=tok(ex["src"],max_length=256,truncation=True,padding=False)
    t=tok(ex["tgt"],max_length=256,truncation=True,padding=False)
    s["labels"]=t["input_ids"]; return s
fwd=HFDataset.from_dict({{"src":[s for s,t in pairs],"tgt":[t for s,t in pairs]}})
fwd=fwd.map(tok_fn,batched=True,remove_columns=["src","tgt"])
rev=HFDataset.from_dict({{"src":[t for s,t in pairs],"tgt":[s for s,t in pairs]}})
rev=rev.map(tok_fn,batched=True,remove_columns=["src","tgt"])
ds=concatenate_datasets([fwd,rev]).shuffle(seed=42)
print(f"Dataset: {{len(ds)}} samples")
model=AutoModelForSeq2SeqLM.from_pretrained("{src_model}",torch_dtype=torch.float32,device_map="auto",max_memory={{0:"12GiB",1:"12GiB"}})
model.gradient_checkpointing_enable()
col=DataCollatorForSeq2Seq(tok,model=model,label_pad_token_id=-100,pad_to_multiple_of=8)
args=Seq2SeqTrainingArguments(output_dir="{out_dir}",num_train_epochs={epochs},per_device_train_batch_size=1,gradient_accumulation_steps=32,learning_rate={lr},warmup_steps=50,weight_decay=0.01,lr_scheduler_type="cosine",fp16=False,bf16=False,save_strategy="epoch",eval_strategy="no",logging_steps=10,save_total_limit=2,predict_with_generate=False,dataloader_num_workers=0,dataloader_pin_memory=True,gradient_checkpointing=True,optim="adamw_torch",save_safetensors=True,save_only_model=True,report_to=["none"])
trainer=Seq2SeqTrainer(model=model,args=args,train_dataset=ds,processing_class=tok,data_collator=col)
trainer.train()
Path("{out_dir}").mkdir(parents=True,exist_ok=True)
model.save_pretrained("{out_dir}"); tok.save_pretrained("{out_dir}")
print("SAVED:", "{out_dir}")
'''

BT_TEMPLATE = '''
import os, sys, torch
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
os.environ["USE_TF"]="0"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
pairs_df = pd.read_csv("{pairs_csv}")
pairs = list(zip(pairs_df["Runyoro"].tolist(), pairs_df["English"].tolist()))
print(f"Back-translating {{len(pairs)}} pairs")
device="cuda"
tok=AutoTokenizer.from_pretrained("{model_path}")
model=AutoModelForSeq2SeqLM.from_pretrained("{model_path}",torch_dtype=torch.float32).to(device)
model.eval()
def tr(texts,bs=16):
    res=[]
    for i in range(0,len(texts),bs):
        b=texts[i:i+bs]
        enc=tok(b,return_tensors="pt",max_length=256,truncation=True,padding=True).to(device)
        with torch.no_grad(): out=model.generate(**enc,num_beams=4,max_length=256)
        res.extend([t.strip() for t in tok.batch_decode(out,skip_special_tokens=True)])
    return res
rny=[r for r,e in pairs]; syn_eng=tr(rny)
bt1=[(r,se) for r,se in zip(rny,syn_eng) if len(se.strip())>3]
eng=[e for r,e in pairs]; syn_rny=tr(eng)
bt2=[(sr,e) for sr,e in zip(syn_rny,eng) if len(sr.strip())>3]
print(f"BT: rny->eng={{len(bt1)}}, eng->rny={{len(bt2)}}")
pd.DataFrame(bt1+bt2,columns=["Runyoro","English"]).to_csv("{out_csv}",index=False)
print("Saved:", "{out_csv}")
'''

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--bt-epochs", type=int, default=10)
    args = parser.parse_args()

    print("="*70)
    print("INCREMENTAL PIPELINE (split processes)")
    print("="*70)

    prev_pairs = []
    prev_model = "facebook/nllb-200-distilled-1.3B"

    for vi, rf in enumerate(RAW_FILES, 1):
        print(f"\n{'='*70}")
        print(f"V{vi}: {rf}")
        print("="*70)

        fp = RAW_DIR / rf
        if not fp.exists(): print(f"  NOT FOUND: {fp}"); continue

        new_p = extract_pairs(fp)
        print(f"  Extracted: {len(new_p)} new pairs")

        all_clean = dedup(prev_pairs + new_p)
        print(f"  Merged: {len(all_clean)} (prev {len(prev_pairs)} + new {len(new_p)})")

        train_p, val_p, test_p = split_data(all_clean)
        print(f"  Split: train={len(train_p)} val={len(val_p)} test={len(test_p)}")

        aug_p = augment(train_p)
        base_train = train_p + aug_p
        random.shuffle(base_train)
        print(f"  Training: {len(base_train)} (orig {len(train_p)} + aug {len(aug_p)})")

        dd = DATA_DIR / f"inc_v{vi}"; dd.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_clean, columns=["Runyoro","English"]).to_csv(dd/"cleaned_pairs.csv", index=False)
        pd.DataFrame(base_train, columns=["Runyoro","English"]).to_csv(dd/"base_training.csv", index=False)
        pd.DataFrame(train_p, columns=["Runyoro","English"]).to_csv(dd/"train_only.csv", index=False)
        pd.DataFrame(val_p, columns=["Runyoro","English"]).to_csv(dd/"val_pairs.csv", index=False)
        pd.DataFrame(test_p, columns=["Runyoro","English"]).to_csv(dd/"test_pairs.csv", index=False)

        src = prev_model
        lr = 5e-5 if vi == 1 else 3e-5
        base_dir = CKPT_DIR / f"runyoro-inc-v{vi}-base"
        final_dir = CKPT_DIR / f"runyoro-inc-v{vi}"

        # Phase 1: Train base
        print(f"\n  Phase 1: Training base V{vi}...")
        code = TRAIN_TEMPLATE.format(pairs_csv=str(dd/"base_training.csv").replace("\\","/"), src_model=str(src).replace("\\","/"), out_dir=str(base_dir).replace("\\","/"), epochs=args.epochs, lr=lr)
        if not run_phase(code):
            print(f"  ERROR: Base training failed for V{vi}"); break

        # Phase 2: Back-translate
        print(f"\n  Phase 2: Back-translating V{vi}...")
        code = BT_TEMPLATE.format(pairs_csv=str(dd/"train_only.csv", model_path=base_dir, out_csv=dd/"back_translated.csv")
        if not run_phase(code):
            print(f"  WARNING: BT failed, using base model as final")
            shutil.copytree(str(base_dir), str(final_dir), dirs_exist_ok=True)
        else:
            # Phase 3: Retrain with BT
            bt_df = pd.read_csv(dd/"back_translated.csv")
            bt_pairs = list(zip(bt_df["Runyoro"].tolist(), bt_df["English"].tolist()))
            final_train = base_train + bt_pairs
            random.shuffle(final_train)
            pd.DataFrame(final_train, columns=["Runyoro","English"]).to_csv(dd/"final_training.csv", index=False)
            print(f"\n  Phase 3: Retraining V{vi} with BT ({len(final_train)} pairs)...")
            code = TRAIN_TEMPLATE.format(pairs_csv=str(dd/"final_training.csv", src_model=base_dir, out_dir=final_dir, epochs=args.bt_epochs, lr=2e-5)
            if not run_phase(code):
                print(f"  WARNING: Final retrain failed, using base")
                shutil.copytree(str(base_dir), str(final_dir), dirs_exist_ok=True)

        # Cleanup base
        if base_dir.exists() and final_dir.exists():
            shutil.rmtree(base_dir)

        meta = {"version":vi,"file":rf,"clean":len(all_clean),"train":len(train_p),"aug":len(aug_p)}
        json.dump(meta, open(dd/"metadata.json","w"), indent=2)

        prev_pairs = all_clean
        prev_model = str(final_dir)
        print(f"\n  V{vi} DONE: {len(all_clean)} pairs, model={final_dir}")

    print(f"\n{'='*70}")
    print(f"ALL DONE! Final model: {prev_model} ({len(prev_pairs)} pairs)")
    print("="*70)

if __name__ == "__main__":
    main()
