#!/usr/bin/env python3
"""
train_all_incremental.py - Full incremental pipeline with cleaning, augmentation, back-translation.
"""
import argparse, gc, json, os, re, random, sys, unicodedata, logging, shutil
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(), logging.FileHandler(str(ROOT / "training_bt.log"))])
logger = logging.getLogger("train_all")
import pandas as pd
import torch
from datasets import Dataset as HFDataset, concatenate_datasets
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed)
RAW_DIR = ROOT.parent / "raw"
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "models" / "checkpoints"
BASE_MODEL = "facebook/nllb-200-distilled-1.3B"
MAX_LEN = 256; SEED = 42
random.seed(SEED); set_seed(SEED)
RAW_FILES = ["100 sentence pairs 01.xlsx","sentence variations (2).xlsx","sentence pairs (3).xlsx","sentence pair (4).xlsx"]
def clean_text(t):
    t = str(t)
    if t.lower()=="nan" or not t.strip(): return ""
    t = unicodedata.normalize("NFC", t); t = re.sub(r"[\u200b\u200c\u200d\ufeff]","",t)
    t = re.sub(r"\s+"," ",t).strip(); t = re.sub(r"^[-\u2013\u2014]+\s*","",t).strip()
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
def train_model(pairs, out_dir, src_model, epochs, lr, desc):
    gc.collect(); torch.cuda.empty_cache()
    tok = AutoTokenizer.from_pretrained(src_model)
    def tok_fn(ex):
        s=tok(ex["src"],max_length=MAX_LEN,truncation=True,padding=False)
        t=tok(ex["tgt"],max_length=MAX_LEN,truncation=True,padding=False)
        s["labels"]=t["input_ids"]; return s
    fwd=HFDataset.from_dict({"src":[s for s,t in pairs],"tgt":[t for s,t in pairs]})
    fwd=fwd.map(tok_fn,batched=True,remove_columns=["src","tgt"],desc=f"{desc} fwd")
    rev=HFDataset.from_dict({"src":[t for s,t in pairs],"tgt":[s for s,t in pairs]})
    rev=rev.map(tok_fn,batched=True,remove_columns=["src","tgt"],desc=f"{desc} rev")
    ds=concatenate_datasets([fwd,rev]).shuffle(seed=SEED)
    logger.info("  [%s] Dataset: %d samples", desc, len(ds))
    out_dir.mkdir(parents=True, exist_ok=True)
    n_gpus = torch.cuda.device_count()
    logger.info("  GPUs available: %d", n_gpus)

    # Load model onto GPU 0 in bf16 (~2.6 GB for 1.3B params).
    # The Trainer will wrap it with DataParallel across all visible GPUs automatically.
    # Do NOT use device_map="auto" here — it conflicts with the Trainer's device handling.
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    model = AutoModelForSeq2SeqLM.from_pretrained(src_model, torch_dtype=dtype)
    model.gradient_checkpointing_enable()

    col = DataCollatorForSeq2Seq(tok, model=model, label_pad_token_id=-100, pad_to_multiple_of=8)
    targs = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=16,   # 16 per GPU × 2 GPUs = 32 effective
        gradient_accumulation_steps=1,
        learning_rate=lr,
        warmup_steps=50,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        fp16=not use_bf16,
        bf16=use_bf16,
        save_strategy="epoch",
        eval_strategy="no",
        logging_steps=10,
        save_total_limit=2,
        predict_with_generate=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
        save_safetensors=True,
        save_only_model=True,
        report_to=["none"],
    )
    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=ds, processing_class=tok, data_collator=col)
    trainer.train()
    model.save_pretrained(str(out_dir)); tok.save_pretrained(str(out_dir))
    logger.info("  [%s] Saved: %s", desc, out_dir)
    del model, trainer; gc.collect(); torch.cuda.empty_cache()
    return str(out_dir)
def back_translate(pairs, model_path, bs=16):
    # Inference only — load on GPU 0 in bf16. Single GPU, no NCCL needed.
    gc.collect(); torch.cuda.empty_cache()
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    tok = AutoTokenizer.from_pretrained(model_path)
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=dtype).to(device)
    model.eval()
    def tr(texts):
        res = []
        for i in range(0, len(texts), bs):
            b = texts[i:i+bs]
            enc = tok(b, return_tensors="pt", max_length=MAX_LEN, truncation=True, padding=True).to(device)
            with torch.no_grad():
                out = model.generate(**enc, num_beams=4, max_length=MAX_LEN)
            res.extend([t.strip() for t in tok.batch_decode(out, skip_special_tokens=True)])
            logger.info("  Back-translated %d / %d", min(i+bs, len(texts)), len(texts))
        return res
    rny_texts = [r for r,e in pairs]; syn_eng = tr(rny_texts)
    bt1 = [(r,se) for r,se in zip(rny_texts, syn_eng) if len(se.strip())>3]
    eng_texts = [e for r,e in pairs]; syn_rny = tr(eng_texts)
    bt2 = [(sr,e) for sr,e in zip(syn_rny, eng_texts) if len(sr.strip())>3]
    logger.info("  BT: rny->eng=%d, eng->rny=%d", len(bt1), len(bt2))
    del model; gc.collect(); torch.cuda.empty_cache()
    return bt1+bt2
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--start-from",type=int,default=1)
    parser.add_argument("--epochs",type=int,default=15)
    parser.add_argument("--bt-epochs",type=int,default=10)
    args=parser.parse_args()
    logger.info("="*70)
    logger.info("INCREMENTAL PIPELINE: clean + augment + BT + continue training")
    logger.info("="*70)
    prev_pairs=[]; prev_model=BASE_MODEL
    for vi,rf in enumerate(RAW_FILES,1):
        if vi<args.start_from:
            dd=DATA_DIR/f"inc_v{vi}"
            if (dd/"cleaned_pairs.csv").exists():
                df=pd.read_csv(dd/"cleaned_pairs.csv")
                prev_pairs=[(str(r["Runyoro"]),str(r["English"])) for _,r in df.iterrows()]
                prev_model=str(CKPT_DIR/f"runyoro-inc-v{vi}")
                logger.info("Skip V%d (loaded %d pairs)",vi,len(prev_pairs))
            continue
        logger.info("\n"+"="*70)
        logger.info("V%d: %s",vi,rf)
        logger.info("="*70)
        fp=RAW_DIR/rf
        if not fp.exists(): logger.error("Not found: %s",fp); continue
        new_p=extract_pairs(fp)
        logger.info("  Extracted: %d new pairs",len(new_p))
        all_clean=dedup(prev_pairs+new_p)
        logger.info("  Merged: %d (prev %d + new %d)",len(all_clean),len(prev_pairs),len(new_p))
        train_p,val_p,test_p=split_data(all_clean)
        logger.info("  Split: train=%d val=%d test=%d",len(train_p),len(val_p),len(test_p))
        aug_p=augment(train_p)
        base_train=train_p+aug_p; random.shuffle(base_train)
        logger.info("  Training: %d (orig %d + aug %d)",len(base_train),len(train_p),len(aug_p))
        dd=DATA_DIR/f"inc_v{vi}"; dd.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(all_clean,columns=["Runyoro","English"]).to_csv(dd/"cleaned_pairs.csv",index=False)
        pd.DataFrame(val_p,columns=["Runyoro","English"]).to_csv(dd/"val_pairs.csv",index=False)
        pd.DataFrame(test_p,columns=["Runyoro","English"]).to_csv(dd/"test_pairs.csv",index=False)
        src=prev_model if vi>1 else BASE_MODEL
        lr=5e-5 if vi==1 else 3e-5
        logger.info("\n  Training base V%d...",vi)
        base_dir=CKPT_DIR/f"runyoro-inc-v{vi}-base"
        base_path=train_model(base_train,base_dir,src,args.epochs,lr,f"V{vi}-base")
        logger.info("\n  Back-translating V%d...",vi)
        bt_p=back_translate(train_p,base_path)
        pd.DataFrame(bt_p,columns=["Runyoro","English"]).to_csv(dd/"back_translated.csv",index=False)
        final_train=base_train+bt_p; random.shuffle(final_train)
        pd.DataFrame(final_train,columns=["Runyoro","English"]).to_csv(dd/"final_training.csv",index=False)
        logger.info("\n  Retraining V%d with BT (%d pairs)...",vi,len(final_train))
        final_dir=CKPT_DIR/f"runyoro-inc-v{vi}"
        final_path=train_model(final_train,final_dir,base_path,args.bt_epochs,2e-5,f"V{vi}-final")
        meta={"version":vi,"file":rf,"clean":len(all_clean),"train":len(train_p),"aug":len(aug_p),"bt":len(bt_p),"final":len(final_train),"val":len(val_p),"test":len(test_p)}
        json.dump(meta,open(dd/"metadata.json","w"),indent=2)
        if base_dir.exists(): shutil.rmtree(base_dir)
        prev_pairs=all_clean; prev_model=final_path
        logger.info("\n  V%d DONE: %d pairs, model=%s",vi,len(all_clean),final_dir)
    logger.info("\n"+"="*70)
    logger.info("ALL DONE! Final: %s (%d pairs)",prev_model,len(prev_pairs))
    logger.info("="*70)
if __name__=="__main__":
    main()
