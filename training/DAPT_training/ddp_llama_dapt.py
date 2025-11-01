"""
ddp_llama_dapt.py
-----------------
Minimal-but-production-ready PyTorch DDP training loop for Domain-Adaptive
Pretraining (CLM) on LLaMA/Gemma/Mistral style models.

Key ideas shown, line-by-line:
- torch.distributed.init_process_group + rank/local_rank/world_size
- set device per process
- DistributedSampler + set_epoch for shuffling
- Tokenization → pack to fixed block_size for throughput (causal LM)
- AMP (autocast + GradScaler), gradient accumulation
- all_reduce to get global mean loss for logging
- Save/resume checkpoints (model/optimizer/scheduler/scaler)

This script reads JSON/JSONL with {"instruction","input",...} and ignores
"output"/"pubid". It packs token ids into blocks of length `block_size`
for CLM next-token prediction.

Launch:
  torchrun --nproc_per_node=4 ddp_llama_dapt.py \
    --train_json ../../data/processed/pubmedqa_full/dapt_200k.json \
    --model_name meta-llama/Meta-Llama-3-8B \
    --tokenizer_name meta-llama/Meta-Llama-3-8B \
    --output_dir ../experiments/llama3_8b_ddp_dapt

Author: BIONEXUS-QA
"""

import os,json,math,time,random,yaml
from pathlib import Path
from typing import List,Dict,Any,Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler, TensorDataset

from transformers import(
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)

def is_dist()->bool:
    return dist.is_available() and dist.is_initialized()

def is_main()->bool:
    return (not is_dist()) or dist.get_rank()==0

def barrier():
    if is_dist():
        dist.barrier()


def log0(msg:str):
    if is_main():
        print(msg,flush=True)

def set_seed(seed:int,rank:int):
    random.seed(seed+rank)
    torch.manual_seed(seed+rank)
    torch.cuda.manual_seed_all(seed+rank)

def human(n:int)->str:
    if n >= 1e9: return f"{n/1e9:.2f}B"
    if n >= 1e6: return f"{n/1e6:.2f}M"
    if n >= 1e3: return f"{n/1e3:.2f}K"
    return str(n)

def save_ckpt(path:str,model,optim,sched,scaler,step:int,epoch:int):
    os.makedirs(path,exist_ok=True)
    state={
        "model":(model.module if isinstance(model,DDP) else model ).state_dict(),
        "optimizer":optim.state_dict(),
        "scheduler":sched.state_dict(),
        "scaler":(scaler.state_dict() if scaler is not None else None),
        "step":step,
        "epoch": epoch,
    }

    torch.save(state,os.path.join(path,"checkpoint.pt"))



def load_ckpt(path:str,model,optim,sched,scaler):
    ck=torch.load(os.path.join(path,"checkpoint,.pt"),map_location="cpu")
    (model.module if isinstance(model,DDP) else model).load_state_dict(ck["model"],strict=False)
    optim.load_state_dict(ck["optimizer"])
    sched.load_state_dict(ck["scheduler"])
    if scaler is not None and ck.get("scaler"): scaler.load_state_dict(ck["scaler"])
    return ck["step"],ck["epoch"]



#=====================Data=============

def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".jsonl"): return [json.loads(l) for l in f]
        return json.load(f)

def _compose_text(ex:Dict[str,Any])->str:
    a=(ex.get("instruction") or "").strip()
    b=(ex.get("input") or "").strip()
    return (a+("\n" if a and b else "")+b).strip()

@torch.no_grad()
def build_packed_dataset(json_path:str, tokenizer, block_size:int, min_chars:int):
    records= _read_json_or_jsonl(json_path)
    texts=[]
    for ex in records:
        t=_compose_text(ex)
        if len(t)>=min_chars: texts.append(t)
    
    if not texts: raise ValueError("No usable texts after filtering")
    log0(f"[DATA] {len(texts):,} texts → tokenize…")

    enc=tokenizer(texts, add_special_tokens=False)
    ids=enc["input_ids"]  #list[list[int]]
    total_tokens=sum(len(x) for x in ids)
    log0(f"[DATA] total tokens pre-pack: {human(total_tokens)}")

    flat=[]
    for seq in ids:
        flat.extend(seq)
    usable=(len(flat) // block_size)*block_size
    flat=flat[:usable]

    input_ids = torch.tensor(flat, dtype=torch.long).view(-1, block_size)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    labels = input_ids.clone()
    ds = TensorDataset(input_ids, attention_mask, labels)
    log0(f"[DATA] packed blocks: {len(ds):,} × {block_size}")
    return ds


#===================Main_traning==============


def main():

    #==============Config==========
    cfg_path=os.environ.get("CONFIG_PATH","./config.yaml")
    with open(cfg_path,"r") as f:
        C= yaml.safe_load(f)
    
    out_dir=Path(C["runtime"]["output_dir"])
    out_dir.mkdir(parents=True,exist_ok=True)
    if is_main():
        with open(out_dir/"config_used.yaml", "w") as f: yaml.safe_dump(C,f)
    

    #=================DDP init=========
    dist.init_process_group("nccl")
    rank=dist.get_rank(); world=dist.get_world_size()
    local_rank=int(os.environ.get("LOCAL_RANK","0"))
    torch.cuda.set_device(local_rank)
    device=torch.device("cuda",local_rank)

    #===========Seed=============
    set_seed(int(C["runtime"]["seed"]), rank)
    
    # ---- Tokenizer/Model ----
    tok_name = C["model"]["tokenizer_name_or_path"] or C["model"]["name_or_path"]
    tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    tok.padding_side = "right"
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    
    cfg=AutoConfig.from_pretrained(C["model"]["name_or_path"])
    model=AutoModelForCausalLM.from_pretrained(C["model"]["name_or_path"],config=cfg)
    model.resize_token_embeddings(len(tok))
    model.to(device)

    if is_main():
        tot=sum(p.numel() for p in model.parameters())
        trn=sum(p.numel() for p in model.parameters() if p.requires_grad)
        log0(f"[MODEL] total={human(tot)} trainable{human(trn)}")

    #=============Data========== 
    ds=build_packed_dataset(
        C["data"]["train_json_path"],
        tokenizer=tok,
        block_size=int(C["data"]["block_size"])
        min_chars=int(C["data"]["min_chars"])

    )

    sampler=DistributedSampler(ds,num_replicas=world,rank=rank,shuffle=True,drop_last=True)
    loader=DataLoader(
        ds,
        batch_size=int(C["optim"]["per_device_batch_size"])
        sampler=sampler,
        num_workers=int(C["runtime"]["num_workers"])
        pin_memory=True,
    )


    #=============Optim&Sched=========

    lr=float(C["optim"]["lr"]); wd=float(C["optim"]["weight_decay"])
    optim=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd, betas=(0.9,0.999), eps=1e-8)
    
    steps_per_epoch=math.floor(len(loader)/int(C["optim"]["grad_accum"]))
    t_total=steps_per_epoch*int(C["optim"]["epochs"])
    warmup=int(t_total* float(C["optim"]["warmup_ratio"]))
    sched=get_cosine_schedule_with_warmup(optim,num_warmup_steps=warmup, num_training_steps=t_total)

    use_fp16= bool(C["precision"]["fp16"])
    use_bf16= bool(C["precision"]["bf16"])
    
    scaler=torch.cuda.amp.grad_scaler(enabled=use_fp16 or use_bf16)
    dtype=torch.float16 if use_fp16 else (torch.bfloat if use_bf16 else torch.float32 )
    
    model=DDP(model, device_ids=[local_rank],output_device=local_rank,find_unused_parameters=False)
    
    #=======Resume========
    global_step,start_epoch=0,0
    ckpt_path=out_dir/ "checkpoint.pt"

    if bool(C["runtime"]["resume"]) and ckpt_path.exist():
        if is_main(): log0(f"[CKPT] Resuming from {ckpt_path}")
        global_step,start_epoch=load_ckpt(str(out_dir),model,optim,sched,scaler)
    
    #=====Preemption-safe save=======
    interrupted={"flag":False}
    def _handler(signum,frame):
        interrupted["flag"]=True
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT,_handler)

    grad_accum=int(C["optim"]["grad_accum"])
    max_grad_norm=float(C["optim"]["max_grad_norm"])
    log_every=int(C["logging"]["log_every"])
    save_every=int(C["logging"]["save_every"])
    epochs=int(C["optim"]["epochs"])

    tok_per_opt=int(C["optim"]["per_device_batch_size"])*int(C["data"]["block_size"])*world
    if is_main():
        log0(f"[TRAIN] world={world} per_dev_bs={C['optim']['per_device_batch_size']} grad_accum={grad_accum} "
             f"tokens/opt≈{tokens_per_opt*grad_accum:,}")
        
    running=0.0
    for epoch in range(start_epoch,epochs):
        sampler.set_epoch(epoch)
        t0=time.time()
        model.train()
        optim.zero_grad(set_to_none=True)

        for it,batch in enumerate(loader):
            if interrupted["flag"]:
                if is_main():
                    log0("[SIGNAL] Caught termination,saving checkpoint...")
                
                if is_main():
                    save_ckpt(str(out_dir),model,optim,sched,scaler,global_step,epoch)
            
            input_ids,attn,labels=[x.to(device,non_blocking=True) for x in batch]
            with torch.autocast(device_type="cuda", dtype=dtype,enabled=use_fp16 or use_bf16):
                out=model(input_ids=input_ids,attention=attn,labels=labels)
                loss=out.loss/grad_accum
            
            scaler.scale(loss).backward()
            running+=loss.detach().float()

            if (it+1)%grad_accum==0:
                scaler.unscale(optim)
                torch.nn.utils.clip_grad_norm(model.parameters(),max_grad_norm)
                scaler.step(optim); scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()

                global_step+=1

                #====Global avg ========
                avg=running.clone()
                dist.all_reduce(avg,op=dist.ReduceOp.SUM)
                avg=(avg/world).item()
                running=0.0

                if is_main() and (global_step % log_every == 0):
                    dt = time.time() - t0
                    tps = (tokens_per_opt * log_every) / max(dt, 1e-6)
                    log0(f"[E{epoch+1}/{epochs} S{global_step}/{t_total}] "
                         f"loss={avg:.4f} lr={sched.get_last_lr()[0]:.2e} "
                         f"throughput≈{int(tps):,} tok/s")
                    t0 = time.time()

                if is_main() and save_every > 0 and (global_step % save_every == 0):
                    save_ckpt(str(out_dir), model, optim, sched, scaler, global_step, epoch)

            if is_main():
                save_ckpt(str(out_dir),model,optim,sched,scaler,global_step,epoch+1)
            
        if is_main():
        # Final HF-style save
        (model.module).save_pretrained(out_dir)
        tok.save_pretrained(out_dir)
        log0(f"[DONE] Saved model+tokenizer to {out_dir}")

    barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()












