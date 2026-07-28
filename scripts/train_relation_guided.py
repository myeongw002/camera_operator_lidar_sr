#!/usr/bin/env python3
"""Train the camera adapter G while keeping a schema-4 L0 source frozen."""
import argparse, json, math
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from camera_operator_sr.data.collate import collate_frames
from camera_operator_sr.data.dataset import ProcessedTrainingDataset
from camera_operator_sr.losses.relation import guided_relation_supervised_loss
from camera_operator_sr.models.relation import CameraGuidedRelationModel, RelationLidarModel
from camera_operator_sr.training.checkpoint import extract_dataset_geometry, load_project_checkpoint, save_checkpoint, validate_checkpoint_geometry
from camera_operator_sr.training.experiment import append_jsonl, assert_resume_compatible, build_manifest, invocation_record, prepare_experiment, source_checkpoint_metadata, write_json_atomic
from camera_operator_sr.training.reproducibility import capture_rng_state, dataloader_generator, seed_everything
from camera_operator_sr.training.resume import is_historical_best, restore_training_state
from camera_operator_sr.training.validation import ValidationRangeAccumulator
from camera_operator_sr.training.modules import generated_mask_for

def move(value, device): return value.to(device) if isinstance(value, torch.Tensor) else {k: move(v, device) for k,v in value.items()} if isinstance(value, dict) else value

def source_model(path, sample, device):
    checkpoint=load_project_checkpoint(path,map_location=device)
    if checkpoint.get("checkpoint_schema_version") != 4 or checkpoint.get("model_config",{}).get("model_type") != "relation_l0": raise ValueError("--l0-checkpoint must be a schema-4 relation_l0 checkpoint")
    validate_checkpoint_geometry(checkpoint,sample)
    config={k:v for k,v in checkpoint["model_config"].items() if k not in {"model_type","candidate_layout","anchor_slots"}}
    model=RelationLidarModel(**config).to(device); model.load_state_dict(checkpoint["model"],strict=True); return model

def main():
    p=argparse.ArgumentParser(allow_abbrev=False)
    for name,kwargs in (("--dataset-root",{"required":True}),("--train-split",{"required":True}),("--val-split",{"required":True}),("--l0-checkpoint",{"required":True,"type":Path}),("--output-root",{"required":True,"type":Path}),("--experiment-name",{"required":True}),("--seed",{"required":True,"type":int})): p.add_argument(name,**kwargs)
    p.add_argument("--epochs",type=int,default=30);p.add_argument("--batch-size",type=int,default=2);p.add_argument("--learning-rate",type=float,default=3e-4);p.add_argument("--weight-decay",type=float,default=0.0);p.add_argument("--camera-point-hidden-dim",type=int,default=16);p.add_argument("--camera-relation-hidden-dim",type=int,default=32);p.add_argument("--camera-correction-limit",type=float,default=3.0);p.add_argument("--camera-correction-reg-weight",type=float,default=1e-3);p.add_argument("--depth-mode",default="correct");p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu");p.add_argument("--deterministic",action="store_true");p.add_argument("--resume",action="store_true");p.add_argument("--overwrite",action="store_true")
    a=p.parse_args(); seed_everything(a.seed,deterministic=a.deterministic)
    train=ProcessedTrainingDataset(a.dataset_root,split_file=a.train_split,depth_mode=a.depth_mode); val=ProcessedTrainingDataset(a.dataset_root,split_file=a.val_split,depth_mode=a.depth_mode)
    l0=source_model(a.l0_checkpoint,train[0],a.device); model=CameraGuidedRelationModel(l0,a.camera_point_hidden_dim,a.camera_relation_hidden_dim,a.camera_correction_limit).to(a.device)
    geometry=extract_dataset_geometry(train[0],candidate_horizontal_radius=model.horizontal_radius).as_dict(); source=source_checkpoint_metadata(l0=a.l0_checkpoint)
    paths=prepare_experiment(output_root=a.output_root,experiment_name=a.experiment_name,seed=a.seed,resume=a.resume,overwrite=a.overwrite)
    manifest=build_manifest(experiment_name=a.experiment_name,experiment_type="relation_guided",seed=a.seed,deterministic=a.deterministic,dataset_root=a.dataset_root,train_split=a.train_split,validation_split=a.val_split,train_frames=len(train),validation_frames=len(val),model_config=model.model_config,geometry=geometry,depth_mode=a.depth_mode,advantage_config=None);manifest["source_checkpoints"]=source
    args={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()}; loss_config={"huber_delta":.1,"camera_correction_reg_weight":a.camera_correction_reg_weight}
    if a.resume: existing=json.loads(paths.manifest.read_text());assert_resume_compatible(existing,manifest);manifest=existing;config=json.loads(paths.config.read_text());index=len(paths.invocations.read_text().splitlines())
    else: config=dict(args,loss_config=loss_config);write_json_atomic(paths.manifest,manifest);write_json_atomic(paths.config,config);index=0
    invocation=invocation_record(invocation_index=index,resume=a.resume,overwrite=a.overwrite,arguments=args);append_jsonl(paths.invocations,invocation)
    metadata={"experiment_name":a.experiment_name,"experiment_type":"relation_guided","seed":a.seed,"deterministic":a.deterministic,"run_config":config,"experiment_config":config,"current_invocation":invocation,"manifest":manifest}
    optimizer=torch.optim.AdamW(model.camera_adapter.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay); generator=dataloader_generator(a.seed); start=step=best_epoch=best_step=0;best=float("inf")
    if a.resume:
        state=restore_training_state(load_project_checkpoint(paths.last_checkpoint,map_location=a.device),model=model,optimizer=optimizer,dataloader_generator=generator,experiment_type="relation_guided");start,step,best,best_epoch,best_step=state.start_epoch,state.global_step,state.best_validation_score,state.best_epoch,state.best_global_step
    loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,collate_fn=collate_frames,generator=generator);vloader=DataLoader(val,batch_size=a.batch_size,collate_fn=collate_frames)
    for epoch in range(start,a.epochs):
        model.train();total=0.
        for batch in loader:
            out=model(move(batch,a.device));loss=guided_relation_supervised_loss(out,move(batch,a.device),camera_correction_reg_weight=a.camera_correction_reg_weight);optimizer.zero_grad(set_to_none=True);loss["loss"].backward();optimizer.step();step+=1;total+=float(loss["loss"].detach())
        model.eval();acc=ValidationRangeAccumulator()
        with torch.no_grad():
            for batch in vloader:
                batch=move(batch,a.device);out=model(batch);mask=generated_mask_for(batch)*batch["target"]["valid"]*out.lidar.has_anchor*out.camera_guidance_valid;acc.update(out.predicted_range,batch["target"]["range"],mask)
        score,count=acc.score(),acc.count;best_now=is_historical_best(score,best)
        if best_now: best,best_epoch,best_step=score,epoch+1,step
        common=dict(epoch=epoch+1,global_step=step,sample=train[0],optimizer=optimizer,dataset_split=a.train_split,depth_mode=a.depth_mode,validation_score=score,validation_count=count,experiment_metadata=metadata,loss_config=loss_config,rng_state=capture_rng_state(generator),best_validation_score=best,best_epoch=best_epoch,best_global_step=best_step,source_checkpoints=source)
        save_checkpoint(paths.last_checkpoint,model,**common)
        if best_now: save_checkpoint(paths.best_checkpoint,model,**common)
        append_jsonl(paths.metrics,{"epoch":epoch+1,"global_step":step,"training_loss":total/max(len(loader),1),"validation_score":score,"validation_count":count,"is_best":best_now})
if __name__=="__main__": main()
