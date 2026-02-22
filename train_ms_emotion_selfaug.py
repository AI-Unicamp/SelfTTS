import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
import random

import commons
import utils
from data_utils import (
  TextAudioSpeakerEmotionLoader,
  TextAudioSpeakerEmotionCollate,
  DistributedBucketSampler
)
from models import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
)

from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss,
  multiposconloss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch, spectrogram_torch
from text.symbols import symbols


torch.backends.cudnn.benchmark = True
global_step = 0

cosinecriterion = nn.CosineEmbeddingLoss()
cecriterion = nn.CrossEntropyLoss()

def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '1629'

  hps = utils.get_hparams()
  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)

  train_dataset = TextAudioSpeakerEmotionLoader(hps.data.training_files, hps.data)
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32,300,400,500,600,700,800,900,1000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextAudioSpeakerEmotionCollate()
  train_loader = DataLoader(train_dataset, num_workers=2, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)
  if rank == 0:
    eval_dataset = TextAudioSpeakerEmotionLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(eval_dataset, num_workers=2, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)

  net_g = SynthesizerTrn(
      len(symbols),
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      n_speakers=hps.data.n_speakers,
      n_emotions=hps.data.n_emotions,
      **hps.model).cuda(rank)
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
  optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  net_g = DDP(net_g, device_ids=[rank])
  net_d = DDP(net_d, device_ids=[rank])

  if(hps.train.pretrained_path_g is not None):
    print(f"Restoring generator from pretrained path = {hps.train.pretrained_path_g}")
    checkpoint_g = torch.load(hps.train.pretrained_path_g, map_location='cpu')
    checkpoint_d = torch.load(hps.train.pretrained_path_d, map_location='cpu')

    net_g = utils.load_model(net_g, checkpoint_g)
    net_d = utils.load_model(net_d, checkpoint_d)

    del checkpoint_d, checkpoint_g

  try:
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    epoch_str = 1
    global_step = 0


  if(hps.train.use_emotion_consistency):
    print("Using emotion consistency")

  if(hps.train.use_speaker_consistency):
    print("Using speaker consistency")

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  scaler = GradScaler(enabled=hps.train.fp16_run)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None)
    scheduler_g.step()
    scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
  
  # print("HEY")
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)
  global global_step

  net_g.train()
  net_d.train()
  for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, y_perturbed, y_perturbed_lengths, speakers, emotions) in enumerate(train_loader):
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
    y_perturbed, y_perturbed_lengths = y_perturbed.cuda(rank, non_blocking=True), y_perturbed_lengths.cuda(rank, non_blocking=True)
    speakers = speakers.cuda(rank, non_blocking=True)
    emotions = emotions.cuda(rank, non_blocking=True)

    # --- Self-Augmentation Logic ---
    y_final_gt = y
    speakers_final_gt = speakers
    # melspec_input will be derived from this
    y_for_encoder_input = y 

    if 0 < hps.train.self_aug <= 1:
        batch_size = y.size(0)
        num_aug = int(batch_size * hps.train.self_aug)
        
        if num_aug > 0:
            with torch.no_grad():

                ## Defining the segment size 
                min_len = y_lengths.min().item()
                max_allowed = min_len
                min_allowed = min_len // 2
                segment_size = random.randint(min_allowed, max_allowed)

                ## Turning into melspec
                y_seg, ids_seg = commons.rand_slice_segments_2d(
                    y,
                    x_lengths=y_lengths,
                    segment_size=segment_size
                )

                y_seg = y_seg.squeeze(1)

                melspec = mel_spectrogram_torch(
                      y_seg,
                      hps.data.filter_length, 
                      hps.data.n_mel_channels, 
                      hps.data.sampling_rate, 
                      hps.data.hop_length, 
                      hps.data.win_length, 
                      hps.data.mel_fmin, 
                      hps.data.mel_fmax
                  )

                y_perturbed_seg, ids_perturbed_seg = commons.rand_slice_segments_2d(
                    y_perturbed,
                    x_lengths=y_perturbed_lengths,
                    segment_size=segment_size
                )

                y_perturbed_seg = y_perturbed_seg.squeeze(1)

                perturbed_melspec = mel_spectrogram_torch(
                      y_perturbed_seg,
                      hps.data.filter_length, 
                      hps.data.n_mel_channels, 
                      hps.data.sampling_rate, 
                      hps.data.hop_length, 
                      hps.data.win_length, 
                      hps.data.mel_fmin, 
                      hps.data.mel_fmax
                  )

                # 1. Get current embeddings for the whole batch
                g_current = net_g.module.get_speaker_embedding(sid=speakers, melspec=melspec, y_seg=y).unsqueeze(-1)
                e_current = net_g.module.get_emotion_embedding(eid=emotions, melspec=perturbed_melspec).unsqueeze(-1)

                # 2. Define indices to augment
                aug_indices = torch.randperm(batch_size)[:num_aug].cuda(rank)
                
                # 3. Create target identities (shuffle speakers within the selected indices)
                shuffle_idx = torch.randperm(num_aug).cuda(rank)
                target_speakers_ids = speakers[aug_indices][shuffle_idx]
                g_target = g_current[aug_indices][shuffle_idx]
                
                # 4. Voice Conversion: Source is original, target is same emotion but shuffled speakers
                y_converted, _, _ = net_g.module.voice_conversion_from_embeddings(
                    spec[aug_indices], spec_lengths[aug_indices], 
                    g_src=g_current[aug_indices], g_tgt=g_target, 
                    e_src=e_current[aug_indices], e_tgt=e_current[aug_indices]
                )

                # 5. Apply Batch Masking
                if(hps.train.apply_aug_loc == "gt" or hps.train.apply_aug_loc == "both"):
                    # Replace GT audio and IDs so the model learns to reconstruct the synthetic sample
                    y_final_gt = y.clone()

                    # Check if we need to pad/crop y_converted to match y[aug_indices]
                    current_len = y_converted.size(-1)
                    target_len = y[aug_indices].size(-1)

                    if current_len < target_len:
                        # Pad with zeros at the end
                        y_conv_padded = F.pad(y_converted, (0, target_len - current_len))
                        y_final_gt[aug_indices] = y_conv_padded
                    else:
                        # Crop if it's somehow longer
                        y_final_gt[aug_indices] = y_converted[:, :, :target_len]

                    y_lengths[aug_indices] = current_len
                    speakers_final_gt = speakers.clone()
                    speakers_final_gt[aug_indices] = target_speakers_ids

                    # New specs
                    # --- Spectrogram Alignment ---
                    new_specs = spectrogram_torch(
                        y_converted.squeeze(1), 
                        hps.data.filter_length, hps.data.sampling_rate, 
                        hps.data.hop_length, hps.data.win_length, center=False
                    )
                    
                    current_spec_frames = new_specs.size(-1)
                    target_spec_frames = spec.size(-1)

                    if current_spec_frames < target_spec_frames:
                        new_specs_aligned = F.pad(new_specs, (0, target_spec_frames - current_spec_frames))
                    else:
                        new_specs_aligned = new_specs[:, :, :target_spec_frames]
                        
                    spec[aug_indices] = new_specs_aligned

                    y_for_encoder_input = y_final_gt
                
                elif hps.train.apply_aug_loc == "encoder":
                    # Keep original y as GT, but use synthetic audio for the style encoder inputs
                    y_for_encoder_input = y.clone()

                    current_len = y_converted.size(-1)
                    target_len = y[aug_indices].size(-1)

                    if current_len < target_len:
                        # Pad with zeros at the end
                        y_conv_padded = F.pad(y_converted, (0, target_len - current_len))
                        y_for_encoder_input[aug_indices] = y_conv_padded
                    else:
                        # Crop if it's somehow longer
                        y_for_encoder_input[aug_indices] = y_converted[:, :, :target_len]


    # Get the inputs for speaker and style encoders

    ## Defining the segment size 
    min_len = y_lengths.min().item()
    max_allowed = min_len
    min_allowed = min_len // 2
    segment_size = random.randint(min_allowed, max_allowed)


    if(hps.train.apply_aug_loc == "encoder"):

      ## If using self augmentation only for the encoder use the original y to generate the melspec for speaker encoder
      y_seg, ids_seg = commons.rand_slice_segments_2d(
          y,
          x_lengths=y_lengths,
          segment_size=segment_size
      )

      y_seg = y_seg.squeeze(1)

      melspec = mel_spectrogram_torch(
            y_seg,
            hps.data.filter_length, 
            hps.data.n_mel_channels, 
            hps.data.sampling_rate, 
            hps.data.hop_length, 
            hps.data.win_length, 
            hps.data.mel_fmin, 
            hps.data.mel_fmax
        )


      # Use the augmented/original audio for encoder inputs
      y_perturbed_seg, ids_perturbed_seg = commons.rand_slice_segments_2d(y_for_encoder_input, y_lengths, segment_size)
      y_perturbed_seg = y_perturbed_seg.squeeze(1)

      perturbed_melspec = mel_spectrogram_torch(y_perturbed_seg, hps.data.filter_length, hps.data.n_mel_channels, 
                                      hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, 
                                      hps.data.mel_fmin, hps.data.mel_fmax)
    else:

      ## If using the self aug for groundtruth wav, use the same speakers new wavs
      y_seg, ids_seg = commons.rand_slice_segments_2d(
          y_for_encoder_input,
          x_lengths=y_lengths,
          segment_size=segment_size
      )

      y_seg = y_seg.squeeze(1)

      melspec = mel_spectrogram_torch(
            y_seg,
            hps.data.filter_length, 
            hps.data.n_mel_channels, 
            hps.data.sampling_rate, 
            hps.data.hop_length, 
            hps.data.win_length, 
            hps.data.mel_fmin, 
            hps.data.mel_fmax
        )

      if(hps.train.apply_aug_loc == "gt"):
        # Use the augmented/original audio for encoder inputs
        y_perturbed_seg, ids_perturbed_seg = commons.rand_slice_segments_2d(y_for_encoder_input, y_lengths, segment_size)
        y_perturbed_seg = y_perturbed_seg.squeeze(1)

        perturbed_melspec = mel_spectrogram_torch(y_perturbed_seg, hps.data.filter_length, hps.data.n_mel_channels, 
                                        hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, 
                                        hps.data.mel_fmin, hps.data.mel_fmax)
                                        
      elif(hps.train.apply_aug_loc == "both"):
        # if using data aug in both paths keep the emotion as the original wavs
        y_perturbed_seg, ids_perturbed_seg = commons.rand_slice_segments_2d(y, y_lengths, segment_size)
        y_perturbed_seg = y_perturbed_seg.squeeze(1)

        perturbed_melspec = mel_spectrogram_torch(y_perturbed_seg, hps.data.filter_length, hps.data.n_mel_channels, 
                                        hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, 
                                        hps.data.mel_fmin, hps.data.mel_fmax)

    with autocast(enabled=hps.train.fp16_run):
      y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
      (z, z_p, m_p, logs_p, m_q, logs_q),  g, e, zp_pred_spk, zp_pred_emo, spk_pred_from_emo_emb, emo_pred_from_spk_emb, g_logits, e_logits = net_g(x, x_lengths, spec, spec_lengths, speakers_final_gt, emotions, melspec, perturbed_melspec, y_seg, y_perturbed_seg)

      mel = spec_to_mel_torch(
          spec, 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate,
          hps.data.mel_fmin, 
          hps.data.mel_fmax)
      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
      y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1), 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate, 
          hps.data.hop_length, 
          hps.data.win_length, 
          hps.data.mel_fmin, 
          hps.data.mel_fmax
      )

      y = commons.slice_segments(y_final_gt, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 

      # Getting emotion and speaker consistency assets if using it
      if(hps.train.use_emotion_consistency):
        with autocast(enabled=False):
          emotion_embs_gt = net_g.module.get_emotion_embedding(melspec = y_mel).detach()
          emotion_embs_resynth = net_g.module.get_emotion_embedding(melspec = y_hat_mel)
        
      if(hps.train.use_speaker_consistency):
        with autocast(enabled=False):
          speaker_embs_gt = net_g.module.get_speaker_embedding(melspec = y_mel).detach()
          speaker_embs_resynth = net_g.module.get_speaker_embedding(melspec = y_hat_mel)

      # Discriminator
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
      with autocast(enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    with autocast(enabled=hps.train.fp16_run):
      # Generator
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
      with autocast(enabled=False):
        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)

        if(hps.model.grl_type is not None):
          # GRL losses 
          if(hps.model.grl_type == "classifier"):
            loss_spk_from_emo = F.cross_entropy(spk_pred_from_emo_emb, speakers_final_gt) * hps.train.c_grl_spkemo
            loss_emo_from_spk = F.cross_entropy(emo_pred_from_spk_emb, emotions) * hps.train.c_grl_spkemo
            loss_zp_spk = F.cross_entropy(zp_pred_spk, speakers_final_gt) * hps.train.c_grl_zp
            loss_zp_emo = F.cross_entropy(zp_pred_emo, emotions) * hps.train.c_grl_zp
          else: # cosine similarity loss
            device = spk_pred_from_emo_emb.device
            loss_spk_from_emo = cosinecriterion(spk_pred_from_emo_emb, g.detach(), torch.Tensor(spk_pred_from_emo_emb.size(0))
                                  .to(device).fill_(1.0)) * hps.train.c_grl_spkemo
            loss_emo_from_spk = cosinecriterion(emo_pred_from_spk_emb, e.detach(), torch.Tensor(emo_pred_from_spk_emb.size(0))
                                  .to(device).fill_(1.0)) * hps.train.c_grl_spkemo
            loss_zp_spk = cosinecriterion(zp_pred_spk, g.detach(), torch.Tensor(zp_pred_spk.size(0))
                                  .to(device).fill_(1.0)) * hps.train.c_grl_zp
            loss_zp_emo = cosinecriterion(zp_pred_emo, e.detach(), torch.Tensor(zp_pred_emo.size(0))
                                  .to(device).fill_(1.0)) * hps.train.c_grl_zp
        else: # torch with item() zero

          device = g.device  # or x.device / rank device

          loss_spk_from_emo = torch.zeros((), device=device)
          loss_emo_from_spk = torch.zeros((), device=device)
          loss_zp_spk = torch.zeros((), device=device)
          loss_zp_emo = torch.zeros((), device=device)

        loss_grl = loss_spk_from_emo + loss_emo_from_spk + loss_zp_spk + loss_zp_emo

        # mulposconloss loss in latent space for g and e embeddings
        if hps.model.encoder_type == "contrastive":
          con_spk = multiposconloss(g, speakers_final_gt.unsqueeze(-1)) * hps.train.c_con_spk
          con_emo = multiposconloss(e, emotions.unsqueeze(-1)) * hps.train.c_con_emo
        elif hps.model.encoder_type == "ce":
          con_spk = cecriterion(g_logits, speakers_final_gt) * hps.train.c_con_spk
          con_emo = cecriterion(e_logits, emotions) * hps.train.c_con_emo
        else:
          con_spk = torch.zeros((), device=device)
          con_emo = torch.zeros((), device=device)

        # emotion and speaker consistency
        if hps.train.use_emotion_consistency:
          # Emotion consistency loss
          emotion_consistency_loss = -torch.nn.functional.cosine_similarity(emotion_embs_gt, emotion_embs_resynth).mean()
        else:
          emotion_consistency_loss = torch.zeros((), device=device)

        if hps.train.use_speaker_consistency:
          # Speaker consistency loss
          speaker_consistency_loss = -torch.nn.functional.cosine_similarity(speaker_embs_gt, speaker_embs_resynth).mean()
        else:
          speaker_consistency_loss = torch.zeros((), device=device)

        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl + loss_grl + con_spk + con_emo + speaker_consistency_loss + emotion_consistency_loss


        # loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl, loss_grl, con_spk, con_emo, emotion_consistency_loss, speaker_consistency_loss]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [global_step, lr])
        
        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl, "loss/g/grl": loss_grl, "loss/g/con_spk": con_spk, "loss/g/con_emo": con_emo, "loss/g/consistency_emo": emotion_consistency_loss, "loss/g/consistency_spk": speaker_consistency_loss})

        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
        image_dict = { 
            "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
            "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
            "all/attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
        }
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images=image_dict,
          scalars=scalar_dict)

      if global_step % hps.train.eval_interval == 0:
        evaluate(hps, net_g, eval_loader, writer_eval)
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, y_perturbed, y_perturbed_lengths, speakers, emotions) in enumerate(eval_loader):
        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
        y, y_lengths = y.cuda(0), y_lengths.cuda(0)
        y_perturbed, y_perturbed_lengths = y_perturbed.cuda(0), y_perturbed_lengths.cuda(0)
        speakers = speakers.cuda(0)
        emotions = emotions.cuda(0)

        melspec = mel_spectrogram_torch(
              y.squeeze(1),
              hps.data.filter_length, 
              hps.data.n_mel_channels, 
              hps.data.sampling_rate, 
              hps.data.hop_length, 
              hps.data.win_length, 
              hps.data.mel_fmin, 
              hps.data.mel_fmax
          )

        perturbed_melspec = mel_spectrogram_torch(
              y_perturbed.squeeze(1),
              hps.data.filter_length, 
              hps.data.n_mel_channels, 
              hps.data.sampling_rate, 
              hps.data.hop_length, 
              hps.data.win_length, 
              hps.data.mel_fmin, 
              hps.data.mel_fmax
          )

        # remove else
        x = x[:1]
        x_lengths = x_lengths[:1]
        spec = spec[:1]
        spec_lengths = spec_lengths[:1]
        y = y[:1]
        y_lengths = y_lengths[:1]
        y_perturbed = y_perturbed[:1]
        y_perturbed_lengths = y_perturbed_lengths[:1]
        speakers = speakers[:1]
        emotions = emotions[:1]
        break

      y_hat, attn, mask, *_ = generator.module.infer(x, x_lengths, speakers, emotions, melspec, perturbed_melspec, y, y_perturbed, max_len=1000)
      y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

      mel = spec_to_mel_torch(
        spec, 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate,
        hps.data.mel_fmin, 
        hps.data.mel_fmax)
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )
    image_dict = {
      "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
      "gen/audio": y_hat[0,:,:y_hat_lengths[0]]
    }
    if global_step == 0:
      image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
      audio_dict.update({"gt/audio": y[0,:,:y_lengths[0]]})

    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()

                           
if __name__ == "__main__":
  main()
