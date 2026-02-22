import torch 
from torch.nn import functional as F

import commons


def feature_loss(fmap_r, fmap_g):
  loss = 0
  for dr, dg in zip(fmap_r, fmap_g):
    for rl, gl in zip(dr, dg):
      rl = rl.float().detach()
      gl = gl.float()
      loss += torch.mean(torch.abs(rl - gl))

  return loss * 2 


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
  loss = 0
  r_losses = []
  g_losses = []
  for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
    dr = dr.float()
    dg = dg.float()
    r_loss = torch.mean((1-dr)**2)
    g_loss = torch.mean(dg**2)
    loss += (r_loss + g_loss)
    r_losses.append(r_loss.item())
    g_losses.append(g_loss.item())

  return loss, r_losses, g_losses


def generator_loss(disc_outputs):
  loss = 0
  gen_losses = []
  for dg in disc_outputs:
    dg = dg.float()
    l = torch.mean((1-dg)**2)
    gen_losses.append(l)
    loss += l

  return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
  """
  z_p, logs_q: [b, h, t_t]
  m_p, logs_p: [b, h, t_t]
  """
  z_p = z_p.float()
  logs_q = logs_q.float()
  m_p = m_p.float()
  logs_p = logs_p.float()
  z_mask = z_mask.float()

  kl = logs_p - logs_q - 0.5
  kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p)
  kl = torch.sum(kl * z_mask)
  l = kl / torch.sum(z_mask)
  return l

def compute_cross_entropy(p, q):
    q = F.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return -loss.mean()


def stablize_logits(logits):
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()
    return logits


def multiposconloss(embs, labels):
    """
    Args:
        embs: Embeddings tensor of shape [B, D]
        labels: Labels tensor of shape [B]
    
    Returns:
        loss: Scalar loss value
    """
    device = embs.device
    batch_size = embs.size(0)
    
    # Normalize embeddings
    feats = F.normalize(embs, dim=-1, p=2)
    
    # Compute similarity matrix
    logits = torch.matmul(feats, feats.T) / 0.1
    
    # Create mask for positive pairs (same label)
    mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float().to(device)
    
    # Remove diagonal (self-similarity)
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask
    
    # Apply mask to logits (set self-similarity to very negative value)
    logits = logits - (1 - logits_mask) * 1e9
    
    # Stabilize logits
    logits = stablize_logits(logits)
    
    # Compute ground-truth distribution
    # Each positive pair gets equal probability
    p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
    
    # Compute loss
    loss = compute_cross_entropy(p, logits)
    
    return loss