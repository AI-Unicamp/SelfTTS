import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
import attentions
import monotonic_align

from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights, get_padding
import style_encoder 
import speaker_encoder 
import modules_grl 

class StochasticDurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, n_flows=4, gin_channels=0, use_emotion=False):
    super().__init__()
    filter_channels = in_channels # it needs to be removed from future version.
    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.n_flows = n_flows
    self.gin_channels = gin_channels
    self.use_emotion = use_emotion

    self.log_flow = modules.Log()
    self.flows = nn.ModuleList()
    self.flows.append(modules.ElementwiseAffine(2))
    for i in range(n_flows):
      self.flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.flows.append(modules.Flip())

    self.post_pre = nn.Conv1d(1, filter_channels, 1)
    self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.post_convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    self.post_flows = nn.ModuleList()
    self.post_flows.append(modules.ElementwiseAffine(2))
    for i in range(4):
      self.post_flows.append(modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3))
      self.post_flows.append(modules.Flip())

    self.pre = nn.Conv1d(in_channels, filter_channels, 1)
    self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
    self.convs = modules.DDSConv(filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout)
    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, filter_channels, 1)
      if self.use_emotion:
        self.cond2 = nn.Conv1d(gin_channels, filter_channels, 1)

  def forward(self, x, x_mask, w=None, g=None, e=None, reverse=False, noise_scale=1.0):
    x = torch.detach(x)
    x = self.pre(x)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)

    if e is not None:
      e = torch.detach(e)
      x = x + self.cond2(e)

    x = self.convs(x, x_mask)
    x = self.proj(x) * x_mask

    if not reverse:
      flows = self.flows
      assert w is not None

      logdet_tot_q = 0 
      h_w = self.post_pre(w)
      h_w = self.post_convs(h_w, x_mask)
      h_w = self.post_proj(h_w) * x_mask
      e_q = torch.randn(w.size(0), 2, w.size(2)).to(device=x.device, dtype=x.dtype) * x_mask
      z_q = e_q
      for flow in self.post_flows:
        z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
        logdet_tot_q += logdet_q
      z_u, z1 = torch.split(z_q, [1, 1], 1) 
      u = torch.sigmoid(z_u) * x_mask
      z0 = (w - u) * x_mask
      logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1,2])
      logq = torch.sum(-0.5 * (math.log(2*math.pi) + (e_q**2)) * x_mask, [1,2]) - logdet_tot_q

      logdet_tot = 0
      z0, logdet = self.log_flow(z0, x_mask)
      logdet_tot += logdet
      z = torch.cat([z0, z1], 1)
      for flow in flows:
        z, logdet = flow(z, x_mask, g=x, reverse=reverse)
        logdet_tot = logdet_tot + logdet
      nll = torch.sum(0.5 * (math.log(2*math.pi) + (z**2)) * x_mask, [1,2]) - logdet_tot
      return nll + logq # [b]
    else:
      flows = list(reversed(self.flows))
      flows = flows[:-2] + [flows[-1]] # remove a useless vflow
      z = torch.randn(x.size(0), 2, x.size(2)).to(device=x.device, dtype=x.dtype) * noise_scale
      for flow in flows:
        z = flow(z, x_mask, g=x, reverse=reverse)
      z0, z1 = torch.split(z, [1, 1], 1)
      logw = z0
      return logw


class DurationPredictor(nn.Module):
  def __init__(self, in_channels, filter_channels, kernel_size, p_dropout, gin_channels=0, use_emotion = False):
    super().__init__()

    self.in_channels = in_channels
    self.filter_channels = filter_channels
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.gin_channels = gin_channels
    self.use_emotion = use_emotion

    self.drop = nn.Dropout(p_dropout)
    self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_1 = modules.LayerNorm(filter_channels)
    self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
    self.norm_2 = modules.LayerNorm(filter_channels)
    self.proj = nn.Conv1d(filter_channels, 1, 1)

    if gin_channels != 0:
      self.cond = nn.Conv1d(gin_channels, in_channels, 1)
      if self.use_emotion:
        self.cond2 = nn.Conv1d(gin_channels, filter_channels, 1)


  def forward(self, x, x_mask, g=None, e=None):
    x = torch.detach(x)
    if g is not None:
      g = torch.detach(g)
      x = x + self.cond(g)

    if e is not None:
      e = torch.detach(e)
      x = x + self.cond2(e)

    x = self.conv_1(x * x_mask)
    x = torch.relu(x)
    x = self.norm_1(x)
    x = self.drop(x)
    x = self.conv_2(x * x_mask)
    x = torch.relu(x)
    x = self.norm_2(x)
    x = self.drop(x)
    x = self.proj(x * x_mask)
    return x * x_mask


class TextEncoder(nn.Module):
  def __init__(self,
      n_vocab,
      out_channels,
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout):
    super().__init__()
    self.n_vocab = n_vocab
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout

    self.emb = nn.Embedding(n_vocab, hidden_channels)
    nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

    self.encoder = attentions.Encoder(
      hidden_channels,
      filter_channels,
      n_heads,
      n_layers,
      kernel_size,
      p_dropout)
    self.proj= nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths):
    x = self.emb(x) * math.sqrt(self.hidden_channels) # [b, t, h]
    x = torch.transpose(x, 1, -1) # [b, h, t]
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

    x = self.encoder(x * x_mask, x_mask)
    stats = self.proj(x) * x_mask

    m, logs = torch.split(stats, self.out_channels, dim=1)
    return x, m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0,
      use_emotion=False):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels
    self.use_emotion = use_emotion

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True, use_emotion=self.use_emotion))
      self.flows.append(modules.Flip())

  def forward(self, x, x_mask, g=None, e=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, e=e, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, e=e, reverse=reverse)
    return x

class ResidualCouplingBlockSNAC(nn.Module):
  def __init__(self,
      channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      n_flows=4,
      gin_channels=0):
    super().__init__()
    self.channels = channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.n_flows = n_flows
    self.gin_channels = gin_channels

    self.flows = nn.ModuleList()
    for i in range(n_flows):
      self.flows.append(modules.ResidualCouplingLayerSNAC(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
      self.flows.append(modules.Flip())

  def forward(self, x, x_mask, g=None, e=None, reverse=False):
    if not reverse:
      for flow in self.flows:
        x, _ = flow(x, x_mask, g=g, e=e, reverse=reverse)
    else:
      for flow in reversed(self.flows):
        x = flow(x, x_mask, g=g, e=e, reverse=reverse)
    return x


class PosteriorEncoder(nn.Module):
  def __init__(self,
      in_channels,
      out_channels,
      hidden_channels,
      kernel_size,
      dilation_rate,
      n_layers,
      gin_channels=0,
      use_emotion=False):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_channels = hidden_channels
    self.kernel_size = kernel_size
    self.dilation_rate = dilation_rate
    self.n_layers = n_layers
    self.gin_channels = gin_channels
    self.use_emotion = use_emotion

    self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
    self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, use_emotion=self.use_emotion)
    self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

  def forward(self, x, x_lengths, g=None, e=None):
    x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
    x = self.pre(x) * x_mask
    x = self.enc(x, x_mask, g=g, e=e)
    stats = self.proj(x) * x_mask
    m, logs = torch.split(stats, self.out_channels, dim=1)
    z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
    return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0, use_emotion=False):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        self.use_emotion = use_emotion

        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
            if self.use_emotion:
              self.cond2 = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
    def forward(self, x, g=None, e=None):
        x = self.conv_pre(x)
        if g is not None:
          x = x + self.cond(g)
        if e is not None:
          x = x + self.cond2(e)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs



class SynthesizerTrn(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(self, 
    n_vocab,
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock, 
    resblock_kernel_sizes, 
    resblock_dilation_sizes, 
    upsample_rates, 
    upsample_initial_channel, 
    upsample_kernel_sizes,
    n_speakers=0,
    speaker_embedding_type=None,
    n_emotions=0,
    emotion_embedding_type=None,
    grl_type=None,
    encoder_type="contrastive", # contrastive use MPCL ce uses CE
    gin_channels=0,
    use_sdp=True,
    use_snac=False, # Use Speaker Normalization Affine 
    filter_mel_bin=0, # Whether to filter or not the input melspec as in Prosospeech and MegaTTS
    **kwargs):

    super().__init__()
    self.n_vocab = n_vocab
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.n_speakers = n_speakers
    self.gin_channels = gin_channels
    self.n_emotions = n_emotions
    self.speaker_embedding_type = speaker_embedding_type
    self.emotion_embedding_type = emotion_embedding_type
    self.grl_type = grl_type
    self.encoder_type = encoder_type
    self.use_snac = use_snac
    self.filter_mel_bin = filter_mel_bin

    if(self.n_emotions > 1):
      self.use_emotion = True
    else:
      self.use_emotion = False
    
  
    print(f"Use emotion = {self.use_emotion}")
    print(f"Use SNAC = {self.use_snac}")

    self.use_sdp = use_sdp

    self.enc_p = TextEncoder(n_vocab,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout)
    self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels, use_emotion=self.use_emotion)
    self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels, use_emotion = self.use_emotion)
    
    if(self.use_snac):
      self.flow = ResidualCouplingBlockSNAC(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)
    else:
      self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels, use_emotion=self.use_emotion)

    if use_sdp:
      self.dp = StochasticDurationPredictor(hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels, use_emotion=self.use_emotion)
    else:
      self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels, use_emotion=self.use_emotion)

    if n_speakers > 1:
      print(f"Using speaker embedding type = {self.speaker_embedding_type}")
      if(self.speaker_embedding_type == 'lookup'):
        self.emb_g = nn.Embedding(n_speakers, gin_channels)
      elif(self.speaker_embedding_type == 'hasp'):
        self.emb_g = speaker_encoder.MainModel(
              nOut=512,
              encoder_type="ASP",
              n_mels=64,
              log_input=True
          )
      elif(self.speaker_embedding_type == 're'):
        self.emb_g = speaker_encoder.ReferenceEncoder(80, gin_channels*2)
      else:
        raise ValueError("Unknown speaker embedding type {}".format(self.speaker_embedding_type))

    if self.encoder_type == "ce":
      self.speaker_classifier_head = nn.Linear(gin_channels, n_speakers)

    self.emb_e = None
    if n_emotions > 1:
      print(f"Using emotion embedding type = {self.emotion_embedding_type}")
      if(self.filter_mel_bin > 0):
        print(f"Using filter mel bin with value = {self.filter_mel_bin}. Changing the style encoder input dim.")
        mel_dim = self.filter_mel_bin
      else:
        mel_dim = 80

      if(self.emotion_embedding_type == 'lookup'):
        self.emb_e = nn.Embedding(n_emotions, gin_channels)
      elif(self.emotion_embedding_type == 're'):
        self.emb_e = style_encoder.ReferenceEncoder(mel_dim, gin_channels*2)
      else:
        raise ValueError("Unknown emotion embedding type {}".format(self.emotion_embedding_type))

    if self.encoder_type == "ce":
      self.emotion_classifier_head = nn.Linear(gin_channels, n_emotions)

    if self.grl_type is not None:
      print(f"Using GRL = {self.grl_type}")
      if self.grl_type == 'encoder':
        self.emo_grl_on_spk_emb = modules_grl.LinearEncoder(gin_channels, gin_channels)
        self.spk_grl_on_emo_emb = modules_grl.LinearEncoder(gin_channels, gin_channels)
        self.emo_grl_on_zp = modules_grl.ConvEncoder(hidden_channels, gin_channels)
        self.spk_grl_on_zp = modules_grl.ConvEncoder(hidden_channels, gin_channels)
      elif self.grl_type == 'classifier':
        self.emo_grl_on_spk_emb = modules_grl.LinearEncoderClassifier(gin_channels, n_emotions)
        self.spk_grl_on_emo_emb = modules_grl.LinearEncoderClassifier(gin_channels, n_speakers)
        self.emo_grl_on_zp = modules_grl.ConvEncoderClassifier(hidden_channels, n_emotions)
        self.spk_grl_on_zp = modules_grl.ConvEncoderClassifier(hidden_channels, n_speakers)
      else:
        raise ValueError("Unknown grl_type {}".format(self.grl_type))

  def forward(self, x, x_lengths, y, y_lengths, sid=None, eid=None, melspec=None, perturbed_melspec=None, y_seg=None, y_seg_pertubed=None):
    zp_pred_spk = zp_pred_emo = None
    spk_pred_from_emo_emb = emo_pred_from_spk_emb = None

    # print(melspec.shape, perturbed_melspec.shape)

    x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
    
    if self.n_speakers > 0:
      if(self.speaker_embedding_type ==  "lookup"):
        g = self.emb_g(sid) 
      elif(self.speaker_embedding_type == "hsap"):
        g = self.emb_g(y_seg)
      elif(self.speaker_embedding_type == "re"):
        g = self.emb_g(melspec)
      else:
        raise ValueError("Unknown speaker embedding type {}".format(self.speaker_embedding_type))

      if self.encoder_type == "ce":
        g_logits = self.speaker_classifier_head(g)
      else:
        g_logits = 0

      if self.use_snac:
        g = F.normalize(g)
      g = g.unsqueeze(-1) # [b, h, 1]
    else:
      g = None

    if self.n_emotions > 0:
      if(self.filter_mel_bin > 0):
        perturbed_melspec = perturbed_melspec[:, :self.filter_mel_bin, :]
        # print(perturbed_melspec.shape)
      if(self.emotion_embedding_type == "lookup"):
        e = self.emb_e(eid)
      elif(self.emotion_embedding_type == "metastyle"):
        e = self.emb_e(perturbed_melspec)
      elif(self.emotion_embedding_type == "stylespeech"):
        e = self.emb_e(perturbed_melspec)
      elif(self.emotion_embedding_type == "re"):
        e = self.emb_e(perturbed_melspec)
      else:
        raise ValueError("Unknown emotion embedding type {}".format(self.emotion_embedding_type)) 

      if self.encoder_type == "ce":
        e_logits = self.emotion_classifier_head(e)
      else:
        e_logits = 0

      if self.use_snac:
        e = F.normalize(e)

      e = e.unsqueeze(-1) # [b, h, 1]
    else:
      e = None



    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g, e=e) 
    z_p = self.flow(z, y_mask, g=g, e=e)

    # z_p should be related only to content information not speaker or emotion, so applying grl layers
    if self.grl_type is not None:
      if self.n_speakers > 0:
        zp_pred_spk = self.spk_grl_on_zp(z_p)
        spk_pred_from_emo_emb = self.spk_grl_on_emo_emb(e.squeeze(-1))
      if self.n_emotions > 0:
        zp_pred_emo = self.emo_grl_on_zp(z_p)
        emo_pred_from_spk_emb = self.emo_grl_on_spk_emb(g.squeeze(-1))



    with torch.no_grad():
      # negative cross-entropy
      s_p_sq_r = torch.exp(-2 * logs_p) # [b, d, t]
      neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True) # [b, 1, t_s]
      neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
      neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r)) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
      neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True) # [b, 1, t_s]
      neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

      attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
      attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).unsqueeze(1).detach()

    w = attn.sum(2)
    if self.use_sdp:
      l_length = self.dp(x, x_mask, w, g=g, e=e)
      l_length = l_length / torch.sum(x_mask)
    else:
      logw_ = torch.log(w + 1e-6) * x_mask
      logw = self.dp(x, x_mask, g=g, e=e)
      l_length = torch.sum((logw - logw_)**2, [1,2]) / torch.sum(x_mask) # for averaging 

    # expand prior
    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

    z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
    o = self.dec(z_slice, g=g, e=e)
    return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q), g.squeeze(-1), e.squeeze(-1), zp_pred_spk, zp_pred_emo, spk_pred_from_emo_emb, emo_pred_from_spk_emb, g_logits, e_logits

  def infer(self, x, x_lengths, sid=None, eid=None, melspec=None, perturbed_melspec=None, y_seg=None, y_seg_pertubed=None, noise_scale=1, length_scale=1, noise_scale_w=1., max_len=None):
    x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)
    if self.n_speakers > 0:
      if(self.speaker_embedding_type ==  "lookup"):
        g = self.emb_g(sid)
      elif(self.speaker_embedding_type == "hsap"):
        g = self.emb_g(y_seg)
      elif(self.speaker_embedding_type == "re"):
        g = self.emb_g(melspec)
      else:
        raise ValueError("Unknown speaker embedding type {}".format(self.speaker_embedding_type))

      if self.use_snac:
        g = F.normalize(g).unsqueeze(-1)
      else:
        g = g.unsqueeze(-1)
    else:
      g = None

    if self.n_emotions > 0:
      if(self.filter_mel_bin > 0):
        perturbed_melspec = perturbed_melspec[:, :self.filter_mel_bin, :]
      if(self.emotion_embedding_type == "lookup"):
        e = self.emb_e(eid) # [b, h, 1]
      elif(self.emotion_embedding_type == "re"):
        e = self.emb_e(perturbed_melspec) 
      else:
        raise ValueError("Unknown emotion embedding type {}".format(self.emotion_embedding_type)) 
      if self.use_snac:
        e = F.normalize(e).unsqueeze(-1)
      else:
        e = e.unsqueeze(-1)
    else:
      e = None

    if self.use_sdp:
      logw = self.dp(x, x_mask, g=g, e=e, reverse=True, noise_scale=noise_scale_w)
    else:
      logw = self.dp(x, x_mask, g=g, e=e)
    w = torch.exp(logw) * x_mask * length_scale
    w_ceil = torch.ceil(w)
    y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
    y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
    attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
    attn = commons.generate_path(w_ceil, attn_mask)

    m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
    logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']

    z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
    z = self.flow(z_p, y_mask, g=g, e=e, reverse=True)
    o = self.dec((z * y_mask)[:,:,:max_len], g=g, e=e)
    return o, attn, y_mask, (z, z_p, m_p, logs_p)

  def voice_conversion(self, y, y_lengths, sid_src, sid_tgt, eid_src=None, eid_tgt=None):
    assert self.n_speakers > 0, "n_speakers have to be larger than 0."
    g_src = self.emb_g(sid_src)
    g_tgt = self.emb_g(sid_tgt)

    if self.use_snac:
      g_src = F.normalize(g_src).unsqueeze(-1)
      g_tgt = F.normalize(g_tgt).unsqueeze(-1)
    else:
      g_src = g_src.unsqueeze(-1)
      g_tgt = g_tgt.unsqueeze(-1)

    e_src = None
    e_tgt = None
    if(eid_src is not None):
      assert self.emb_e is not None, "emb_e must be not None if passing emotion id"
      e_src = self.emb_e(eid_src)
      e_tgt = self.emb_e(eid_tgt)

    if self.use_snac:
      e_src = F.normalize(e_src).unsqueeze(-1)
      e_tgt = F.normalize(e_tgt).unsqueeze(-1)
    else:
      e_src = e_src.unsqueeze(-1)
      e_tgt = e_tgt.unsqueeze(-1)

    z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src, e=e_src)
    z_p = self.flow(z, y_mask, g=g_src, e=e_src)
    z_hat = self.flow(z_p, y_mask, g=g_tgt, e=e_tgt, reverse=True)
    o_hat = self.dec(z_hat * y_mask, g=g_tgt, e=e_tgt)
    return o_hat, y_mask, (z, z_p, z_hat)

  def get_speaker_embedding(
      self,
      sid=None,
      melspec=None,
      y_seg=None
  ):
      """
      Returns speaker embedding g with shape [B, gin_channels, 1]
      """
      assert self.n_speakers > 0, "n_speakers must be > 0."

      if self.speaker_embedding_type == "lookup":
          assert sid is not None, "sid must be provided for lookup speaker embedding"
          g = self.emb_g(sid)

      elif self.speaker_embedding_type == "hsap":
          assert y_seg is not None, "y_seg must be provided for HSAP speaker embedding"
          g = self.emb_g(y_seg)

      elif self.speaker_embedding_type == "re":
          assert melspec is not None, "melspec must be provided for RE speaker embedding"
          g = self.emb_g(melspec)

      else:
          raise ValueError(f"Unknown speaker embedding type {self.speaker_embedding_type}")

      return g  # [B, gin_channels]


  def get_emotion_embedding(
      self,
      eid=None,
      melspec=None,
      y_seg=None
  ):
      """
      Returns emotion embedding e with shape [B, gin_channels, 1]
      """
      assert self.n_emotions > 0, "n_emotions must be > 0."
      assert self.emb_e is not None, "emb_e is None but n_emotions > 0"

      if(self.filter_mel_bin > 0):
        melspec = melspec[:, :self.filter_mel_bin, :]

      if self.emotion_embedding_type == "lookup":
          assert eid is not None, "eid must be provided for lookup emotion embedding"
          e = self.emb_e(eid)

      elif self.emotion_embedding_type == "re":
          assert melspec is not None, \
              "melspec must be provided for RE emotion embedding"
          e = self.emb_e(melspec)

      else:
          raise ValueError(f"Unknown emotion embedding type {self.emotion_embedding_type}")

      return e  # [B, gin_channels]

  def infer_from_embeddings(
      self,
      x,
      x_lengths,
      g,   # [B, gin_channels, 1]
      e,   # [B, gin_channels, 1] or None
      noise_scale=1.0,
      length_scale=1.0,
      noise_scale_w=1.0,
      max_len=None
  ):
      # Probably unnecessary to squeeze unsqueeze, but lets keep it just to remind the shapes
      if self.use_snac:
        e = F.normalize(e.squeeze(-1)).unsqueeze(-1)
        g = F.normalize(g.squeeze(-1)).unsqueeze(-1)

      x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths)

      if self.use_sdp:
          logw = self.dp(x, x_mask, g=g, e=e, reverse=True, noise_scale=noise_scale_w)
      else:
          logw = self.dp(x, x_mask, g=g, e=e)

      w = torch.exp(logw) * x_mask * length_scale
      w_ceil = torch.ceil(w)

      y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
      y_mask = torch.unsqueeze(
          commons.sequence_mask(y_lengths, None),
          1
      ).to(x_mask.dtype)

      attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
      attn = commons.generate_path(w_ceil, attn_mask)

      m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
      logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

      z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
      z = self.flow(z_p, y_mask, g=g, e=e, reverse=True)

      o = self.dec((z * y_mask)[:, :, :max_len], g=g, e=e)

      return o, attn, y_mask, (z, z_p, m_p, logs_p)
  
  def voice_conversion_from_embeddings(
        self, 
        y, 
        y_lengths, 
        g_src, 
        g_tgt, 
        e_src=None, 
        e_tgt=None
    ):
      """
      Performs voice conversion using pre-computed embedding tensors.
      
      Args:
          y: Source audio/spectrogram [B, spec_channels, T]
          y_lengths: Lengths of source audio [B]
          g_src: Source speaker embedding [B, gin_channels, 1]
          g_tgt: Target speaker embedding [B, gin_channels, 1]
          e_src: Source emotion embedding [B, gin_channels, 1] (optional)
          e_tgt: Target emotion embedding [B, gin_channels, 1] (optional)
      """
      assert self.n_speakers > 0, "n_speakers must be larger than 0 for conversion."
      
      if self.use_snac:
        e_src = F.normalize(e_src.squeeze(-1)).unsqueeze(-1)
        e_tgt = F.normalize(e_tgt.squeeze(-1)).unsqueeze(-1)
        g_src = F.normalize(g_src.squeeze(-1)).unsqueeze(-1)
        g_tgt = F.normalize(g_tgt.squeeze(-1)).unsqueeze(-1)

      # 1. Encode source audio into the latent space z using source identities
      # enc_q is the Posterior Encoder
      z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src, e=e_src)

      # 2. Project z to the normalized prior space z_p via the Flow (Forward)
      # This step 'removes' the source speaker/emotion characteristics
      z_p = self.flow(z, y_mask, g=g_src, e=e_src)

      # 3. Project z_p back to latent space z_hat via the Flow (Reverse)
      # This step 'adds' the target speaker/emotion characteristics
      z_hat = self.flow(z_p, y_mask, g=g_tgt, e=e_tgt, reverse=True)

      # 4. Decode the new latent representation into audio
      o_hat = self.dec(z_hat * y_mask, g=g_tgt, e=e_tgt)

      return o_hat, y_mask, (z, z_p, z_hat)

  def voice_conversion_from_embeddings_flow_embs(
        self, 
        y, 
        y_lengths, 
        g_src, 
        g_tgt, 
        e_src=None, 
        e_tgt=None
    ):
        """
        Performs voice conversion capturing only the 8 actual coupling states.
        (4 Forward + 4 Reverse)
        """
        assert self.n_speakers > 0, "n_speakers must be > 0 for conversion."
        
        if self.use_snac:
            import torch.nn.functional as F
            if e_src is not None: e_src = F.normalize(e_src.squeeze(-1), dim=1).unsqueeze(-1)
            if e_tgt is not None: e_tgt = F.normalize(e_tgt.squeeze(-1), dim=1).unsqueeze(-1)
            g_src = F.normalize(g_src.squeeze(-1), dim=1).unsqueeze(-1)
            g_tgt = F.normalize(g_tgt.squeeze(-1), dim=1).unsqueeze(-1)

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src, e=e_src)
        if isinstance(y_mask, (list, tuple)):
            y_mask = y_mask[0]

        intermediate_embs = []

        # 1. Forward Flow (Source -> Prior)
        x = z
        for flow in self.flow.flows:
            if isinstance(flow, modules.Flip):
                res = flow(x, x_mask=y_mask, reverse=False)
                x = res[0] if isinstance(res, (list, tuple)) else res
            else:
                # This is a ResidualCouplingLayer
                res = flow(x, y_mask, g=g_src, e=e_src, reverse=False)
                x = res[0] if isinstance(res, (list, tuple)) else res
                # Capture state after the transformation
                intermediate_embs.append(x.mean(dim=-1))
            
        z_p = x 

        # 2. Reverse Flow (Prior -> Target)
        # Note: In reverse, we go through the list backward
        for flow in reversed(self.flow.flows):
            if isinstance(flow, modules.Flip):
                res = flow(x, x_mask=y_mask, reverse=True)
                x = res[0] if isinstance(res, (list, tuple)) else res
            else:
                # This is a ResidualCouplingLayer
                res = flow(x, y_mask, g=g_tgt, e=e_tgt, reverse=True)
                x = res[0] if isinstance(res, (list, tuple)) else res
                # Capture state after the transformation
                intermediate_embs.append(x.mean(dim=-1))

        z_hat = x
        o_hat = self.dec(z_hat * y_mask, g=g_tgt, e=e_tgt)

        return o_hat, y_mask, (z, z_p, z_hat), intermediate_embs
