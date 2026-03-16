"""Voice encoder wrappers: WavLM, ECAPA-TDNN, HuBERT, wav2vec2."""

import torch
import torch.nn as nn


class WavLMEncoder(nn.Module):
    """Frozen WavLM-Large encoder. Outputs 1024-d embeddings via mean pooling."""

    def __init__(self, model_name: str = "microsoft/wavlm-large"):
        super().__init__()
        from transformers import WavLMModel

        self.model = WavLMModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        self._output_dim = self.model.config.hidden_size

    def _mean_pool(self, hidden_states, attention_mask):
        if attention_mask is not None:
            seq_len = hidden_states.size(1)
            reduced_mask = attention_mask[:, :: attention_mask.size(1) // seq_len][
                :, :seq_len
            ].to(hidden_states.dtype)
            lengths = reduced_mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (hidden_states * reduced_mask.unsqueeze(-1)).sum(dim=1) / lengths
        else:
            pooled = hidden_states.mean(dim=1)
        return pooled

    @torch.no_grad()
    def forward(
        self, waveforms: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        outputs = self.model(input_values=waveforms, attention_mask=attention_mask)
        return self._mean_pool(outputs.last_hidden_state, attention_mask)

    @property
    def output_dim(self) -> int:
        return self._output_dim


class ECAPATDNNEncoder(nn.Module):
    """ECAPA-TDNN via torchaudio pipeline. Outputs 192-d speaker embeddings."""

    def __init__(self):
        super().__init__()
        import torchaudio

        bundle = torchaudio.pipelines.WAVLM_BASE
        # Use torchaudio's built-in speaker verification model
        # Falls back to a simpler approach: load via torch hub
        try:
            from torchaudio.pipelines import ECAPA_TDNN_VOXCELEB
            self._pipeline = ECAPA_TDNN_VOXCELEB
            self.model = self._pipeline.get_model()
            self._output_dim = 192
        except (ImportError, AttributeError):
            # If torchaudio doesn't have ECAPA pipeline, use speechbrain directly
            self._load_speechbrain()

    def _load_speechbrain(self):
        """Fallback: load ECAPA-TDNN via speechbrain with compatibility patches."""
        import torchaudio
        if not hasattr(torchaudio, 'list_audio_backends'):
            torchaudio.list_audio_backends = lambda: ["soundfile"]

        import huggingface_hub
        _orig_download = huggingface_hub.hf_hub_download
        def _patched_download(*args, **kwargs):
            kwargs.pop('use_auth_token', None)
            return _orig_download(*args, **kwargs)
        huggingface_hub.hf_hub_download = _patched_download

        from speechbrain.inference.speaker import EncoderClassifier
        # Try multiple model source names (repo structure has changed over time)
        for source in [
            "speechbrain/spkrec-ecapa-voxceleb",
            "speechbrain/spkrec-ecapa-voxceleb12",
            "speechbrain/spkrec-ecapa-voxceleb-mel-spec",
        ]:
            try:
                self._sb_model = EncoderClassifier.from_hparams(
                    source=source, run_opts={"device": "cpu"},
                )
                break
            except Exception:
                continue
        else:
            raise RuntimeError("Could not load any ECAPA-TDNN model from SpeechBrain")
        self.model = None  # sentinel: use _sb_model path
        self._output_dim = 192

    @torch.no_grad()
    def forward(
        self, waveforms: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.model is not None:
            # torchaudio pipeline path
            return self.model(waveforms)
        else:
            # SpeechBrain path
            if waveforms.dim() == 1:
                waveforms = waveforms.unsqueeze(0)
            if attention_mask is not None:
                wav_lens = attention_mask.sum(dim=1).float() / waveforms.shape[1]
            else:
                wav_lens = torch.ones(waveforms.shape[0], device=waveforms.device)
            embeddings = self._sb_model.encode_batch(waveforms, wav_lens)
            return embeddings.squeeze(1)

    @property
    def output_dim(self) -> int:
        return self._output_dim


class HuBERTEncoder(nn.Module):
    """Frozen HuBERT-Large encoder. Outputs 1024-d embeddings via mean pooling."""

    def __init__(self, model_name: str = "facebook/hubert-large-ls960-ft"):
        super().__init__()
        from transformers import HubertModel

        self.model = HubertModel.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        self._output_dim = self.model.config.hidden_size

    def _mean_pool(self, hidden_states, attention_mask):
        if attention_mask is not None:
            seq_len = hidden_states.size(1)
            reduced_mask = attention_mask[:, :: attention_mask.size(1) // seq_len][
                :, :seq_len
            ].to(hidden_states.dtype)
            lengths = reduced_mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (hidden_states * reduced_mask.unsqueeze(-1)).sum(dim=1) / lengths
        else:
            pooled = hidden_states.mean(dim=1)
        return pooled

    @torch.no_grad()
    def forward(
        self, waveforms: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        outputs = self.model(input_values=waveforms, attention_mask=attention_mask)
        return self._mean_pool(outputs.last_hidden_state, attention_mask)

    @property
    def output_dim(self) -> int:
        return self._output_dim


class Wav2Vec2Encoder(nn.Module):
    """Frozen wav2vec2-Large encoder. Outputs 1024-d embeddings via mean pooling."""

    def __init__(self, model_name: str = "facebook/wav2vec2-large-960h"):
        super().__init__()
        from transformers import Wav2Vec2Model

        self.model = Wav2Vec2Model.from_pretrained(model_name)
        for param in self.model.parameters():
            param.requires_grad = False
        self._output_dim = self.model.config.hidden_size

    def _mean_pool(self, hidden_states, attention_mask):
        if attention_mask is not None:
            seq_len = hidden_states.size(1)
            reduced_mask = attention_mask[:, :: attention_mask.size(1) // seq_len][
                :, :seq_len
            ].to(hidden_states.dtype)
            lengths = reduced_mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = (hidden_states * reduced_mask.unsqueeze(-1)).sum(dim=1) / lengths
        else:
            pooled = hidden_states.mean(dim=1)
        return pooled

    @torch.no_grad()
    def forward(
        self, waveforms: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        outputs = self.model(input_values=waveforms, attention_mask=attention_mask)
        return self._mean_pool(outputs.last_hidden_state, attention_mask)

    @property
    def output_dim(self) -> int:
        return self._output_dim
