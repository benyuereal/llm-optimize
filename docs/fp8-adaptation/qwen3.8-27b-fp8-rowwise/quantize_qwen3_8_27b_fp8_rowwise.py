"""
Quantize Qwen3.8-27B (BF16, multimodal Qwen3_5ForConditionalGeneration) to
FP8 rowwise (per-channel weights) using llm-compressor's model_free_ptq pathway.

- Scheme: FP8_DYNAMIC with weights observer set to memoryless_minmax
    weights  -> strategy='channel'  (rowwise, 1D scale [out_features])  <- rocBLAS rowwise
    actives  -> per-token dynamic (computed at inference, not stored)
  The memoryless_minmax observer computes the per-channel weight scale directly
  from the weight tensor (data-free), which is what model_free_ptq needs.
- Source : /models/Qwen3.8-27B  (BF16)
- Output : /models/Qwen3.8-27B-FP8-ROWWISE
- Device : cuda:5 (free GPU card)

NOTE on ignore list (differs from the pure-transformer Qwen3-32B recipe):
  * match_name() uses EXACT match or "re:" regex — NOT prefix matching. So the
    vision tower must be ignored with a regex, not the bare string "model.visual".
  * This hybrid model has linear-attention (mamba-like) layers whose
    `linear_attn.conv1d.weight` is a 3D depthwise conv [10240,1,4].
    validate_weight_for_quantization() raises on any non-2D weight, so conv1d
    must be ignored (kept BF16). The sibling Qwen3-32B has no such conv.
  * model.visual.patch_embed.proj is also 3D — covered by the visual regex.
  * The MTP (multi-token-prediction) module has two 1D RMSNorm weights
    (mtp.pre_fc_norm_embedding / mtp.pre_fc_norm_hidden) whose module names do
    NOT end in "norm", so they are not auto-skipped and would crash the 2D
    check — ignored (kept BF16). MTP's 2D linears are quantized normally.
  * Other norms (1D) and A_log/dt_bias (1D) are auto-skipped (norm suffix / non-weight).
"""

from compressed_tensors.quantization import preset_name_to_scheme
from llmcompressor import model_free_ptq

MODEL_ID = "/models/Qwen3.8-27B"
SAVE_DIR = "/models/Qwen3.8-27B-FP8-ROWWISE"

# Start from the FP8_DYNAMIC preset (rowwise/channel weights + dynamic per-token act)
# and give the weights a memoryless observer so model_free_ptq can compute the
# per-channel scale from the weight itself (data-free).
scheme = preset_name_to_scheme("FP8_DYNAMIC", [])
scheme = scheme.model_copy(
    update={
        "weights": scheme.weights.model_copy(
            update={"observer": "memoryless_minmax"}
        )
    }
)

model_free_ptq(
    model_stub=MODEL_ID,
    save_directory=SAVE_DIR,
    scheme=scheme,
    ignore=[
        "lm_head",
        "model.language_model.embed_tokens",
        "re:model\\.visual\\.",          # vision tower: keep BF16 (regex, not prefix)
        "re:model\\.language_model\\.layers\\.\\d+\\.linear_attn\\.conv1d",  # 3D conv
        "re:mtp\\.pre_fc_norm_(embedding|hidden)",  # 1D MTP norms (not auto-skipped)
    ],
    max_workers=8,
    device="cuda:5",
)
