#!/usr/bin/env bash
set -euo pipefail

# Emotion-enabled live launcher: identical to the AH AudioPace winner
# (IMTalker/start_winner_live.sh, VARIANT=AH) but starts the server through
# emotion_finetune/launch_emotion_live.py, which loads the stock AH server and
# applies the trained emotion adapter to the FM generator.
#
# Emotion configuration (environment):
#   EMOTION_ADAPTER_PATH   trained emotion adapter checkpoint (required for emotion)
#   AVATAR_EMOTION         neutral | happy | sad | angry | surprise | fear (default: neutral)
#   EMOTION_SCALE          emotion intensity multiplier (default: 1.0)
#   EMOTION_LORA_RANK      adapter training rank (default: 16)
#   EMOTION_LORA_ALPHA     adapter training alpha (default: 32)
#
# With EMOTION_ADAPTER_PATH unset the server is bit-identical to the stock AH server.

IMTALKER_DIR="${IMTALKER_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/IMTalker" && pwd)}"
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$IMTALKER_DIR")}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"

pick_existing() {
  local candidate
  for candidate in "$@"; do
    if [[ -e "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  printf 'Missing required asset. Checked:\n' >&2
  printf '  %s\n' "$@" >&2
  return 1
}

PORT="${PORT:-8998}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
A_CFG_SCALE="${A_CFG_SCALE:-1.13}"
NFE="${NFE:-5}"
VOICE_PROMPT="${VOICE_PROMPT:-VARM3.pt}"

LAUNCHER="$PROJECT_ROOT/emotion_finetune/launch_emotion_live.py"

PERSONAPLEX_DIR="${PERSONAPLEX_DIR:-$(pick_existing \
  /workspace/personaplex_bnb4 \
  "$PROJECT_ROOT/checkpoints/personaplex_bnb4")}"
ADAPTER_PATH="${ADAPTER_PATH:-$(pick_existing \
  /workspace/hf_assets/personaplex_lookahead_rms_adapter/checkpoints/personaplex_lookahead096_future048_rms50_adapter.pt \
  "$PROJECT_ROOT/checkpoints/personaplex_lookahead_rms_adapter/checkpoints/personaplex_lookahead096_future048_rms50_adapter.pt")}"
SILENCE_HELIUM_PATH="${SILENCE_HELIUM_PATH:-$(pick_existing \
  /workspace/hf_assets/personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt \
  /workspace/personaplex_frontend_adapter_dataset/stats/silence_helium_mean.pt \
  "$PROJECT_ROOT/checkpoints/personaplex_lookahead_rms_adapter/stats/silence_helium_mean.pt")}"
DISABLE_LORA="${DISABLE_LORA:-0}"
if [[ "$DISABLE_LORA" == "1" ]]; then
  LORA_GENERATOR_PATH=""
  LORA_ARGS=()
else
  LORA_GENERATOR_PATH="${LORA_GENERATOR_PATH:-$(pick_existing \
    "$PROJECT_ROOT/checkpoints/live_winner/lora/ditto_blink_lora_withaudio_r64_096_continue_2h_last.ckpt" \
    "$IMTALKER_DIR/checkpoints/ditto_blink_lora_withaudio_r64_1h_last.ckpt" \
    /workspace/hf_assets/lora/ditto_blink_lora_withaudio_r64_1h_last.ckpt \
    "$PROJECT_ROOT/checkpoints/lora/ditto_blink_lora_withaudio_r64_1h_last.ckpt")}"
  LORA_ARGS=(
    --lora_generator_path "$LORA_GENERATOR_PATH"
    --lora_rank 64
    --lora_alpha 128
    --lora_dropout 0.05
  )
fi

# --- Emotion adapter ----------------------------------------------------------
EMOTION_ADAPTER_PATH="${EMOTION_ADAPTER_PATH:-}"
AVATAR_EMOTION="${AVATAR_EMOTION:-neutral}"
EMOTION_SCALE="${EMOTION_SCALE:-1.0}"
EMOTION_LORA_RANK="${EMOTION_LORA_RANK:-16}"
EMOTION_LORA_ALPHA="${EMOTION_LORA_ALPHA:-32}"
if [[ -n "$EMOTION_ADAPTER_PATH" && ! -e "$EMOTION_ADAPTER_PATH" ]]; then
  echo "Missing emotion adapter: $EMOTION_ADAPTER_PATH" >&2
  exit 1
fi
export EMOTION_ADAPTER_PATH AVATAR_EMOTION EMOTION_SCALE EMOTION_LORA_RANK EMOTION_LORA_ALPHA
# STT + query routing + web search -- opt-in, additive. ENABLE_SEARCH=0
# (default) reproduces the plain conversational launch command with zero new
# flags appended. ENABLE_RAG is still honored as a legacy alias.
ENABLE_SEARCH="${ENABLE_SEARCH:-${ENABLE_RAG:-0}}"
if [[ "$ENABLE_SEARCH" == "1" ]]; then
  REF_LORA_DIR="${REF_LORA_DIR:-${RAG_CHECKPOINT_DIR:-$(pick_existing \
    "$PROJECT_ROOT/checkpoints/rag_lora" \
    /workspace/hf_assets/rag_lora)}}"
  STT_PKG_DIR="${STT_PKG_DIR:-$PROJECT_ROOT/checkpoints/stt}"
  CONVERSATION_LOG_DIR="${CONVERSATION_LOG_DIR:-$PROJECT_ROOT/conversation_logs}"
  THINKING_SOUND_PATH="${THINKING_SOUND_PATH:-$PROJECT_ROOT/personaplex/ai-thinking-sound.wav}"
  SEARCH_ARGS=(
    --conversation_log_dir "$CONVERSATION_LOG_DIR"
    --ref_lora_dir "$REF_LORA_DIR"
    --stt_hf_repo "${STT_HF_REPO:-kyutai/stt-1b-en_fr-candle}"
    --stt_pkg_dir "$STT_PKG_DIR"
    --vad_threshold "${VAD_THRESHOLD:-0.5}"
    --stt_reject_foreign_script "${STT_REJECT_FOREIGN_SCRIPT:-1}"
    --stt_max_non_latin_ratio "${STT_MAX_NON_LATIN_RATIO:-0.15}"
    --stt_require_english "${STT_REQUIRE_ENGLISH:-1}"
    --suppress_text_during_search "${SUPPRESS_TEXT_DURING_SEARCH:-1}"
    --prompt_settle_sec "${PROMPT_SETTLE_SEC:-0.0}"
    --max_input_buffer_sec "${MAX_INPUT_BUFFER_SEC:-2.0}"
    --compressor_model "${COMPRESSOR_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
    --compressor_device "${COMPRESSOR_DEVICE:-cuda}"
    --router_threshold "${ROUTER_THRESHOLD:-0.40}"
    --router_rules "${ROUTER_RULES:-1}"
    --thinking_sound_path "$THINKING_SOUND_PATH"
    --search_max_filler_sec "${SEARCH_MAX_FILLER_SEC:-${RAG_MAX_FILLER_SEC:-6.0}}"
    --web_search_min_score "${WEB_SEARCH_MIN_SCORE:-0.15}"
  )
  if [[ -n "${WEB_SEARCH_API_KEY:-}" ]]; then
    WEB_SEARCH_ENABLED="${WEB_SEARCH_ENABLED:-1}"
  else
    WEB_SEARCH_ENABLED="${WEB_SEARCH_ENABLED:-0}"
  fi
  if [[ "$WEB_SEARCH_ENABLED" == "1" ]]; then
    SEARCH_ARGS+=(
      --web_search_enabled
      --web_search_api_key "${WEB_SEARCH_API_KEY:?set WEB_SEARCH_API_KEY when WEB_SEARCH_ENABLED=1}"
      --web_search_provider "${WEB_SEARCH_PROVIDER:-tavily}"
      --web_search_max_results "${WEB_SEARCH_MAX_RESULTS:-3}"
      --web_search_timeout "${WEB_SEARCH_TIMEOUT:-3.0}"
    )
  else
    echo "[warn] ENABLE_SEARCH=1 but no WEB_SEARCH_API_KEY -- the router will still run," >&2
    echo "       but questions needing live data will fall back to the model's own knowledge." >&2
  fi
  echo "Search enabled: ref_lora=$REF_LORA_DIR stt_pkg=$STT_PKG_DIR web_search=$WEB_SEARCH_ENABLED conversation_log_dir=$CONVERSATION_LOG_DIR"
else
  SEARCH_ARGS=()
fi

GENERATOR_PATH="${GENERATOR_PATH:-$IMTALKER_DIR/checkpoints/generator.ckpt}"
RENDERER_PATH="${RENDERER_PATH:-$IMTALKER_DIR/checkpoints/renderer.ckpt}"
WAV2VEC_MODEL_PATH="${WAV2VEC_MODEL_PATH:-$IMTALKER_DIR/checkpoints/wav2vec2-base-960h}"
REF_PATH="${REF_PATH:-$IMTALKER_DIR/assets/3robert.jpeg}"
PROMPT_FILE="${PROMPT_FILE:-$IMTALKER_DIR/prompts/RB_Robert_System_Prompt_full.txt}"
HTML_PATH="${HTML_PATH:-$IMTALKER_DIR/static/index_v3_binary_fullscreen_aj_nodrop.html}"

for required in \
  "$LAUNCHER" "$GENERATOR_PATH" "$RENDERER_PATH" \
  "$ADAPTER_PATH" "$SILENCE_HELIUM_PATH" "$REF_PATH" "$PROMPT_FILE" "$HTML_PATH" \
  "$PERSONAPLEX_DIR/model_bnb_4bit.pt" \
  "$PERSONAPLEX_DIR/tokenizer-e351c8d8-checkpoint125.safetensors" \
  "$PERSONAPLEX_DIR/tokenizer_spm_32k_3.model"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
done

if [[ "$ENABLE_SEARCH" == "1" ]]; then
  for required in \
    "$REF_LORA_DIR/lora/adapter_config.json"; do
    [[ -e "$required" ]] || { echo "Missing required search path: $required (re-run scripts/download_live_assets.sh, or set ENABLE_SEARCH=0)" >&2; exit 1; }
  done
fi

if [[ -z "${VOICE_PROMPT_DIR:-}" ]]; then
  for candidate in \
    "$PERSONAPLEX_DIR/voices" \
    /workspace/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/*/voices \
    /root/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/*/voices \
    "$HOME"/.cache/huggingface/hub/models--nvidia--personaplex-7b-v1/snapshots/*/voices; do
    if [[ -f "$candidate/$VOICE_PROMPT" ]]; then
      VOICE_PROMPT_DIR="$candidate"
      break
    fi
  done
fi
[[ -f "${VOICE_PROMPT_DIR:-}/$VOICE_PROMPT" ]] || {
  echo "Cannot find $VOICE_PROMPT. Set VOICE_PROMPT_DIR explicitly." >&2
  exit 1
}

source "$VENV_DIR/bin/activate"
cd "$IMTALKER_DIR"

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="$IMTALKER_DIR:$PERSONAPLEX_DIR/moshi:$PERSONAPLEX_DIR:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
export IMTALKER_PROMPT_STATE_CACHE="${IMTALKER_PROMPT_STATE_CACHE:-1}"

ROBERT_PROMPT="$(tr '\n' ' ' < "$PROMPT_FILE")"
DUMP_DIR="${DUMP_DIR:-$IMTALKER_DIR/live_dumps_emotion_${PORT}_varm3}"

echo "Starting EMOTION AH on port $PORT, physical GPU $CUDA_VISIBLE_DEVICES"
echo "Voice: $VOICE_PROMPT | CFG: $A_CFG_SCALE | NFE: $NFE"
if [[ -n "$EMOTION_ADAPTER_PATH" ]]; then
  echo "Emotion: adapter=$EMOTION_ADAPTER_PATH avatar_emotion=$AVATAR_EMOTION scale=$EMOTION_SCALE"
else
  echo "Emotion: disabled (stock AH behavior)"
fi

exec python -u "$LAUNCHER" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --html_path "$HTML_PATH" \
  --generator_path "$GENERATOR_PATH" \
  --renderer_path "$RENDERER_PATH" \
  "${LORA_ARGS[@]}" \
  --adapter_path "$ADAPTER_PATH" \
  --adapter_type unitalk_last_layer \
  --adapter_num_layers 12 \
  --adapter_dropout 0.0 \
  --adapter_window_mode lookahead \
  --adapter_future_steps 6 \
  --ref_path "$REF_PATH" \
  --wav2vec_model_path "$WAV2VEC_MODEL_PATH" \
  --moshi_root "$PERSONAPLEX_DIR" \
  --mimi_hf_repo nvidia/personaplex-7b-v1 \
  --moshi_weight "$PERSONAPLEX_DIR/model_bnb_4bit.pt" \
  --mimi_weight "$PERSONAPLEX_DIR/tokenizer-e351c8d8-checkpoint125.safetensors" \
  --tokenizer "$PERSONAPLEX_DIR/tokenizer_spm_32k_3.model" \
  --quantize_4bit \
  --text_prompt "$ROBERT_PROMPT" \
  --voice_prompt "$VOICE_PROMPT" \
  --voice_prompt_dir "$VOICE_PROMPT_DIR" \
  --enable_moshi_reply \
  --direct_reply_hidden \
  --reply_hidden_steps_per_chunk 12 \
  --audio_chunk_sec 0.96 \
  --wav2vec_sec 0.96 \
  --fm_chunk_frames 24 \
  --prebuffer_chunks 1 \
  --render_sub_batch 8 \
  --renderer_precision fp32 \
  --frame_q_backpressure 32 \
  --buffer_ms 160 \
  --skip_fm_audio_encoder \
  --assistant_speech_rms_threshold "${ASSISTANT_SPEECH_RMS_THRESHOLD:-0.006}" \
  --assistant_speech_hold_chunks "${ASSISTANT_SPEECH_HOLD_CHUNKS:-1}" \
  --a_cfg_scale "$A_CFG_SCALE" \
  --nfe "$NFE" \
  --seed 42 \
  --noise_seed 42 \
  --shared_noise \
  --fp32 \
  --tf32 \
  --dump_motion \
  --dump_dir "$DUMP_DIR" \
  --silence_helium_path "$SILENCE_HELIUM_PATH" \
  --jpeg_quality 58 \
  --device cuda \
  --reply_audio_gain 1.0 \
  --output_audio_codec opus \
  "${SEARCH_ARGS[@]}"