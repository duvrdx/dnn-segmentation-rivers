#!/usr/bin/env bash
set -euo pipefail

# ─── Configuração ───────────────────────────────────────────────────────────
DATA_DIRS=(
    "./Datasets/itapemirim_river"
    "./Datasets/doce_river"
)
BACKBONE="resnet152"
EPOCHS=50
BATCH_SIZE=8
IMG_SIZE=512
VAL_SPLIT=0.2
CROP_FACTOR=1
RESULTS_DIR="./test_results"

# Carrega configurações locais do .env se existir
if [[ -f ".env" ]]; then
    # shellcheck disable=SC1091
    source .env
fi

# Telegram (opcional - deixe vazio para desabilitar; lido do .env)
TELEGRAM_TOKEN="${TELEGRAM_TOKEN:-}"
TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-}"
# ────────────────────────────────────────────────────────────────────────────

# Log setup
mkdir -p ./logs
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="./logs/run_${TIMESTAMP}.log"
STDERR_FILE="./logs/run_${TIMESTAMP}.stderr"

# ─── Telegram helpers ────────────────────────────────────────────────────────
send_telegram() {
    local message="$1"
    if [[ -z "$TELEGRAM_TOKEN" || -z "$TELEGRAM_CHAT_ID" ]]; then
        echo "[telegram] não configurado, pulando notificação."
        return 0
    fi
    local response
    response=$(curl -s --max-time 10 \
        -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
        -d chat_id="${TELEGRAM_CHAT_ID}" \
        --data-urlencode "text=${message}" 2>&1)
    if [[ "$response" != *'"ok":true'* ]]; then
        echo "[telegram] erro sendMessage: $response"
    fi
}

send_telegram_photo() {
    local photo_path="$1"
    local caption="${2:-}"
    if [[ -z "$TELEGRAM_TOKEN" || -z "$TELEGRAM_CHAT_ID" ]]; then
        return 0
    fi
    local response
    response=$(curl -s --max-time 30 \
        -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendPhoto" \
        -F chat_id="${TELEGRAM_CHAT_ID}" \
        -F "photo=@${photo_path}" \
        -F "caption=${caption}" 2>&1)
    if [[ "$response" != *'"ok":true'* ]]; then
        echo "[telegram] erro sendPhoto (${photo_path}): $response"
    fi
}
# ────────────────────────────────────────────────────────────────────────────

# Ativa venv se existir
if [[ -f "./venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source ./venv/bin/activate
fi

send_telegram "🚀 Treino Iniciado!
Backbone: $BACKBONE | Epochs: $EPOCHS | Img size: $IMG_SIZE | Crop factor: $CROP_FACTOR
Datasets: ${DATA_DIRS[*]}"

# ─── Execução: stdout → tee (terminal + LOG_FILE), stderr → STDERR_FILE ─────
DATA_DIRS_STR=$(IFS=,; echo "${DATA_DIRS[*]}")
set +e
python test_siamese_2branch.py \
    --data_dir "$DATA_DIRS_STR" \
    --backbone "$BACKBONE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --val_split "$VAL_SPLIT" \
    --img_size "$IMG_SIZE" \
    --crop_factor "$CROP_FACTOR" \
    --results_dir "$RESULTS_DIR" \
    2>"$STDERR_FILE" | tee "$LOG_FILE"
EXIT_CODE="${PIPESTATUS[0]}"
set -e
# ────────────────────────────────────────────────────────────────────────────

if [[ "$EXIT_CODE" -ne 0 ]]; then
    echo ""
    echo "❌ Treino FALHOU (exit $EXIT_CODE)"

    # Append stderr ao final do LOG_FILE
    {
        echo ""
        echo "=== STDERR (exit $EXIT_CODE) ==="
        cat "$STDERR_FILE"
    } >> "$LOG_FILE"

    # Envia stderr limpo pelo Telegram
    STDERR_CONTENT="$(cat "$STDERR_FILE" 2>/dev/null || echo '(stderr indisponível)')"
    # Trunca em 3800 chars para respeitar limite do Telegram (4096)
    if [[ "${#STDERR_CONTENT}" -gt 3800 ]]; then
        STDERR_CONTENT="...(truncado)
${STDERR_CONTENT: -3800}"
    fi

    send_telegram "❌ Treino FALHOU (exit $EXIT_CODE)
Backbone: $BACKBONE | Img size: $IMG_SIZE | Crop factor: $CROP_FACTOR

--- STDERR ---
${STDERR_CONTENT}"

else
    echo ""
    echo "✅ Treino concluído! Log: $LOG_FILE"

    # Extrai bloco de métricas: de "RESULTADOS:" até o próximo "==...=="
    METRICS_BLOCK="$(awk '/^RESULTADOS:/{found=1} found{print} found && /^={10,}/{count++; if(count==1 && !/RESULTADOS/){exit}}' "$LOG_FILE")"
    if [[ -z "$METRICS_BLOCK" ]]; then
        METRICS_BLOCK="(bloco RESULTADOS não encontrado no log)"
    fi

    send_telegram "✅ Treino concluído!
Backbone: $BACKBONE | Epochs: $EPOCHS | Img size: $IMG_SIZE | Crop factor: $CROP_FACTOR
Datasets: ${DATA_DIRS[*]}

${METRICS_BLOCK}"

    # Envia imagens de test_results via sendPhoto
    PHOTO_COUNT=0
    while IFS= read -r -d '' img; do
        send_telegram_photo "$img" "$(basename "$img")"
        PHOTO_COUNT=$((PHOTO_COUNT + 1))
    done < <(find "$RESULTS_DIR" -maxdepth 1 \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) -print0 2>/dev/null | sort -z)

    if [[ "$PHOTO_COUNT" -eq 0 ]]; then
        echo "[telegram] nenhuma imagem encontrada em $RESULTS_DIR"
    else
        echo "[telegram] $PHOTO_COUNT imagem(ns) enviada(s)."
    fi
fi

exit "$EXIT_CODE"
