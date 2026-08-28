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
LOG_FILE="./logs/run_$(date +%Y%m%d_%H%M%S).log"

# Telegram helper
send_telegram() {
    local message="$1"
    if [[ -z "$TELEGRAM_TOKEN" || -z "$TELEGRAM_CHAT_ID" ]]; then
        echo "[telegram] não configurado, pulando notificação."
        return 0
    fi
    curl -s --max-time 10 \
        -X POST "https://api.telegram.org/bot${TELEGRAM_TOKEN}/sendMessage" \
        -d chat_id="${TELEGRAM_CHAT_ID}" \
        -d text="${message}" \
        -d parse_mode="HTML" \
        > /dev/null 2>&1 \
        || echo "[telegram] falha no curl, continuando."
}

# Ativa venv se existir
if [[ -f "./venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source ./venv/bin/activate
fi

# Roda com tee: terminal + log simultaneamente
{
    DATA_DIRS_STR=$(IFS=,; echo "${DATA_DIRS[*]}")
    python test_siamese_2branch.py \
        --data_dir "$DATA_DIRS_STR" \
        --backbone "$BACKBONE" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --val_split "$VAL_SPLIT" \
        --img_size "$IMG_SIZE" \
        --crop_factor "$CROP_FACTOR" \
        --results_dir "$RESULTS_DIR"
} 2>&1 | tee "$LOG_FILE"

EXIT_CODE="${PIPESTATUS[0]}"

if [[ "$EXIT_CODE" -eq 0 ]]; then
    echo ""
    echo "✅ Treino concluído! Log: $LOG_FILE"
    send_telegram "✅ Treino concluído!
Backbone: $BACKBONE | Epochs: $EPOCHS | Img size: $IMG_SIZE | Crop factor: $CROP_FACTOR
Datasets: ${DATA_DIRS[*]}
Log: $LOG_FILE"
else
    echo ""
    echo "❌ Treino FALHOU (exit $EXIT_CODE)"
    LAST_LINES="$(tail -n 20 "$LOG_FILE" 2>/dev/null || echo '(log indisponível)')"
    send_telegram "❌ Treino FALHOU (exit $EXIT_CODE)
Backbone: $BACKBONE | Img size: $IMG_SIZE | Crop factor: $CROP_FACTOR
Últimas 20 linhas do log:
$LAST_LINES"
fi

exit "$EXIT_CODE"
