#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_CODE_BASE='3ef46650595e94e1a4d2ca48dc01e4ef1c424f1b'

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${repository_root}"

repository_commit="$(git rev-parse HEAD)"
echo "base_diagnosis_commit=${EXPECTED_CODE_BASE}"
echo "repository_commit=${repository_commit}"
if ! git merge-base --is-ancestor "${EXPECTED_CODE_BASE}" "${repository_commit}"; then
    echo "错误：当前仓库不包含要求的 Gate 1 诊断基线 ${EXPECTED_CODE_BASE}" >&2
    exit 1
fi

readonly log_root='logs/performance_diagnosis'
readonly report_root='reports/performance_diagnosis'
readonly checkpoint_root='checkpoints/performance_diagnosis'
mkdir -p "${log_root}" "${report_root}" "${checkpoint_root}"

report_paths=(
    "${report_root}/gradient_audit_seed_2026.json"
    "${report_root}/no_edge_top_p_seed_2026.json"
    "${report_root}/no_top_p_seed_2026.json"
    "${report_root}/stid_seed_2026.json"
    "${report_root}/no_family_top_p_seed_2026.json"
)
for report_path in "${report_paths[@]}"; do
    if [[ -e "${report_path}" ]]; then
        echo "错误：目标报告已存在：${report_path}" >&2
        echo "请先手动移动或删除旧结果，然后重新运行。" >&2
        exit 1
    fi
done

run_step() {
    local name="$1"
    local log_path="$2"
    shift 2
    echo '============================================================'
    echo "RUNNING: ${name}"
    echo '============================================================'
    "$@" 2>&1 | tee "${log_path}"
    echo "COMPLETE: ${name}"
}

run_step 'gradient audit' "${log_root}/gradient_audit_seed_2026.log" \
    python scripts/audit_hidfilter_gradients.py \
        --seed 2026 \
        --report-path "${report_root}/gradient_audit_seed_2026.json"

run_step 'no_edge_top_p' "${log_root}/no_edge_top_p_seed_2026.log" \
    python scripts/run_performance_diagnosis.py \
        --variant no_edge_top_p \
        --seed 2026 \
        --max-epochs 100 \
        --checkpoint-dir "${checkpoint_root}/no_edge_top_p_seed_2026" \
        --report-path "${report_root}/no_edge_top_p_seed_2026.json"

run_step 'no_top_p' "${log_root}/no_top_p_seed_2026.log" \
    python scripts/run_performance_diagnosis.py \
        --variant no_top_p \
        --seed 2026 \
        --max-epochs 100 \
        --checkpoint-dir "${checkpoint_root}/no_top_p_seed_2026" \
        --report-path "${report_root}/no_top_p_seed_2026.json"

run_step 'STID' "${log_root}/stid_seed_2026.log" \
    python scripts/run_stid_performance_diagnosis.py \
        --seed 2026 \
        --max-epochs 100 \
        --checkpoint-dir "${checkpoint_root}/stid_seed_2026" \
        --report-path "${report_root}/stid_seed_2026.json"

run_step 'no_family_top_p' "${log_root}/no_family_top_p_seed_2026.log" \
    python scripts/run_performance_diagnosis.py \
        --variant no_family_top_p \
        --seed 2026 \
        --max-epochs 100 \
        --checkpoint-dir "${checkpoint_root}/no_family_top_p_seed_2026" \
        --report-path "${report_root}/no_family_top_p_seed_2026.json"

echo 'GATE 1 AUTODL RUNS COMPLETE'
echo 'Reports:'
echo "- gradient audit: ${report_root}/gradient_audit_seed_2026.json"
echo "- no_edge_top_p: ${report_root}/no_edge_top_p_seed_2026.json"
echo "- no_top_p: ${report_root}/no_top_p_seed_2026.json"
echo "- STID: ${report_root}/stid_seed_2026.json"
echo "- no_family_top_p: ${report_root}/no_family_top_p_seed_2026.json"
echo 'NO TEST WAS RUN.'
