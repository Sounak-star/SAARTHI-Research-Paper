import json

with open('results/saarthi_results.json') as f:
    r = json.load(f)

cm = r['confusion_matrices']['SAARTHI (XGBoost)']
tn, fp = cm[0][0], cm[0][1]
fn, tp = cm[1][0], cm[1][1]
prec = tp / (tp + fp)
rec  = tp / (tp + fn)
print(f'SAARTHI Precision={prec:.4f}  Recall={rec:.4f}')
print(f'  TN={tn}  FP={fp}  FN={fn}  TP={tp}')

sweep = r['delta_sensitivity']
delta_f1s = [v['delta_f1'] for v in sweep.values()]
print(f'Min |DeltaF1|={min(abs(x) for x in delta_f1s)}')
print(f'Max |DeltaF1|={max(abs(x) for x in delta_f1s)}')

fs = r['feature_stats']
shift = fs['kurtosis_p90_normal'] - fs['kurtosis_p90_anomaly']
print(f'Kurtosis p90 shift (Normal-Anomaly) = {shift:.3f}')

sig = r['significance']
print(f'Friedman chi2={sig["friedman_chi2"]}, p={sig["friedman_p"]}')
print(f'Wilcoxon p={sig["wilcoxon_p"]}')

hw = r['hardware_benchmarks']
print(f'Pickle={hw["model_footprint"]["pickle_kb"]} KB')
print(f'Latency mean={hw["inference_latency"]["mean_ms"]} ms')
print(f'Latency p99={hw["inference_latency"]["p99_ms"]} ms')
