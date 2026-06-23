# Registry mapping --baseline name -> adapter module exposing train/eval/score(args).
# None means the baseline folder exists under baselines/ but isn't wired up yet.
REGISTRY = {
    "molex": "baselines.molex._adapter",
    # SEE-MoLEx ablations live in the repo-level src/ package (see --ablation).
    "see_molex": "src._adapter",
    "aasist": "baselines.aasist._adapter",
    "aasist_l": "baselines.aasist._adapter",
    "eaasist": None,
    "moef": "baselines.moef_icassp._adapter",
    "moef_icassp": "baselines.moef_icassp._adapter",
    "nes2net": "baselines.nes2net._adapter",
    "rawtfnet": "baselines.rawtfnet._adapter",
    "sls": "baselines.xlsr_sls._adapter",
    "w2v2_aasist": "baselines.wav2vec2_aasist._adapter",
    "wav2vec2_aasist": "baselines.wav2vec2_aasist._adapter",
    "xlsr_sls": "baselines.xlsr_sls._adapter",
}
