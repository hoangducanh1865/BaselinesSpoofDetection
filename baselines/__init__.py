# Registry mapping --baseline name -> adapter module exposing train/eval/score(args).
# None means the baseline folder exists under baselines/ but isn't wired up yet.
REGISTRY = {
    "molex": "baselines.molex._adapter",
    "aasist": "baselines.aasist._adapter",
    "aasist_l": "baselines.aasist._adapter",
    "eaasist": None,
    "moef": None,
    "moef_icassp": None,
    "nes2net": "baselines.nes2net._adapter",
    "sls": None,
    "wav2vec2_aasist": None,
}
