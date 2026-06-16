# Registry mapping --baseline name -> adapter module exposing train/eval/score(args).
# None means the baseline folder exists under baselines/ but isn't wired up yet.
REGISTRY = {
    "molex": "baselines.molex._adapter",
    "aasist": None,
    "eaasist": None,
    "moef": None,
    "moef_icassp": None,
    "sls": None,
    "wav2vec2_aasist": None,
}
