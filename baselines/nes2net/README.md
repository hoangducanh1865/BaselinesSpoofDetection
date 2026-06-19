# Reproduce Nes2Net on ASVspoof 5

Huong dan nay ghi lai cach chay lai thiet lap ASVspoof 5 trong paper
`Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech
Anti-spoofing`, dung WavLM Large lam front-end va Nes2Net lam back-end.

Paper dung ba metric cho ASVspoof 5 Track 1:

- `CLLR`
- `minDCF`
- `EER`

Ket qua benchmark trong paper cho ASVspoof 5 xap xi:

| Model | CLLR | minDCF | EER |
| --- | ---: | ---: | ---: |
| AASIST | 0.9587 | 0.1645 | 6.08 |
| Nes2Net | 0.7912 | 0.1568 | 6.13 |
| Nes2Net-X | 0.7344 | 0.1535 | 5.92 |

## 1. Environment

Dung Python 3.9 va PyTorch 1.13.1 CUDA 11.7.

```powershell
conda create --name asvspoof5 python=3.9
conda activate asvspoof5
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
python -m pip install -r requirements.txt
```

`requirements.txt` pin `numpy<2` de tranh loi:

```text
Failed to initialize NumPy: _ARRAY_API not found
```

Kiem tra nhanh:

```powershell
python -c "import torch, numpy; print(torch.__version__, torch.cuda.is_available(), numpy.__version__)"
```

## 2. Required Files

Can co ba nhom file:

1. ASVspoof 5 protocols:

```text
ASVspoof5_protocols/
  ASVspoof5.train.tsv
  ASVspoof5.dev.track_1.tsv
  ASVspoof5.eval.track_1.tsv
```

2. ASVspoof 5 eval audio, dang split:

```text
flac_E_aa/flac_E_eval/*.flac
flac_E_ab/flac_E_eval/*.flac
flac_E_ac/flac_E_eval/*.flac
flac_E_ad/flac_E_eval/*.flac
flac_E_ae/flac_E_eval/*.flac
flac_E_af/flac_E_eval/*.flac
```

3. Checkpoints:

```text
pretrained_models/WavLM-Large.pt
epoch_14_0.051.pth
```

`pretrained_models/WavLM-Large.pt` phai la WavLM Large checkpoint kieu
Microsoft/S3PRL co key `cfg` va `model`, khong phai HuggingFace
`pytorch_model.bin`. Code se fail-fast neu checkpoint khong dung Large config:

```text
encoder_layers = 24
encoder_embed_dim = 1024
encoder_attention_heads = 16
```

Neu dat checkpoint WavLM o noi khac, tao `.env`:

```text
WAVLM_LARGE_PATH=D:\path\to\WavLM-Large.pt
```

`.env` da duoc ignore boi git. Khong commit token hoac path rieng tu.

## 3. Config

Config reproduction nam o:

```text
config/WavLM_Nes2Net_ASVspoof5.conf
```

Model config dang dung:

```json
{
  "architecture": "wavlm_nes2net",
  "nb_samp": 64600,
  "dilation": 1,
  "pool_func": "mean",
  "SE_ratio": [1],
  "Nes_ratio": [8, 8]
}
```

`database_path` mac dinh la `data`, nhung eval-only script cung fallback sang
thu muc repo hien tai neu khong tim thay `data`. Cach chac chan nhat la de
`ASVspoof5_protocols/` va cac folder `flac_E_*` ngay trong repo, nhu layout o
muc 2.

## 4. Reproduce Eval With Pretrained Nes2Net

Chay 10,000 mau random, giong lenh da dung de debug va doi chieu nhanh:

```powershell
python main.py `
  --config config\WavLM_Nes2Net_ASVspoof5.conf `
  --eval_model_weights epoch_14_0.051.pth `
  --eval_subset_size 10000 `
  --eval_subset_seed 1234
```

Lenh nay se:

- load WavLM Large tu `pretrained_models/WavLM-Large.pt`
- load full finetuned checkpoint `epoch_14_0.051.pth`
- sample tu `ASVspoof5.eval.track_1.tsv` sau khi match voi audio local
- goi `model.eval()` truoc khi scoring
- ghi score va tinh `minDCF`, `EER`, `CLLR`

Output duoc ghi vao:

```text
exp_result/WavLM_Nes2Net_ASVspoof5_fulldev_estop5_bs64/eval_scores_using_best_dev_model.txt
exp_result/WavLM_Nes2Net_ASVspoof5_fulldev_estop5_bs64/loaded_model_result.txt
```

Ket qua 10,000 mau random seed `1234` tren local run hien tai:

```text
CM SYSTEM
        min DCF = 0.14212704235415605
        EER     = 5.397193534 %
        CLLR    = 0.732063262
```

Neu thay EER gan `50%`, kha nang cao la dang chay sai checkpoint WavLM, sai
label/protocol, hoac model dang o train mode. Ban code hien tai da guard cac
loi nay.

## 5. Full Eval

De chay tat ca eval files match duoc voi protocol local:

```powershell
python main.py `
  --config config\WavLM_Nes2Net_ASVspoof5.conf `
  --eval_model_weights epoch_14_0.051.pth `
  --eval_subset_size 0
```

Luu y: official eval protocol co 680,774 utterances. Neu local chi co mot phan
audio, script se in warning dang:

```text
WARNING: 271646/680774 eval protocol utterances were not found locally.
```

Ket qua full chi so sanh cong bang voi paper khi co day du eval audio theo
protocol.

## 6. Train From Scratch

Training path goc ky vong layout khac eval-only:

```text
data/
  ASVspoof5.train.tsv
  ASVspoof5.dev.track_1.tsv
  ASVspoof5.eval.track_1.tsv
  flac_T/*.flac
  flac_D/*.flac
  eval_full/flac_E_eval/*.flac
```

Sau khi dat `database_path` trong config tro den folder tren, chay:

```powershell
python main.py --train --config config\WavLM_Nes2Net_ASVspoof5.conf
```

Best checkpoints se duoc luu trong:

```text
exp_result/WavLM_Nes2Net_ASVspoof5_fulldev_estop5_bs64/weights/
```

## 7. Sanity Checks

Kiem tra WavLM checkpoint:

```powershell
python -c "import torch; ckpt=torch.load('pretrained_models/WavLM-Large.pt', map_location='cpu'); print(ckpt.keys()); print(ckpt['cfg']['encoder_layers'], ckpt['cfg']['encoder_embed_dim'], ckpt['cfg']['encoder_attention_heads'])"
```

Expected:

```text
dict_keys(['cfg', 'model'])
24 1024 16
```

Kiem tra score file co dung 10,000 dong:

```powershell
(Get-Content exp_result\WavLM_Nes2Net_ASVspoof5_fulldev_estop5_bs64\eval_scores_using_best_dev_model.txt | Measure-Object -Line).Lines
```

Kiem tra label distribution cua score file:

```powershell
python -c "from collections import Counter; p='exp_result/WavLM_Nes2Net_ASVspoof5_fulldev_estop5_bs64/eval_scores_using_best_dev_model.txt'; c=Counter(line.split()[3] for line in open(p) if len(line.split()) >= 4); print(c)"
```

## 8. Important Notes

- Score format la: `speaker_id utt_id score label`.
- `label` duoc lay tu cot 9 cua `ASVspoof5.eval.track_1.tsv`.
- Score dung class index `1`, tuong ung bonafide class trong training labels.
- `minDCF` va `CLLR` khong phai percent; chi `EER` in theo percent.
- Khong dung `inference_asvspoof5.py` cho reproduction nay; file do la batch
  inference cu cho cac model `wav2vec2_*`.

## Acknowledgement

Repo nay dua tren:

- Baseline-AASIST: https://github.com/asvspoof-challenge/asvspoof5/tree/main/Baseline-AASIST
- HM-Conformer noise augmentation: https://github.com/talkingnow/HM-Conformer/tree/main
- Microsoft WavLM implementation: https://github.com/microsoft/unilm/tree/master/wavlm

## Citation

```bibtex
@ARTICLE{Nes2Net,
  author={Liu, Tianchi and Truong, Duc-Tuan and Das, Rohan Kumar and Lee, Kong Aik and Li, Haizhou},
  journal={IEEE Transactions on Information Forensics and Security},
  title={Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-Spoofing},
  year={2025},
  volume={20},
  pages={12005-12018},
  keywords={Foundation models;Feature extraction;Computational modeling;Computer architecture;Computational efficiency;Dimensionality reduction;Acoustics;Kernel;Robustness;Deepfakes;Deepfake detection;speech anti-spoofing;Res2Net;Nes2Net;SSL;speech foundation model},
  doi={10.1109/TIFS.2025.3626963}
}
```
