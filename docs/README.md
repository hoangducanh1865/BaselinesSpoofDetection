# Huong dan cac baseline

Thu muc nay tong hop cach chuan bi pretrained model/checkpoint va cach chay cac baseline trong `./baselines`.

Voi cac baseline da duoc noi vao CLI chung, nen chay tu root repo:

```bash
cd /home/user14/anhhd/spoof/BaselinesSpoofDetection
python main.py --baseline <ten_baseline> --mode eval --dataset <dataset>
```

Danh sach tai lieu:

- [AASIST](./aasist.md)
- [wav2vec2_aasist](./wav2vec2_aasist.md)
- [XLSR-SLS](./xlsr_sls.md)
- [MoLEx](./molex.md)
- [MoEF](./moef.md)
- [Nes2Net](./nes2net.md)
- [RawTFNet](./rawtfnet.md)
- [Whisper-MFCC-MesoNet](./whisper_mfcc_mesonet.md)
- [Evolving AASIST](./eaasist.md)
- [VSASV speaker-disjoint resplit](./vsasv_resplit.md)

Ghi chu chung:

- Cac baseline deu huong den bai toan audio anti-spoofing/deepfake detection va thuong can GPU CUDA.
- Duong dan dataset cho CLI chung duoc quan ly trong `datasets/registry.py`; co the override bang `SPOOF_DATA_ROOT`.
- Neu score file da ton tai, mot so script ghi bang che do append (`a+`), nen xoa score cu truoc khi evaluate lai de tranh lap dong.
