# Huong dan cac baseline

Thu muc nay tong hop cach chuan bi pretrained model/checkpoint va cach chay cac baseline trong `./baselines`.

Moi baseline nen chay tu dung thu muc cua no de cac duong dan tuong doi trong code hoat dong dung:

```bash
cd baselines/<ten_baseline>
```

Danh sach tai lieu:

- [AASIST](./aasist.md)
- [wav2vec2_aasist](./wav2vec2_aasist.md)
- [XLSR-SLS](./xlsr_sls.md)
- [MoLEx](./molex.md)
- [MoEF](./moef.md)
- [Nes2Net](./nes2net.md)
- [Evolving AASIST](./eaasist.md)

Ghi chu chung:

- Cac baseline deu huong den bai toan audio anti-spoofing/deepfake detection va thuong can GPU CUDA.
- Duong dan dataset trong code/config hau het la placeholder, can sua thanh duong dan local truoc khi chay.
- Neu score file da ton tai, mot so script ghi bang che do append (`a+`), nen xoa score cu truoc khi evaluate lai de tranh lap dong.
