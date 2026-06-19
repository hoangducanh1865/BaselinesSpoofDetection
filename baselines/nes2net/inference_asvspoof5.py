import argparse
import os
import torch
import librosa
import numpy as np
from tqdm import tqdm


def pad(x, max_len):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def load_protocol(protocol_path):
    """Parse ASVspoof5 .tsv protocol. Returns list of (utt_id, label)."""
    entries = []
    with open(protocol_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            utt_id = parts[1]   # e.g. E_0009538969
            label = parts[8]    # 'spoof' or 'bonafide'
            entries.append((utt_id, label))
    return entries


def build_file_index(data_dir):
    """Walk flac_E_*/flac_E_eval/ and index utt_id -> full path."""
    index = {}
    for entry in os.listdir(data_dir):
        eval_dir = os.path.join(data_dir, entry, 'flac_E_eval')
        if not os.path.isdir(eval_dir):
            continue
        for fname in os.listdir(eval_dir):
            if fname.lower().endswith('.flac'):
                utt_id = os.path.splitext(fname)[0]
                index[utt_id] = os.path.join(eval_dir, fname)
    return index


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.model_name == 'wav2vec2_AASIST':
        from model_scripts.wav2vec2_AASIST import Model
    elif args.model_name == 'wav2vec2_Nes2Net_X':
        from model_scripts.wav2vec2_Nes2Net_X import wav2vec2_Nes2Net_no_Res_w_allT as Model
    else:
        raise ValueError(f"Unknown model: {args.model_name}")

    model = Model(args, device).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Model loaded: {args.model_path}")
    model.eval()

    # Load protocol to get ordered utterance list
    protocol = load_protocol(args.protocol)
    print(f"Protocol entries: {len(protocol)}")

    # Build index: utt_id -> file path
    file_index = build_file_index(args.data_dir)
    print(f"Indexed files: {len(file_index)}")

    # Check coverage
    missing = [uid for uid, _ in protocol if uid not in file_index]
    if missing:
        print(f"WARNING: {len(missing)} utterances in protocol not found in data_dir")

    # Load already-scored utterances to allow resuming
    already_scored = set()
    if os.path.exists(args.eval_output):
        with open(args.eval_output, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    already_scored.add(parts[0])
        print(f"Resuming: {len(already_scored)} already scored, skipping.")

    to_process = [(uid, lbl) for uid, lbl in protocol
                  if uid in file_index and uid not in already_scored]
    print(f"To process: {len(to_process)}")

    if not to_process:
        print("All utterances already scored. Done.")
        return

    with torch.no_grad(), open(args.eval_output, 'a') as fout:
        for utt_id, label in tqdm(to_process, desc="Scoring"):
            audio_path = file_index[utt_id]
            try:
                audio, _ = librosa.load(audio_path, sr=16000, mono=True)
                if args.test_mode == '4s':
                    audio = pad(audio, 64000)
                x = torch.tensor(audio).unsqueeze(0).to(device)
                score = model(x)[:, 1].item()
                fout.write(f"{utt_id} {score}\n")
            except Exception as e:
                print(f"Error on {utt_id}: {e}")

    print(f"Scores saved to {args.eval_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof5 batch inference")
    # model config
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["wav2vec2_AASIST", "wav2vec2_Nes2Net_X"])
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--n_output_logits", type=int, default=2)
    parser.add_argument("--dilation", type=int, default=2)
    parser.add_argument("--pool_func", type=str, default="mean", choices=["mean", "ASTP"])
    parser.add_argument("--SE_ratio", type=int, nargs="+", default=[1])
    parser.add_argument("--Nes_ratio", type=int, nargs="+", default=[8, 8])
    parser.add_argument("--AASIST_scale", type=int, default=32)
    # data / protocol / output
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root folder containing flac_E_aa/, flac_E_ab/, ...")
    parser.add_argument("--protocol", type=str, required=True,
                        help="Path to ASVspoof5 .tsv protocol file")
    parser.add_argument("--eval_output", type=str, default="scores_asvspoof5.txt")
    parser.add_argument("--test_mode", type=str, default="full", choices=["4s", "full"])
    args = parser.parse_args()
    main(args)
