"""Reference-free V5 inference.

Only Japanese text (and a fixed internal speaker id) is accepted. No reference
audio path, no prompt text, no reference cache. Run with STRICT_NO_REF=1 to make
the known audio-read/ref APIs raise instead of silently reading audio.

This is a sanitized strict wrapper. It adds six read/ref guards on top of the
historical two-guard runtime used for the paper's logs; the six-guard version
has not been run against the private corpus. It still requires a local
GPT-SoVITS V5 package and a trained checkpoint; neither is published here.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

PROJ = Path(__file__).resolve().parents[1]
CALLER_CWD = Path.cwd()


def deny(*a, **k):
    raise RuntimeError("audio read/ref API called in STRICT_NO_REF mode")


def main():
    ap = argparse.ArgumentParser(description="Strict reference-free GPT-SoVITS V5 inference")
    ap.add_argument("--v5-root", default=os.environ.get("V5_ROOT", ""),
                    help="path to GPT-SoVITS-v5 package; or set V5_ROOT")
    ap.add_argument("--checkpoint", default=str(PROJ / "weights" / "e2_v5dev.pth"))
    ap.add_argument("--text", default="")
    ap.add_argument("--texts-file", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--speaker", default="kan")
    ap.add_argument("--t2s", default="")
    ap.add_argument("--film", default="", help="optional E5 speaker-FiLM delta")
    ap.add_argument("--spk-scale", type=float, default=1.0)
    ap.add_argument("--vocoder", default="")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--cfg", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--raw", action="store_true", help="disable fixed C1M mastering")
    ap.add_argument("--c1-fir", default=os.environ.get("C1_FIR", str(PROJ / "data" / "c1_fir.npy")))
    args = ap.parse_args()

    if not args.v5_root:
        raise SystemExit("--v5-root or V5_ROOT is required (V5 package is not published)")
    V5 = Path(args.v5_root).resolve()
    if not (V5 / "GPT_SoVITS").is_dir():
        raise SystemExit(f"not a GPT-SoVITS V5 root: {V5}")
    sys.path.insert(0, str(V5))
    sys.path.insert(0, str(PROJ / "code"))
    os.chdir(V5)

    for name in ("checkpoint", "texts_file", "out", "out_dir", "film", "vocoder"):
        value = getattr(args, name)
        if value and not Path(value).is_absolute():
            setattr(args, name, str((CALLER_CWD / value).resolve()))

    if not Path(args.checkpoint).is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if args.c1_fir:
        os.environ["C1_FIR"] = str(Path(args.c1_fir).resolve())

    # Import after the V5 root is known.
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config  # noqa: E402
    from c1_master import process as master_c1  # noqa: E402

    if os.environ.get("STRICT_NO_REF") == "1":
        import torchaudio
        import GPT_SoVITS.TTS_infer_pack.TTS as tts_mod

        torchaudio.load = deny
        sf.read = deny
        try:
            import librosa
            librosa.load = deny
        except ImportError:
            pass
        for attr in ("set_ref_audio", "_set_ref_spec", "_set_prompt_semantic"):
            if hasattr(tts_mod.TTS, attr):
                setattr(tts_mod.TTS, attr, deny)

    base = yaml.safe_load((V5 / "GPT_SoVITS/configs/tts_infer.yaml").read_text(encoding="utf-8"))
    custom = dict(base["v5dev"])
    t2s = Path(args.t2s).resolve() if args.t2s else V5 / "GPT_SoVITS/pretrained_models/s1v3.ckpt"
    custom.update({
        "device": args.device,
        "is_half": not args.fp32,
        "version": "v5dev",
        "t2s_weights_path": str(t2s),
        "vits_weights_path": str(V5 / "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5dev.pth"),
    })
    for key in ("bert_base_path", "cnhuhbert_base_path"):
        custom[key] = str(V5 / custom[key])
    pipe = TTS(TTS_Config({"custom": custom, "v5dev": custom}))
    pipe.init_vits_weights(args.checkpoint)

    if args.vocoder:
        vd = torch.load(args.vocoder, map_location="cpu", weights_only=False)
        sd = vd["state_dict"] if isinstance(vd, dict) and "state_dict" in vd else vd
        miss, unexpected = pipe.vocoder.load_state_dict(sd, strict=False)
        print(f"[vocoder] loaded={args.vocoder} missing={len(miss)} unexpected={len(unexpected)}")

    if args.film:
        from process_ckpt import load_sovits_new
        state = load_sovits_new(args.film)["weight"]
        film = {k.split("spk_film.", 1)[1]: v for k, v in state.items() if "spk_film." in k}
        target = getattr(pipe.vits_model.cfm, "spk_film", None)
        if target is None:
            raise RuntimeError("film delta requested but model has no spk_film")
        miss, unexpected = target.load_state_dict(film, strict=False)
        print(f"[film] loaded keys={len(film)} missing={len(miss)} unexpected={len(unexpected)}")
        if "spk_emb" in state:
            with torch.no_grad():
                pipe.vits_model.spk_emb.copy_(state["spk_emb"].to(pipe.vits_model.spk_emb.dtype))

    if args.spk_scale != 1.0:
        if not hasattr(pipe.vits_model, "spk_emb"):
            raise RuntimeError("--spk-scale requested but checkpoint has no trained spk_emb")
        with torch.no_grad():
            pipe.vits_model.spk_emb.mul_(args.spk_scale)
        print(f"[spk] scale={args.spk_scale} norm={float(pipe.vits_model.spk_emb.float().norm()):.3f}")

    if args.texts_file:
        texts = [x.strip() for x in Path(args.texts_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    elif args.text:
        texts = [args.text]
    else:
        raise SystemExit("provide --text or --texts-file")
    if len(texts) > 1 and not args.out_dir:
        raise SystemExit("--out-dir is required for multiple texts")
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for index, text in enumerate(texts, 1):
        inputs = {
            "text": text, "text_lang": "ja", "ref_free": True,
            "speaker_id": args.speaker, "top_k": 15, "top_p": 1.0,
            "temperature": 1.0, "text_split_method": "cut1", "batch_size": 1,
            "speed_factor": args.speed, "parallel_infer": False,
            "repetition_penalty": 1.35, "sample_steps": args.steps,
            "cfg_rate": args.cfg, "seed": args.seed,
            "return_fragment": False, "streaming_mode": False,
        }
        chunks, sr = [], None
        for out_sr, audio in pipe.run(inputs):
            sr = int(out_sr)
            chunks.append(np.asarray(audio))
        if not chunks:
            raise RuntimeError(f"no audio returned for text {index}")
        wav = np.concatenate([c.reshape(-1) for c in chunks])
        mastered = None
        if not args.raw:
            wav, mastered = master_c1(wav.astype(np.float32), int(sr))
        if len(texts) == 1 and args.out:
            out = Path(args.out)
        else:
            out = out_dir / f"s{index:02d}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, wav, sr)
        meta = {
            "index": index, "text": text, "speaker": args.speaker,
            "checkpoint": str(args.checkpoint), "ref_free": True,
            "ref_audio_path": None, "prompt_text": None,
            "sample_rate": sr, "samples": int(wav.size),
            "duration_sec": round(float(wav.size) / sr, 4),
            "rms": float(np.sqrt(np.mean(wav.astype(np.float64) ** 2))),
            "peak": float(np.max(np.abs(wav.astype(np.float64)))),
            "out": str(out), "mastered": mastered,
        }
        results.append(meta)
        print(json.dumps(meta, ensure_ascii=False))
        out.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    if out_dir:
        (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
