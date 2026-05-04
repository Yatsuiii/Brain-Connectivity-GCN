#!/usr/bin/env python3
"""
Full pipeline demo: .1D / .npz → GCN prediction → LLM clinical report → Gradio UI.

Usage:
    python serve.py                              # full pipeline UI
    python serve.py --lora_dir finetune_data/asd_interpreter_lora
    python serve.py --llm_model meta-llama/Llama-3.1-8B-Instruct  # base model fallback
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# ── GCN pipeline (from predict.py) ─────────────────────────────────────────
_REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO))

from predict import (
    _CHECKPOINTS,
    _load_ensemble,
    predict_file,
    _FC_THRESHOLD,
    _MAX_WINDOWS,
    _STEP,
    _WINDOW_LEN,
)

SYSTEM_PROMPT = (
    "You are a clinical AI assistant specializing in functional MRI brain "
    "connectivity analysis for autism spectrum disorder (ASD) diagnosis support. "
    "You interpret outputs from a validated graph neural network (GCN) trained on "
    "the ABIDE I dataset and provide structured clinical summaries for neurologists "
    "and psychiatrists. Your reports are informative and evidence-based but always "
    "clarify that findings are AI-assisted and should be integrated with full "
    "clinical assessment. You do not make a diagnosis."
)


def confidence_label(p: float) -> str:
    if p >= 0.75:   return "HIGH"
    if p >= 0.6:    return "MODERATE"
    if p >= 0.4:    return "LOW / UNCERTAIN"
    if p >= 0.25:   return "MODERATE (TC)"
    return "HIGH (TC)"


def build_prompt(gcn_result: dict) -> str:
    p    = gcn_result["p_asd"]
    pm   = gcn_result["per_model"]
    site = gcn_result.get("site", "Unknown")
    n_tp = gcn_result.get("n_timepoints", "N/A")
    consensus = sum(1 for _, s in pm if s > 0.5)
    per_model_str = "\n".join(f"  {s}: {'ASD' if v>0.5 else 'TC'} (p={v:.3f})" for s, v in pm)
    return (
        f"Brain Connectivity GCN Analysis Report\n"
        f"{'='*40}\n"
        f"Acquisition Site : {site}\n"
        f"Timepoints       : {n_tp} TRs\n"
        f"p(ASD)           : {p:.3f}\n"
        f"Confidence Level : {confidence_label(p)}\n"
        f"Model Consensus  : {consensus}/4 site models predict ASD\n\n"
        f"Per-Model Breakdown (LOSO ensemble):\n{per_model_str}\n\n"
        f"Please provide a structured clinical interpretation of these findings."
    )


def llm_interpret(prompt: str, model, tokenizer, device: str, max_new_tokens: int = 512) -> str:
    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def load_llm(lora_dir: str | None, base_model: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = lora_dir if lora_dir and Path(lora_dir).exists() else base_model
    print(f"Loading LLM from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if lora_dir and Path(lora_dir).exists() and model_path == lora_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_dir)
        print("LoRA weights loaded.")

    model.eval()
    return model, tokenizer


def launch_gradio(gcn_models, llm_model, tokenizer, device: str) -> None:
    import gradio as gr

    def _predict(file_path: str | None) -> tuple[str, str]:
        if file_path is None:
            return "Upload a .1D or .npz file.", ""
        path = Path(file_path)
        try:
            result = predict_file(path, gcn_models, device, verbose=False)
        except Exception as exc:
            return f"GCN Error: {exc}", ""

        # GCN summary
        pct  = result["confidence"] * 100
        gcn_text = (
            f"Prediction : {result['prediction']}\n"
            f"Confidence : {pct:.1f}%  (p_ASD = {result['p_asd']:.3f})\n\n"
            f"Per-model breakdown:\n"
        )
        for site, p in result["per_model"]:
            bar = "█" * int(p * 20) + "░" * (20 - int(p * 20))
            lbl = "ASD" if p > 0.5 else "TC "
            gcn_text += f"  {site:>4}  {lbl}  {bar}  {p:.3f}\n"

        # LLM report
        try:
            prompt = build_prompt(result)
            report = llm_interpret(prompt, llm_model, tokenizer, device)
        except Exception as exc:
            report = f"LLM Error: {exc}"

        return gcn_text, report

    demo = gr.Interface(
        fn=_predict,
        inputs=gr.File(
            label="Upload ABIDE CC200 .1D or .npz file",
            file_types=[".1D", ".npz"],
            type="filepath",
        ),
        outputs=[
            gr.Textbox(label="GCN Prediction", lines=10),
            gr.Textbox(label="Clinical Report (LLM)", lines=20),
        ],
        title="BrainConnect ASD — Clinical AI Pipeline",
        description=(
            "Upload a CC200 resting-state fMRI file (.1D or .npz).\n"
            "Step 1: Scanner-site-invariant GCN predicts ASD vs Typical Control "
            "(LOSO AUC = 0.7872 across 4 independent institutions).\n"
            "Step 2: Fine-tuned LLM generates a structured clinical report.\n"
            "Running on AMD Instinct MI300X."
        ),
    )
    demo.launch(show_error=True, share=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_dir",   type=str, default="finetune_data/asd_interpreter_lora")
    parser.add_argument("--llm_model",  type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device",     type=str, default="auto")
    args = parser.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"Device: {device}")

    print(f"\nLoading GCN ensemble ({len(_CHECKPOINTS)} LOSO models)...")
    gcn_models = _load_ensemble(device)
    print(f"  → {len(gcn_models)} GCN models ready")

    print("\nLoading LLM...")
    llm_model, tokenizer = load_llm(args.lora_dir, args.llm_model, device)
    print("  → LLM ready")

    launch_gradio(gcn_models, llm_model, tokenizer, device)


if __name__ == "__main__":
    main()
