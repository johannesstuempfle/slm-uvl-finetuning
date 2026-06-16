#!/usr/bin/env python3
"""
Evaluate any plain HuggingFace model on the UVL test dataset without fine-tuning,
computing the same metrics as the main pipeline (Score, Feat. F1, Sem. F1, etc.).

Usage:
    uv run baseline.py Qwen/Qwen2.5-7B-Instruct
    uv run baseline.py --dataset uvl-repos/test_dataset.json Qwen/Qwen2.5-7B-Instruct
    uv run baseline.py --load-in-8bit google/gemma-3-4b-it
"""

import argparse
import json
import os
import sys

import torch
import transformers
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.evaluate import (
    _print_summary,
    compute_semantic_similarity,
    evaluate_flamapy_summary,
    evaluate_pair,
    evaluate_score,
    evaluate_semantic_score,
    extract_feature_names,
    get_pure_uvl,
    parsable,
    run_flamapy_analysis,
    write_ground_truths,
)


def run(model_name, output_dir, dataset_file, semantic_model_name, load_in_8bit, hf_token):
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)

    os.makedirs(output_dir, exist_ok=True)

    quant_kwargs = {}
    if load_in_8bit:
        from transformers import BitsAndBytesConfig
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", **quant_kwargs
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    pipeline = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        dtype=None if load_in_8bit else torch.bfloat16,
        device_map="auto",
    )

    dataset_file = os.path.abspath(dataset_file)
    with open(dataset_file) as f:
        test_data = json.load(f)

    ground_truth_dir = os.path.join(os.path.dirname(dataset_file), "ground_truth")
    write_ground_truths(dataset_file, ground_truth_dir)

    generated_uvl_dir = os.path.join(output_dir, "generated_uvl")
    os.makedirs(generated_uvl_dir, exist_ok=True)

    embedding_model = SentenceTransformer(semantic_model_name)

    result_data = []
    n = len(test_data["data"])
    for i, sample in enumerate(test_data["data"]):
        text = sample["text"]
        print(f"[{i+1}/{n}] {text[:70]}...")

        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "Can you please translate this text to uvl-Code? "
                    "Please consider all informations from the text, but make sure, "
                    "that the correct syntax is used: "
                ),
            },
            {"role": "user", "content": text},
        ]
        prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        output = pipeline(prompt, max_new_tokens=1024)
        model_output_only = output[0]["generated_text"][len(prompt):].strip()
        uvl_code = get_pure_uvl(model_output_only)

        gen_path = os.path.join(generated_uvl_dir, f"sample_{i:02d}.uvl")
        with open(gen_path, "w", encoding="utf-8") as f:
            f.write(uvl_code)

        model_valid = parsable(uvl_code)
        flamapy_analysis = run_flamapy_analysis(uvl_code) if model_valid else None
        feature_names = extract_feature_names(uvl_code)
        sem_sim = (
            compute_semantic_similarity(text, " ".join(feature_names), embedding_model)
            if feature_names
            else 0.0
        )

        result_data.append(
            {
                "text": text,
                "output": uvl_code,
                "output_raw": model_output_only,
                "parsable": model_valid,
                "flamapy_analysis": flamapy_analysis,
                "feature_names": feature_names,
                "semantic_similarity": sem_sim,
            }
        )

    for i, rd in enumerate(result_data):
        gen_path = os.path.join(generated_uvl_dir, f"sample_{i:02d}.uvl")
        gt_path = os.path.join(ground_truth_dir, f"sample_{i:02d}.uvl")
        if os.path.exists(gt_path):
            try:
                rd["semantic_eval"] = evaluate_pair(gen_path, gt_path)
            except Exception as e:
                rd["semantic_eval"] = {"status": "error", "reason": str(e)}
        else:
            rd["semantic_eval"] = {"status": "no_ground_truth"}

    sem_evals = [rd["semantic_eval"] for rd in result_data]
    ok = [s for s in sem_evals if s.get("status") == "ok"]
    fo = [s for s in sem_evals if s.get("status") == "features_only"]
    fa = [s for s in sem_evals if s.get("status") == "failed"]
    semantic_eval_summary = {
        "ok": len(ok),
        "features_only": len(fo),
        "failed": len(fa),
        "mean_feature_f1": sum(s["features"]["f1"] for s in ok + fo if s.get("features")) / max(len(ok) + len(fo), 1),
        "mean_semantic_f1": sum(s["semantics"]["f1"] for s in ok) / max(len(ok), 1),
        "mean_structural_f1": sum(s["structure"]["f1"] for s in ok + fo if s.get("structure")) / max(len(ok) + len(fo), 1),
    }

    results = {
        "score": evaluate_score(result_data),
        "semantic_score": evaluate_semantic_score(result_data),
        "flamapy_summary": evaluate_flamapy_summary(result_data),
        "semantic_eval_summary": semantic_eval_summary,
        "eval_data": result_data,
    }

    out_path = os.path.join(output_dir, "full_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    _print_summary(results)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a plain HuggingFace model on the UVL test dataset."
    )
    parser.add_argument(
        "model_name",
        help="HuggingFace model id, e.g. Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--dataset",
        default="uvl-repos/test_dataset.json",
        help="Path to test dataset JSON (default: uvl-repos/test_dataset.json)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: results/Baseline/<model-name>)",
    )
    parser.add_argument(
        "--semantic-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model for semantic similarity (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--load-in-8bit",
        action="store_true",
        help="Load the model in 8-bit quantization to reduce GPU memory usage",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace access token (defaults to HF_TOKEN env var)",
    )
    args = parser.parse_args()

    model_short = args.model_name.split("/")[-1]
    output_dir = args.output_dir or os.path.join("results", "Baseline", model_short)

    run(
        model_name=args.model_name,
        output_dir=output_dir,
        dataset_file=args.dataset,
        semantic_model_name=args.semantic_model,
        load_in_8bit=args.load_in_8bit,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
