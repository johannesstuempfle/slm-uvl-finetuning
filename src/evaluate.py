"""
evaluate.py — unified evaluation module

Public API (importable):
    evaluate_pair(gen_path, gt_path)   compare two UVL files semantically
    evaluate(...)                       gin-configurable pipeline evaluation
    reeval(results_json_path)           re-score existing outputs without re-running inference

CLI:
    uv run src/evaluate.py compare generated.uvl ground_truth.uvl
    uv run src/evaluate.py batch generated_dir/ ground_truth_dir/
    uv run src/evaluate.py reeval results.json|model_dir|results_root/
"""

import difflib
import gin
import json
import numpy as np
import os
import peft
import re
import sys
import torch
import transformers

from flamapy.interfaces.python.flamapy_feature_model import FLAMAFeatureModel
from pathlib import Path
from sentence_transformers import SentenceTransformer


# ── UVL pair evaluation (formerly eval_semantics.py) ─────────────────────────

class InvalidUVL(Exception):
    pass


def _load_uvl(path):
    """Parse UVL and enumerate configurations. Raises InvalidUVL on any failure."""
    try:
        fm = FLAMAFeatureModel(path)
        model = fm.fm_model
        features = [f.name for f in model.get_features()]
    except Exception as e:
        raise InvalidUVL(f"parse error: {type(e).__name__}: {e}") from e
    if not features:
        raise InvalidUVL("parse error: parsed but contains no features")
    declared = set(features)
    referenced = set()
    for c in model.get_constraints():
        referenced |= set(c.get_features())
    dangling = referenced - declared
    if dangling:
        raise InvalidUVL(f"dangling reference: undeclared feature(s) {sorted(dangling)}")
    try:
        products = [frozenset(c.get_selected_elements()) for c in fm.configurations()]
    except Exception as e:
        raise InvalidUVL(f"analysis error: {type(e).__name__}: {e}") from e
    return features, products


_GROUP_KW = {"features", "constraints", "mandatory", "optional", "alternative",
             "or", "imports", "include", "namespace"}


def _extract_features_tolerant(path):
    feats, in_features, seen = [], False, set()
    for raw in open(path, encoding="utf-8", errors="replace"):
        st = raw.strip()
        if not st:
            continue
        if st == "features":
            in_features = True
            continue
        if st.startswith("constraints"):
            break
        if not in_features:
            continue
        if st.split()[0].strip('"').lower() in _GROUP_KW:
            continue
        m = re.match(r'"([^"]+)"', st)
        name = (m.group(1) if m else re.split(r"[\s{\[]", st)[0]).strip('"')
        if name and name.lower() not in _GROUP_KW and name not in seen:
            seen.add(name)
            feats.append(name)
    return feats


def _features_only(path):
    try:
        return [f.name for f in FLAMAFeatureModel(path).fm_model.get_features()]
    except Exception:
        return _extract_features_tolerant(path)


def _normalize(name):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    s = s.replace("_", " ").replace("-", " ").replace('"', " ")
    return " ".join(s.lower().split())


def _default_sim(a, b):
    return difflib.SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _align_features(gen_feats, gt_feats, sim=_default_sim, threshold=0.6):
    scored = sorted(
        ((sim(g, t), g, t) for g in gen_feats for t in gt_feats if sim(g, t) >= threshold),
        reverse=True,
    )
    used_g, used_t, matches = set(), set(), []
    for _, g, t in scored:
        if g in used_g or t in used_t:
            continue
        matches.append((g, t))
        used_g.add(g)
        used_t.add(t)
    gen_to_canon = {g: f"F::{t}" for g, t in matches}
    gt_to_canon = {t: f"F::{t}" for g, t in matches}
    return gen_to_canon, gt_to_canon, matches


def _project(products, name_map):
    return {frozenset(name_map[f] for f in p if f in name_map) for p in products}


_GROUPS = {"mandatory", "optional", "or", "alternative"}


def _extract_tree(path):
    edges, group_of, stack, in_features = set(), {}, [], False
    for raw in open(path, encoding="utf-8", errors="replace"):
        line = raw.replace("\t", "    ").rstrip("\n")
        st = line.strip()
        if not st:
            continue
        if st == "features":
            in_features = True
            continue
        if st.startswith("constraints"):
            break
        if not in_features:
            continue
        indent = len(line) - len(line.lstrip(" "))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        head = st.split()[0].strip('"').lower()
        if head in _GROUPS:
            stack.append((indent, "group", head))
            continue
        m = re.match(r'"([^"]+)"', st)
        name = m.group(1) if m else re.split(r"[\s{\[]", st)[0]
        gtype = parent = None
        for _, kind, nm in reversed(stack):
            if gtype is None and kind == "group":
                gtype = nm
            if kind == "feat":
                parent = nm
                break
        if parent is not None:
            edges.add((parent, name))
            if gtype:
                group_of[name] = gtype
        stack.append((indent, "feat", name))
    return edges, group_of


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _feature_metrics(gen_feats, gt_feats, sim, threshold):
    g2c, t2c, matches = _align_features(gen_feats, gt_feats, sim, threshold)
    tp = len(matches)
    fp = len(gen_feats) - tp
    fn = len(gt_feats) - tp
    p, r, f = _prf(tp, fp, fn)
    feat = {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f,
            "missed": sorted(set(gt_feats) - {t for _, t in matches}),
            "hallucinated": sorted(set(gen_feats) - {g for g, _ in matches})}
    return feat, g2c, t2c


def _semantic_metrics(gen_prods, gt_prods, g2c, t2c):
    G = _project(gen_prods, g2c)
    T = _project(gt_prods, t2c)
    tp = len(G & T)
    fp = len(G) - tp
    fn = len(T) - tp
    p, r, f = _prf(tp, fp, fn)
    return {"n_products_gen": len(G), "n_products_gt": len(T),
            "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f,
            "exact_equivalence": (fp == 0 and fn == 0)}


def _structural_metrics(gen_path, gt_path, sim, threshold):
    g_edges, g_groups = _extract_tree(gen_path)
    t_edges, t_groups = _extract_tree(gt_path)
    gf = sorted({f for e in g_edges for f in e})
    tf = sorted({f for e in t_edges for f in e})
    g2c, t2c, _ = _align_features(gf, tf, sim, threshold)

    def proj(edges, m):
        return {(m[p], m[c]) for p, c in edges if p in m and c in m}

    G, T = proj(g_edges, g2c), proj(t_edges, t2c)
    tp, fp, fn = len(G & T), len(G - T), len(T - G)
    p, r, f = _prf(tp, fp, fn)
    g_grp = {g2c[c]: t for c, t in g_groups.items() if c in g2c}
    t_grp = {t2c[c]: t for c, t in t_groups.items() if c in t2c}
    shared = set(g_grp) & set(t_grp)
    agree = sum(1 for c in shared if g_grp[c] == t_grp[c])
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f,
            "n_edges_gen": len(G), "n_edges_gt": len(T),
            "group_type_agreement": (agree / len(shared) if shared else None),
            "n_group_compared": len(shared)}


def evaluate_pair(gen_path, gt_path, sim=_default_sim, threshold=0.6):
    """Compare a generated UVL file against a ground-truth UVL file.

    Returns a dict with status 'ok', 'features_only', or 'failed'.
    """
    gt_feats, gt_prods = _load_uvl(gt_path)
    gt_nprod = len(_project(gt_prods, {t: f"F::{t}" for t in gt_feats}))
    base = {"gt_products": gt_nprod, "reason": None,
            "features": None, "structure": None, "semantics": None}
    try:
        gen_feats, gen_prods = _load_uvl(gen_path)
    except InvalidUVL as e:
        feats = _features_only(gen_path)
        if not feats:
            return {**base, "status": "failed", "reason": str(e)}
        feat, _, _ = _feature_metrics(feats, gt_feats, sim, threshold)
        struct = _structural_metrics(gen_path, gt_path, sim, threshold)
        return {**base, "status": "features_only", "reason": str(e),
                "features": feat, "structure": struct}
    feat, g2c, t2c = _feature_metrics(gen_feats, gt_feats, sim, threshold)
    struct = _structural_metrics(gen_path, gt_path, sim, threshold)
    sem = _semantic_metrics(gen_prods, gt_prods, g2c, t2c)
    return {**base, "status": "ok", "features": feat, "structure": struct, "semantics": sem}


# ── Pipeline helpers ──────────────────────────────────────────────────────────

_UVL_KEYWORDS = {
    'features', 'constraints', 'mandatory', 'optional', 'alternative',
    'or', 'imports', 'as', 'true', 'false', 'requires', 'excludes',
    'all', 'no', 'any', 'group',
}


def extract_feature_names(uvl_text):
    names = []
    in_features = False
    for line in uvl_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("'''") or stripped.startswith('"""'):
            continue
        if stripped == 'features':
            in_features = True
            continue
        if stripped == 'constraints':
            break
        if in_features:
            if stripped.startswith('"'):
                m = re.match(r'"([^"]+)"', stripped)
                if m:
                    names.append(m.group(1))
            else:
                word = stripped.split()[0]
                if word.lower() not in _UVL_KEYWORDS:
                    names.append(word)
    return names


def compute_semantic_similarity(text_a, text_b, embedding_model):
    if not text_a.strip() or not text_b.strip():
        return 0.0
    emb = embedding_model.encode([text_a, text_b], convert_to_numpy=True)
    norm_a, norm_b = np.linalg.norm(emb[0]), np.linalg.norm(emb[1])
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(emb[0], emb[1]) / (norm_a * norm_b))


def _flamapy_from_text(text):
    """Write UVL text to a named temp file and return a FLAMAFeatureModel."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".uvl", delete=False) as tmp:
        tmp.write(text.encode("ascii", "ignore"))
        tmp_path = tmp.name
    try:
        return FLAMAFeatureModel(tmp_path)
    finally:
        os.unlink(tmp_path)


def parsable(text):
    try:
        valid = _flamapy_from_text(text).satisfiable()
    except Exception:
        valid = False
    return valid


def run_flamapy_analysis(uvl_text):
    try:
        fm = _flamapy_from_text(uvl_text)
        return {
            "void": not fm.satisfiable(),
            "dead_features": fm.dead_features(),
            "false_optional_features": fm.false_optional_features(),
            "core_features": fm.core_features(),
            "configurations_number": fm.configurations_number(),
        }
    except Exception:
        return None


def evaluate_score(results):
    total = len(results)
    parsable_count = sum(1 for r in results if r.get("parsable") is True)
    return parsable_count / total if total else 0


def evaluate_semantic_score(results):
    if not results:
        return 0.0
    return sum(r.get("semantic_similarity", 0.0) for r in results) / len(results)


def evaluate_flamapy_summary(results):
    analyses = [r["flamapy_analysis"] for r in results if r.get("flamapy_analysis") is not None]
    if not analyses:
        return {}
    n = len(analyses)
    return {
        "analyzed": n,
        "void_rate": sum(1 for a in analyses if a["void"]) / n,
        "mean_dead_features": sum(len(a["dead_features"]) for a in analyses) / n,
        "mean_false_optional_features": sum(len(a["false_optional_features"]) for a in analyses) / n,
        "mean_configurations": sum(a["configurations_number"] for a in analyses) / n,
    }


def write_ground_truths(dataset_file, ground_truth_dir):
    os.makedirs(ground_truth_dir, exist_ok=True)
    with open(dataset_file) as f:
        data = json.load(f)
    for i, sample in enumerate(data["data"]):
        code = sample.get("code", "")
        if code:
            with open(os.path.join(ground_truth_dir, f"sample_{i:02d}.uvl"), "w", encoding="utf-8") as f:
                f.write(code)


def get_pure_uvl(text):
    match = re.search(r"'''uvl(.*?)'''", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _compute_metrics(result_data, generated_uvl_dir, ground_truth_dir):
    """Run semantic evaluation on result_data and return the full results dict."""
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
    return {
        "score": evaluate_score(result_data),
        "semantic_score": evaluate_semantic_score(result_data),
        "flamapy_summary": evaluate_flamapy_summary(result_data),
        "semantic_eval_summary": semantic_eval_summary,
        "eval_data": result_data,
    }


def _print_summary(results):
    sep = "=" * 55
    print(sep)
    print("EVALUATION SUMMARY")
    print(sep)
    print(f"  Syntax (parsable):       {results['score']:.3f}")
    print(f"  Sentence similarity:     {results['semantic_score']:.3f}")
    fl = results.get("flamapy_summary", {})
    if fl:
        print(f"  Flamapy analyzed:        {fl.get('analyzed', 0)}")
        print(f"    void rate:             {fl.get('void_rate', 0):.3f}")
        print(f"    mean dead features:    {fl.get('mean_dead_features', 0):.2f}")
        print(f"    mean configurations:   {fl.get('mean_configurations', 0):.1f}")
    se = results.get("semantic_eval_summary", {})
    if se:
        print()
        print("  Ground-truth comparison:")
        print(f"    ok / features_only / failed:  "
              f"{se['ok']} / {se['features_only']} / {se['failed']}")
        print(f"    mean feature F1:       {se['mean_feature_f1']:.3f}")
        print(f"    mean structural F1:    {se['mean_structural_f1']:.3f}")
        print(f"    mean semantic F1:      {se['mean_semantic_f1']:.3f}")
    print(sep)


# ── Gin-configurable pipeline evaluation ─────────────────────────────────────

@gin.configurable
def evaluate(model_name, output_dir, dataset_file, from_finetuned,
             semantic_model_name="all-MiniLM-L6-v2"):
    dataset_file = os.path.join(os.path.dirname(__file__), os.pardir, dataset_file)

    if from_finetuned:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        model = peft.PeftModel.from_pretrained(model=model, model_id=output_dir, device_map="auto")
        model = model.merge_and_unload()
        tokenizer = transformers.AutoTokenizer.from_pretrained(output_dir)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    pipeline = transformers.pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    with open(dataset_file) as f:
        test_data = json.load(f)

    dataset_dir = os.path.dirname(dataset_file)
    ground_truth_dir = os.path.join(dataset_dir, "ground_truth")
    write_ground_truths(dataset_file, ground_truth_dir)

    generated_uvl_dir = os.path.join(output_dir, "generated_uvl")
    os.makedirs(generated_uvl_dir, exist_ok=True)

    embedding_model = SentenceTransformer(semantic_model_name)

    result_data = []
    for i, sample in enumerate(test_data["data"]):
        text = sample["text"]
        prompt_messages = [
            {"role": "system", "content": "Can you please translate this text to uvl-Code? Please consider all informations from the text, but make sure, that the correct syntax is used: "},
            {"role": "user", "content": text},
        ]
        prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        output = pipeline(prompt, max_new_tokens=1024)
        model_output_only = output[0]["generated_text"][len(prompt):].strip()
        uvl_code = get_pure_uvl(model_output_only)

        gen_path = os.path.join(generated_uvl_dir, f"sample_{i:02d}.uvl")
        with open(gen_path, "w", encoding="utf-8") as f:
            f.write(uvl_code)

        model_valid = parsable(uvl_code)
        feature_names = extract_feature_names(uvl_code)
        result_data.append({
            "text": text,
            "output": uvl_code,
            "output_raw": model_output_only,
            "parsable": model_valid,
            "flamapy_analysis": run_flamapy_analysis(uvl_code) if model_valid else None,
            "feature_names": feature_names,
            "semantic_similarity": compute_semantic_similarity(
                text, " ".join(feature_names), embedding_model
            ) if feature_names else 0.0,
        })

    results = _compute_metrics(result_data, generated_uvl_dir, ground_truth_dir)

    with open(os.path.join(output_dir, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    _print_summary(results)
    return results


# ── Re-evaluation of existing outputs ────────────────────────────────────────

_DATASET_FILE = str(Path(__file__).parent.parent / "uvl-repos/test_dataset.json")
_GROUND_TRUTH_DIR = str(Path(__file__).parent.parent / "uvl-repos/ground_truth")
_SEMANTIC_MODEL = "all-MiniLM-L6-v2"
_embedding_model = None


def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(_SEMANTIC_MODEL)
    return _embedding_model


def reeval(results_json_path):
    """Re-score an existing results.json without re-running model inference."""
    results_json_path = Path(results_json_path)
    if results_json_path.is_dir():
        results_json_path = results_json_path / "full_results.json"

    if results_json_path.stat().st_size == 0:
        print(f"  SKIP (empty file): {results_json_path}")
        return None

    with open(results_json_path) as f:
        data = json.load(f)

    samples = data.get("test_results") or data.get("eval_data") or []
    if not samples:
        print(f"  SKIP (no samples): {results_json_path}")
        return None

    write_ground_truths(_DATASET_FILE, _GROUND_TRUTH_DIR)

    gen_dir = results_json_path.parent / "generated_uvl"
    gen_dir.mkdir(exist_ok=True)

    embedding_model = _get_embedding_model()

    result_data = []
    for i, sample in enumerate(samples):
        text = sample["text"]
        raw_output = sample["output"]
        uvl_code = get_pure_uvl(raw_output)

        gen_path = gen_dir / f"sample_{i:02d}.uvl"
        gen_path.write_text(uvl_code, encoding="utf-8")

        model_valid = parsable(uvl_code)
        feature_names = extract_feature_names(uvl_code)
        result_data.append({
            "text": text,
            "output": uvl_code,
            "output_raw": raw_output,
            "parsable": model_valid,
            "flamapy_analysis": run_flamapy_analysis(uvl_code) if model_valid else None,
            "feature_names": feature_names,
            "semantic_similarity": compute_semantic_similarity(
                text, " ".join(feature_names), embedding_model
            ) if feature_names else 0.0,
        })

    results = _compute_metrics(result_data, str(gen_dir), _GROUND_TRUTH_DIR)

    out_path = results_json_path.parent / "full_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    label = f"{results_json_path.parent.parent.name} / {results_json_path.parent.name}"
    print(f"\n{'=' * 55}\n  {label}")
    _print_summary(results)
    print(f"  Saved -> {out_path}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli_report(r):
    if r["status"] == "failed":
        print("STATUS: FAILED -- not parseable, no features extractable")
        print(f"  reason: {r['reason']}")
        return
    if r["status"] == "features_only":
        print("STATUS: NOT PARSEABLE -- feature-level only")
        print(f"  reason: {r['reason']}")
    f = r["features"]
    print(f"Feature F1: P={f['precision']:.3f} R={f['recall']:.3f} F1={f['f1']:.3f}"
          f"  (TP={f['tp']} FP={f['fp']} FN={f['fn']})")
    if f["missed"]:       print(f"  missed:       {f['missed']}")
    if f["hallucinated"]: print(f"  hallucinated: {f['hallucinated']}")
    st = r.get("structure")
    if st:
        print(f"Structure F1: P={st['precision']:.3f} R={st['recall']:.3f} F1={st['f1']:.3f}")
    s = r.get("semantics")
    if s:
        print(f"Semantic F1:  P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f}"
              f"  (products gen={s['n_products_gen']} gt={s['n_products_gt']})")
        print(f"  exact equivalence: {s['exact_equivalence']}")
    else:
        print("Semantic F1:  N/A (model not parseable)")


def _cli_batch(gen_dir, gt_dir):
    import glob
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "sample_*.uvl")))
    if not gt_files:
        print(f"No sample_*.uvl files in {gt_dir}")
        sys.exit(1)
    all_results = []
    for gt_path in gt_files:
        name = os.path.basename(gt_path)
        gen_path = os.path.join(gen_dir, name)
        if not os.path.exists(gen_path):
            print(f"  SKIP {name}: not found in {gen_dir}")
            all_results.append({"sample": name, "status": "missing_generated"})
            continue
        print(f"--- {name} ---")
        try:
            r = evaluate_pair(gen_path, gt_path)
        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({"sample": name, "status": "error", "reason": str(e)})
            continue
        _cli_report(r)
        all_results.append({"sample": name, **r})
    out = os.path.join(gen_dir, "semantic_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nWrote {out}")
    ok = [r for r in all_results if r.get("status") == "ok"]
    fo = [r for r in all_results if r.get("status") == "features_only"]
    fa = [r for r in all_results if r.get("status") == "failed"]
    print(f"Summary: {len(ok)} ok / {len(fo)} features_only / {len(fa)} failed")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="UVL evaluation utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  uv run src/evaluate.py compare generated.uvl ground_truth.uvl\n"
            "  uv run src/evaluate.py batch generated_uvl/ uvl-repos/ground_truth/\n"
            "  uv run src/evaluate.py reeval results/Supervised/gemma-3-1b-it/\n"
            "  uv run src/evaluate.py reeval results/\n"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_compare = sub.add_parser("compare", help="Compare a single generated/ground-truth pair")
    p_compare.add_argument("generated", help="Generated UVL file")
    p_compare.add_argument("ground_truth", help="Ground-truth UVL file")

    p_batch = sub.add_parser("batch", help="Batch-compare all sample_XX.uvl pairs")
    p_batch.add_argument("generated_dir", help="Directory of generated UVL files")
    p_batch.add_argument("ground_truth_dir", help="Directory of ground-truth UVL files")

    p_reeval = sub.add_parser("reeval", help="Re-score existing results without re-running inference")
    p_reeval.add_argument("target", help="results.json file, model directory, or results root directory")

    args = parser.parse_args()

    if args.cmd == "compare":
        _cli_report(evaluate_pair(args.generated, args.ground_truth))

    elif args.cmd == "batch":
        _cli_batch(args.generated_dir, args.ground_truth_dir)

    elif args.cmd == "reeval":
        target = Path(args.target)
        if target.is_dir():
            files = sorted(target.rglob("full_results.json"))
            if not files:
                print(f"No full_results.json found under {target}")
                sys.exit(1)
            print(f"Found {len(files)} result files — loading embedding model once...")
            _get_embedding_model()
            for f in files:
                reeval(f)
        else:
            reeval(target)
