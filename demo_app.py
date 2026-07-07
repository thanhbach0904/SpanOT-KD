"""
Streamlit demo for SpanOT-KD student checkpoints.

Loads a (proposed_method, vanilla) checkpoint pair for a chosen architecture/seed
from ../saved_model and runs side-by-side greedy generation on a QED-style prompt.

Run with:
    streamlit run demo_app.py
"""
import gc
import json
import random
import re
import time
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
SAVED_MODEL_DIR = SCRIPT_DIR.parent / "saved_model"
QED_DATASET_DIR = SCRIPT_DIR / "llm_distillation" / "datasets" / "hf" / "qedllama3"
# Produced by compare_difference.ipynb for the OPT proposed_method/vanilla pair only.
DIFFERING_CASES_PATH = SCRIPT_DIR / "compare_outputs" / "analysis_results.json"

# Tokenizer isn't saved inside a checkpoint dir (weights only), so it's loaded
# separately per architecture: prefer the local copy used at training time,
# fall back to the public hub id if that folder isn't present on this machine.
ARCH_TOKENIZERS = {
    "opt": {"local": SCRIPT_DIR / "EleutherAI" / "opt-350m", "hub": "facebook/opt-350m"},
    "pythia": {"local": SCRIPT_DIR / "EleutherAI" / "pythia-410m", "hub": "EleutherAI/pythia-410m"},
}

WEIGHT_FILENAMES = ("model.safetensors", "pytorch_model.bin")


def _has_weights(checkpoint_dir: Path) -> bool:
    return any((checkpoint_dir / name).is_file() for name in WEIGHT_FILENAMES)


@st.cache_data(show_spinner=False)
def discover_pairs(saved_model_dir: str) -> dict[str, list[str]]:
    """arch -> sorted list of seed dir names that have BOTH a proposed_method and vanilla checkpoint."""
    root = Path(saved_model_dir)
    pairs: dict[str, list[str]] = {}
    if not root.is_dir():
        return pairs
    for arch_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        proposed_dir = arch_dir / "proposed_method"
        vanilla_dir = arch_dir / "vanilla"
        if not proposed_dir.is_dir() or not vanilla_dir.is_dir():
            continue
        proposed_seeds = {d.name for d in proposed_dir.iterdir() if d.is_dir() and _has_weights(d)}
        vanilla_seeds = {d.name for d in vanilla_dir.iterdir() if d.is_dir() and _has_weights(d)}
        common = sorted(proposed_seeds & vanilla_seeds)
        if common:
            pairs[arch_dir.name] = common
    return pairs


def load_tokenizer(arch: str):
    paths = ARCH_TOKENIZERS[arch]
    source = paths["local"] if paths["local"].is_dir() else paths["hub"]
    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    tokenizer.padding_side = "left"  # required for correct batched/causal generation
    return tokenizer


def load_pair(arch: str, seed: str, device: torch.device, dtype: torch.dtype) -> dict:
    tokenizer = load_tokenizer(arch)
    models = {}
    for variant in ("proposed_method", "vanilla"):
        checkpoint_dir = SAVED_MODEL_DIR / arch / variant / seed
        assert _has_weights(checkpoint_dir), f"no weight file found under {checkpoint_dir}"
        model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, dtype=dtype)
        model.resize_token_embeddings(len(tokenizer))  # matches training-time tokenizer size
        model.config.pad_token_id = tokenizer.pad_token_id
        model.to(device)
        model.eval()
        models[variant] = model
    return {
        "arch": arch,
        "seed": seed,
        "device": device,
        "dtype": dtype,
        "tokenizer": tokenizer,
        "models": models,
    }


def unload_loaded():
    loaded = st.session_state.get("loaded")
    if loaded is None:
        return
    for model in loaded["models"].values():
        del model
    st.session_state["loaded"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@st.cache_data(show_spinner=False)
def load_qed_validation(dataset_dir: str):
    from datasets import load_from_disk

    return load_from_disk(dataset_dir)["validation"]


@st.cache_data(show_spinner=False)
def qed_index_by_example_id(dataset_dir: str) -> dict[int, str]:
    """example_id -> paragraph_text, so a differing_case record (which has no
    paragraph_text of its own) can be turned back into a full prompt."""
    dataset = load_qed_validation(dataset_dir)
    return {int(row["example_id"]): row["paragraph_text"] for row in dataset}


@st.cache_data(show_spinner=False)
def load_differing_cases(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["prediction_difference_analysis"]["differing_cases"]


def clean_ground_truth(text: str) -> str:
    """QED ground truths are Penn-Treebank tokenized ("`` Deadman 's Gun ''",
    "20 - yard line"). Undo the spacing so it reads like normal English;
    purely cosmetic, does not touch the underlying value used for scoring."""
    if not text:
        return text
    text = re.sub(r"\s+([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+('s|n't)\b", r"\1", text)
    text = re.sub(r"``\s*", '"', text)
    text = re.sub(r"\s*''", '"', text)
    text = re.sub(r"\$\s+(?=\d)", "$", text)
    text = re.sub(r"(\d)\s+-\s+(?=\w)", r"\1-", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def build_prompt(paragraph: str, question: str) -> str:
    return f"Context: {paragraph}\nQuestion: {question}\nAnswer:"


def generate(model, tokenizer, prompt: str, device: torch.device, max_new_tokens: int,
             do_sample: bool, temperature: float, top_p: float) -> tuple[str, float, int]:
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    t0 = time.perf_counter()
    with torch.no_grad():
        output = model.generate(**encoded, **gen_kwargs)
    elapsed = time.perf_counter() - t0

    prompt_len = encoded["input_ids"].shape[1]
    new_tokens = output[:, prompt_len:]
    text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    text = text.split("\n")[0].strip()  # matches the eval/notebook convention: first line only
    return text, elapsed, new_tokens.shape[1]


st.set_page_config(page_title="SpanOT-KD Student Demo", layout="wide")
st.title("SpanOT-KD Student Demo — Proposed vs. Vanilla")

if "loaded" not in st.session_state:
    st.session_state["loaded"] = None

pairs = discover_pairs(str(SAVED_MODEL_DIR))

with st.sidebar:
    st.header("Model")

    if not pairs:
        st.error(f"No complete (proposed_method, vanilla) checkpoint pairs found under {SAVED_MODEL_DIR}")
        st.stop()

    arch = st.selectbox("Architecture", options=sorted(pairs.keys()))
    seed = st.selectbox("Seed", options=pairs[arch])

    cuda_available = torch.cuda.is_available()
    device_label = st.radio("Device", options=(["cuda", "cpu"] if cuda_available else ["cpu"]), horizontal=True)
    device = torch.device(device_label)
    use_bf16 = st.checkbox("Use bfloat16", value=False, disabled=(device_label != "cuda"))
    dtype = torch.bfloat16 if (use_bf16 and device_label == "cuda") else torch.float32

    load_col, unload_col = st.columns(2)
    load_clicked = load_col.button("Load pair", use_container_width=True)
    unload_clicked = unload_col.button("Unload", use_container_width=True)

    if unload_clicked:
        unload_loaded()
        st.success("Model unloaded.")

    if load_clicked:
        unload_loaded()  # avoid stacking checkpoints in memory/VRAM across loads
        with st.spinner(f"Loading {arch}/{{proposed_method,vanilla}}/{seed} onto {device_label}..."):
            st.session_state["loaded"] = load_pair(arch, seed, device, dtype)
        st.success(f"Loaded {arch} — seed {seed}")

    loaded = st.session_state["loaded"]
    st.divider()
    if loaded is None:
        st.info("No model loaded.")
    else:
        n_params = sum(p.numel() for p in loaded["models"]["vanilla"].parameters()) / 1e6
        st.markdown(
            f"**Loaded:** `{loaded['arch']}` / seed `{loaded['seed']}`\n\n"
            f"Device: `{loaded['device']}` · dtype: `{loaded['dtype']}` · "
            f"{n_params:.1f}M params per model"
        )

st.subheader("Prompt")
input_mode = st.radio("Input mode", ["QED-style (context + question)", "Raw prompt"], horizontal=True)

if "paragraph" not in st.session_state:
    st.session_state["paragraph"] = ""
if "question" not in st.session_state:
    st.session_state["question"] = ""
if "archived_predictions" not in st.session_state:
    st.session_state["archived_predictions"] = None

if input_mode == "QED-style (context + question)":
    example_sources = ["Manual entry"]
    if QED_DATASET_DIR.is_dir():
        example_sources.append("Random QED validation example")
    if DIFFERING_CASES_PATH.is_file():
        example_sources.append("Proposed ≠ Vanilla case (109 known cases)")

    example_source = st.radio("Fill from", example_sources, horizontal=True)

    if example_source == "Random QED validation example" and st.button("Load random example"):
        example = random.choice(load_qed_validation(str(QED_DATASET_DIR)))
        st.session_state["paragraph"] = example["paragraph_text"]
        st.session_state["question"] = example["question"]
        answers = example.get("answers")
        st.session_state["ground_truth"] = answers[0] if isinstance(answers, list) and answers else str(answers)
        st.session_state["archived_predictions"] = None

    elif example_source == "Proposed ≠ Vanilla case (109 known cases)":
        cases = load_differing_cases(str(DIFFERING_CASES_PATH))
        st.caption(
            f"{len(cases)} QED validation examples where the OPT proposed_method and vanilla "
            "checkpoints (seed 26) produced different answers, from `compare_outputs/analysis_results.json`."
        )
        labels = [f"{i}. {c['question']}" for i, c in enumerate(cases)]
        picked = st.selectbox("Case", options=range(len(cases)), format_func=lambda i: labels[i])
        case = cases[picked]
        if QED_DATASET_DIR.is_dir():
            paragraph_lookup = qed_index_by_example_id(str(QED_DATASET_DIR))
            st.session_state["paragraph"] = paragraph_lookup.get(case["example_id"], "")
        st.session_state["question"] = case["question"]
        st.session_state["ground_truth"] = case["ground_truth"]
        st.session_state["archived_predictions"] = {
            "proposed_method": case["proposed_method_prediction"],
            "vanilla": case["vanilla_prediction"],
        }

    paragraph = st.text_area("Context", value=st.session_state["paragraph"], height=150)
    question = st.text_input("Question", value=st.session_state["question"])
    prompt = build_prompt(paragraph, question)

    if st.session_state.get("ground_truth"):
        st.success(f"**Ground truth:** {clean_ground_truth(st.session_state['ground_truth'])}")

    archived = st.session_state.get("archived_predictions")
    if archived is not None:
        with st.expander("Previously recorded predictions (OPT full-validation eval run)"):
            st.caption(
                "From the earlier 1347-example batch run, not regenerated live here — "
                "batching/precision can shift greedy output slightly, so this may not match "
                "the live 'Generate' result exactly."
            )
            st.markdown(f"**Proposed method:** {archived['proposed_method']}")
            st.markdown(f"**Vanilla:** {archived['vanilla']}")
else:
    prompt = st.text_area("Prompt", height=150)

with st.expander("Generation settings"):
    max_new_tokens = st.slider("Max new tokens", min_value=8, max_value=300, value=150, step=8)
    do_sample = st.checkbox("Sample (unchecked = greedy)", value=False)
    temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1, disabled=not do_sample)
    top_p = st.slider("Top-p", min_value=0.1, max_value=1.0, value=0.95, step=0.05, disabled=not do_sample)

st.divider()
generate_clicked = st.button("Generate", type="primary", disabled=(st.session_state["loaded"] is None))

if st.session_state["loaded"] is None:
    st.warning("Load a checkpoint pair from the sidebar before generating.")
elif generate_clicked:
    if not prompt.strip():
        st.error("Prompt is empty.")
    else:
        loaded = st.session_state["loaded"]
        tokenizer = loaded["tokenizer"]
        col_proposed, col_vanilla = st.columns(2)
        for col, variant, label in (
            (col_proposed, "proposed_method", "Proposed method (SpanOT-KD)"),
            (col_vanilla, "vanilla", "Vanilla (MultiLevelOT)"),
        ):
            with col:
                st.markdown(f"### {label}")
                with st.spinner(f"Generating ({label})..."):
                    text, elapsed, n_tokens = generate(
                        loaded["models"][variant], tokenizer, prompt, loaded["device"],
                        max_new_tokens, do_sample, temperature, top_p,
                    )
                st.write(text if text else "_(empty output)_")
                st.caption(f"{elapsed:.3f}s · {n_tokens} tokens generated")
