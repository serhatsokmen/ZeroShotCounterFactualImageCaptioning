# Zero-Shot Counterfactual Image Captioning & Hallucination Mitigation

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-orange)](https://huggingface.co/)
[![OpenAI CLIP](https://img.shields.io/badge/OpenAI-CLIP-brightgreen)](https://github.com/openai/CLIP)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A training-free, zero-shot hallucination mitigation framework for Vision-Language Models (VLMs). This repository implements an inference-time optimization pipeline that couples **Salesforce BLIP** (generative captioner) with **OpenAI CLIP ViT-B/32** (multimodal semantic evaluator) via an iterative counterfactual feedback loop. The architecture identifies, prunes, and systematically re-generates ungrounded tokens without modifying underlying neural weights or utilizing external supervision data.

---

## Key Highlights

- **Training-Free & Zero-Shot:** Operates strictly at inference time. No backpropagation, fine-tuning, domain adaptation, or ground-truth annotation corpora are required.
- **Counterfactual Pruning:** Isolates and truncates ungrounded entity tails across variable token depths ($k \in \{1, 2, 3\}$), avoiding full sentence re-generation from scratch.
- **Logit-Level Constrained Generation:** Suppresses generic conjunctions, auxiliary verbs, and punctuation via `bad_words_ids` to eliminate the "comma trap" and force visually grounded entity synthesis.
- **Confidence Guardrail ($\tau = 0.32$):** Bypasses stable predictions to prevent over-correction and semantic drift on already grounded baseline descriptions.

---

## Architectural Workflow

The framework operates via an iterative hypothesis-testing loop between a generative vision-language model and a contrastive multimodal verifier:

```
[ Input Image ]
       │
       ▼
┌──────────────┐
│  BLIP Base   │ ────> Initial Caption (C0)
└──────────────┘
       │
       ▼
┌──────────────┐
│ CLIP ViT-B32 │ ────> Cosine Similarity Score (S0)
└──────────────┘
       │
       ├─── IF S0 >= tau (0.32) ─────────────────────────> [ Final Grounded Caption ]
       │                                                   (Negative Control / Pass)
       └─── IF S0 < tau (Hallucination Suspected)
               │
               ▼
       ┌────────────────────────────────────────────────────────┐
       │ Counterfactual Search Loop (Max Steps: 3)              │
       │                                                        │
       │ 1. Truncate trailing tokens at depth k in {1, 2, 3}    │
       │ 2. Condition BLIP on safe prefix text                  │
       │ 3. Logit Filtering: Mask BANNED_TOKENS                 │
       │ 4. Generate Top-5 candidates via Nucleus Sampling      │
       │ 5. Rescore candidates with CLIP                        │
       │ 6. Select argmax(S_cand) or trigger Fallback           │
       └────────────────────────────────────────────────────────┘
               │
               ▼
       [ Refined Grounded Caption ]
```

---

## Methodology & Mechanical Details

### 1. Contrastive Verification & Empirical Gating ($\tau = 0.32$)
Given visual representation $I$ and candidate caption $T$, cosine similarity is evaluated in the normalized joint multimodal embedding space:

$$\text{Sim}(I, T) = \frac{\Phi_I(I) \cdot \Phi_T(T)}{\Vert{}\Phi_I(I)\Vert{}_2 \Vert{}\Phi_T(T)\Vert{}_2}$$

- **Empirical Boundary Choice:** Contrastive embeddings in OpenAI CLIP ViT-B/32 compress natural image-text pairs into a narrow operational cosine similarity range ($[0.15, 0.40]$). Ungrounded descriptions and hallucinated tail tokens systematically cluster below $0.32$ (typically between $0.25$ and $0.31$), whereas semantically grounded descriptions yield scores $\ge 0.32$.
- **Boundary Trade-off:** 
  - Selecting $\tau > 0.35$ enforces an overly aggressive threshold, triggering unnecessary perturbations on correct captions (over-correction / semantic drift).
  - Selecting $\tau < 0.28$ blinds the evaluation mechanism to ungrounded entities.
  - $\tau = 0.32$ serves as the empirical equilibrium boundary separating grounded output from ungrounded generations.

### 2. Logit-Level Constrained Generation
Generative models frequently attempt to maximize cross-modal similarity by appending high-frequency generic terms or punctuation (e.g., `, and`, `with`, `is`). To enforce genuine entity alignment, the decoding pipeline penalizes and blocks banned tokens directly during beam/nucleus exploration via `bad_words_ids`:

- **Structural Punctuation:** `,`, `.`, `!`, `?`, `;`, `-`, `_`
- **Generic Prepositions & Conjunctions:** `and`, `the`, `a`, `an`, `of`, `with`, `in`, `on`, `at`, `by`, `for`, `to`, `from`
- **Auxiliary Verbs:** `is`, `are`, `was`, `were`, `be`, `been`, `being`

### 3. Multi-Depth Counterfactual Pruning & Fallback
The engine systematically investigates counterfactual prefix states by peeling back trailing tokens at variable depths:

$$\text{Prefix}_k = \text{Tokenize}(C)[:-k], \quad k \in \{1, 2, 3\}$$

If candidate branches fail to exceed the current best score across all pruning depths, the engine falls back to the safe truncation base ($\text{words}[:-2]$), terminating further drift.

---

## Empirical Evaluation & Results

Visual inspection and comparative analysis generated via `outputs/` reports:

### Case 1: Substantial Semantic Realignment (+22.0% Gain)
The generator originally hallucinated context not supported by the visual features. Pruning the ungrounded tail sequence allowed the model to resynthesize the scene with high multimodal alignment.

![Result Case 1](outputs/result_deneme3.jpg)

---

### Case 2: Multi-Step Refinement (+17.3% Gain)
Demonstrates the iterative multi-depth exploration where intermediate tokens are evaluated and replaced under strict lexical constraints.

![Result Case 2](outputs/result_demo3.jpg)

---

### Case 3: Qualitative Grounding & Object Correction
While the quantitative score increase is modest, the counterfactual engine successfully substitutes an incorrectly recognized background entity with its true visual counterpart.

![Result Case 3](outputs/result_zzz4.jpg)

---

### Case 4: Negative Control (Stability Baseline)
The initial caption generated by BLIP already exceeded the confidence threshold ($\tau = 0.32$). The guardrail mechanism preserved the original text without triggering needless perturbations.

![Result Case 4](outputs/result_zzz6.jpg)

---

## Repository Structure

```
ZeroShotCounterFactualImageCaptioning/
├── images/                 # Input test images (.jpg, .jpeg, .png)
├── outputs/                # Visual verification and inspection reports
├── src/
│   ├── __init__.py         # Package declaration
│   ├── config.py           # Hyperparameters, paths, and banned tokens
│   ├── engine.py           # Counterfactual search loop and pruning logic
│   ├── models.py           # BLIP and CLIP model architectures & sampling
│   ├── utils.py            # Image loader and visualization engine
│   └── main.py             # Batch pipeline driver
├── run.py                  # Project entry point
├── requirements.txt        # Production dependencies
├── .gitignore              # Environment and cache ignore rules
└── README.md               # Technical documentation
```

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/serhatsokmen/ZeroShotCounterFactualImageCaptioning.git
cd ZeroShotCounterFactualImageCaptioning
```

### 2. Environment Setup
Recommended: Python 3.9+ with CUDA-enabled PyTorch.

```bash
# Create virtual environment
python -m venv myenv

# Activate environment
# On Windows:
myenv\Scripts\activate
# On Linux/macOS:
source myenv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage & Experiments

### Run Batch Processing
1. Place target images (`.jpg`, `.jpeg`, `.png`) inside the `images/` directory.
2. Execute the pipeline from the project root:

```bash
python run.py
```

3. Comparative visual reports (original vs. refined captions, scores, and percentage changes) will be automatically generated and saved to `outputs/result_<image_name>.jpg`.

### Experimenting with Custom Inputs
To evaluate a single custom image, drop it into the `images/` directory and execute `python run.py`. The system processes all images within the folder sequentially.

---

## Configuration & Hyperparameter Tuning

All pipeline parameters can be modified in `src/config.py`:

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `HALUCINATION_THRESHOLD` | `0.32` | Confidence boundary ($\tau$) below which refinement triggers. |
| `MAX_REFINEMENT_STEPS` | `3` | Maximum iterative depth steps per sentence. |
| `BLIP_MODEL_NAME` | `"Salesforce/blip-image-captioning-base"` | Autoregressive vision-language backbone. |
| `CLIP_MODEL_NAME` | `"ViT-B/32"` | Multimodal scoring and alignment backbone. |
| `BANNED_TOKENS` | `[...]` | Punctuation, generic conjunctions, and auxiliaries blocked at logit level. |

---

## Citation & References

If you utilize this pipeline or build upon this architecture, please refer to the foundational architectures:

```bibtex
@inproceedings{li2022blip,
  title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation},
  author={Junnan Li and Dongxu Li and Caiming Xiong and Steven Hoi},
  booktitle={ICML},
  year={2022}
}

@inproceedings{radford2021learning,
  title={Learning Transferable Visual Models From Natural Language Supervision},
  author={Alec Radford and Jong Wook Kim and Chris Hallacy and Aditya Ramesh and Gabriel Goh and Sandhini Agarwal and Girish Sastry and Amanda Askell and Pamela Mishkin and Jack Clark and Gretchen Krueger and Ilya Sutskever},
  booktitle={ICML},
  year={2021}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.