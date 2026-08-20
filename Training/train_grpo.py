"""
GRPO (RLVR) sur le modele Direct-FT, en LoRA. Reward GRADUEE.
Point de depart = modele SFT-Direct ; la reference KL = ce meme modele SFT.


Deps :  pip install "trl>=0.14" peft transformers accelerate datasets bitsandbytes
Lancer :  python Training/train_grpo.py
"""
import os
import sys
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig, PeftModel
from trl import GRPOTrainer, GRPOConfig
from transformers.trainer_utils import get_last_checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "Solver"))
from eval_lib import parse_final, grade, BOXES   # parsing + notation oracle

# ==================== HYPERPARAMETRES ====================
BASE_ID          = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_ADAPTER      = os.path.join(HERE, "sft_direct_2_epochs_8_lorar32")   # init + reference KL
# Dataset de prompts GRPO : puzzles GENERES, DISJOINTS du SFT-train ET des sets d'eval.
# Format attendu par ligne : {"prompt": [{"role":"user","content": <prompt complet>}],
#                             "gem": "WHITE", "box_truths": {"BLUE":[...], ...}}
GRPO_DATA        = os.path.join(HERE, "..", "Dataset", "grpo_prompts.jsonl")
OUTPUT_DIR       = os.path.join(HERE, "grpo_sft_direct_v2_lr1e5")


# --- reward gradue ---
W_GEM            = 1.0     # poids de la gemme correcte
W_BITS           = .8     # poids de la fraction de bits de verite justes

# --- GRPO ---
NUM_GENERATIONS    = 8      # G : taille du groupe (qualite de l'estimation d'avantage)
BETA_KL            = 0.04   # coefficient KL vers la reference SFT (exploration <-> collapse)
LEARNING_RATE      = 1e-5
MAX_PROMPT_LEN     = 768
MAX_COMPLETION_LEN = 128    # Longueur de sortie pour Direct-FT
PER_DEVICE_BATCH   = 2
GRAD_ACCUM         = 4
MAX_STEPS          = 300
TEMPERATURE        = 0.7
LORA_R             = 32
LORA_ALPHA         = 64
LORA_DROPOUT       = 0.05
LORA_TARGET        = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SEED               = 42
# ===================================================================


def _completion_text(comp):
    # [VERSION] trl passe soit une str, soit une liste de messages selon le mode.
    if isinstance(comp, str):
        return comp
    return comp[-1]["content"]


def _fraction_bits_correct(parsed_truths, gold_box_truths):
    tot = ok = 0
    if not parsed_truths:
        return 0.0
    for b in BOXES:
        got = parsed_truths.get(b) or []
        for i, ev in enumerate(gold_box_truths[b]):
            tot += 1
            if i < len(got) and got[i] == ev:
                ok += 1
    return ok / tot if tot else 0.0


def reward_graded(completions, gem=None, box_truths=None, **kwargs):
    """Reward GRADUEE — signature trl : recoit les completions + les colonnes du dataset.
    Retourne une liste de floats (une reward par completion).
    ---> C'EST ICI TON DESIGN. La formule ci-dessous est un point de depart, pas un choix. <---
    """
    rewards = []
    for i, comp in enumerate(completions):
        text = _completion_text(comp)
        gold = {"gem": gem[i], "box_truths": box_truths[i]}
        g = grade(parse_final(text), gold)

        if not g["parse_ok"]:
            r = 0.0
        else:
            # ---------- TON DESIGN : formule graduee ----------
            gem_term = 1.0 if g["gem_correct"] else 0.0
            frac_bits = _fraction_bits_correct(g["parsed_truths"], gold["box_truths"])
            if frac_bits == 1 and gem_term == 1:
                r = 1.0 + W_GEM * gem_term + W_BITS * frac_bits
            else:
                r = W_GEM * gem_term + W_BITS * frac_bits
            # (idees a explorer : bonus d'unicite gem+bits (strict), penalite si bits sans gemme,
            #  normalisation, etc.)
            # --------------------------------------------------
        rewards.append(float(r))
    return rewards


def main():
    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Modele de depart = base + adaptateur SFT, FUSIONNE.
    #    -> les poids SFT deviennent l'init de la policy ET la reference KL (adaptateurs desactives).
    model = AutoModelForCausalLM.from_pretrained(
        BASE_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(model, SFT_ADAPTER)
    model = model.merge_and_unload()
    model.config.use_cache = False
    # [MEMOIRE] Si OOM sur 8 Go : envisage un chargement 4-bit du modele fusionne
    #           (re-quantisation), et/ou baisse NUM_GENERATIONS / PER_DEVICE_BATCH / MAX_COMPLETION_LEN.

    # 2. Nouvelle LoRA pour le RL (trl l'applique ; la reference KL = modele adaptateurs-off)
    peft_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET, bias="none", task_type="CAUSAL_LM",
    )

    # 3. Dataset de prompts (colonnes prompt / gem / box_truths pour la reward)
    ds = load_dataset("json", data_files={"train": GRPO_DATA})["train"]

    # 4. Config GRPO
    cfg = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_generations=NUM_GENERATIONS,
        beta=BETA_KL,                          # [VERSION] coef KL (nom peut varier)
        learning_rate=LEARNING_RATE,
        max_completion_length=MAX_COMPLETION_LEN,
        temperature=TEMPERATURE,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=MAX_STEPS,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        seed=SEED,
        report_to="none",
        # use_vllm=False,  # [MEMOIRE] generation HF native ; vLLM co-localise est tendu sur 8 Go
    )

    # 5. Trainer
    trainer = GRPOTrainer(
        model=model,
        args=cfg,
        reward_funcs=[reward_graded],          # [VERSION] certaines versions : reward_funcs=reward_graded
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=tokenizer,            # [VERSION] anciennes : tokenizer=tokenizer
    )

    last_ckpt = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
    if last_ckpt:
        print(f"[RESUME] reprise depuis {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[OK] Adaptateur GRPO sauvegarde dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
