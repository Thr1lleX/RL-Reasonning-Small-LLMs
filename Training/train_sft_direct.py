
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from transformers.trainer_utils import get_last_checkpoint

# ==================== HYPERPARAMETRES ====================
MODEL_ID          = "Qwen/Qwen2.5-1.5B-Instruct"
USE_QLORA         = True     # True: 4-bit (bitsandbytes) ; False: LoRA bf16 (fallback Windows)

LORA_R            = 32
LORA_ALPHA        = 64
LORA_DROPOUT      = 0.05
LORA_TARGET       = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj" ]

LEARNING_RATE     = 2e-4
NUM_EPOCHS        = 8
PER_DEVICE_BATCH  = 4
GRAD_ACCUM        = 4         # batch effectif = PER_DEVICE_BATCH * GRAD_ACCUM
MAX_SEQ_LEN       = 1024
WARMUP_RATIO      = 0.03
WARMUP_STEPS      = 30 # 2100 exemples, batch effectif de 16 = environ 262 pas sur 2 époques), un warmup de ~3 % correspond à environ 10 pas. 
LR_SCHEDULER      = "cosine"
SEED              = 42
# ===================================================================

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "Dataset", "ft_direct")
OUTPUT_DIR = os.path.join(HERE, "sft_direct_2_epochs_8_lorar32")


def main():
    set_seed(SEED)

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Modele (qLoRA 4-bit ou LoRA bf16)
    if USE_QLORA:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb, device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16,
        )
    model.config.use_cache = False  # requis avec gradient checkpointing

    # 3. Config LoRA
    peft_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET, bias="none", task_type="CAUSAL_LM",
    )

    # 4. Datasets conversationnels (champ "messages")
    ds = load_dataset("json", data_files={
        "train": os.path.join(DATA_DIR, "train.jsonl"),
        "val":   os.path.join(DATA_DIR, "val.jsonl"),
    })

    


    # 7. Config d'entrainement
    cfg = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=WARMUP_STEPS,
        max_length=MAX_SEQ_LEN,
        completion_only_loss=True,           # Masque automatiquement le prompt et n'apprend que la réponse assistant
        packing=False,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        seed=SEED,
        report_to="none",
    )

    # 8. Trainer
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    last_ckpt = get_last_checkpoint(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else None
    if last_ckpt:
        print(f"[RESUME] reprise depuis {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)   # None = frais, chemin = reprise
    trainer.save_model(OUTPUT_DIR)           # sauvegarde les adaptateurs LoRA
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"[OK] Adaptateurs LoRA sauvegardes dans {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
