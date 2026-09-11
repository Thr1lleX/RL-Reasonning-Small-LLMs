import os,json,time
from openai import OpenAI, RateLimitError, APIStatusError, APIConnectionError
from dotenv import load_dotenv
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "Solver"))
from eval_lib import parse_final, grade, BOXES



load_dotenv()  # Charge automatiquement les variables du fichier .env

deepseek_key = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=deepseek_key,
    base_url="https://api.deepseek.com"
)


INPUT_FILE = "Dataset/ft_direct/train.jsonl"
OUTPUT_FILE = "Dataset/ft_cot/train_cot.jsonl"
REJECTED_FILE     = "Dataset/ft_cot/raw_rejected_traces.jsonl"
MAX_API_RETRIES   = 5
MAX_LOGIC_RETRIES = 3
BASE_BACKOFF = 1.0
MODEL_NAME = "deepseek-v4-flash"

# creer le dir
os.makedirs("Dataset/ft_cot", exist_ok=True)
# Charger les puzzles déjà résolus et les traces d'échecs
solved_ids = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                solved_ids.add(r["id"])

rejected_counts = {}
if os.path.exists(REJECTED_FILE):
    with open(REJECTED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                pid = r.get("id")
                if pid:
                    rejected_counts[pid] = rejected_counts.get(pid, 0) + 1

# Puzzles définitivement abandonnés (ayant épuisé MAX_LOGIC_RETRIES sans succès)
abandoned_ids = {pid for pid, count in rejected_counts.items() if count >= MAX_LOGIC_RETRIES and pid not in solved_ids}

# Puzzles avec échecs partiels (session interrompue avant d'atteindre MAX_LOGIC_RETRIES)
partial_ids = {pid for pid, count in rejected_counts.items() if 0 < count < MAX_LOGIC_RETRIES and pid not in solved_ids}

print("=" * 60)
print(f"Statut au démarrage :")
print(f" - Puzzles résolus (train_cot.jsonl)         : {len(solved_ids)}")
print(f" - Puzzles abandonnés (≥ {MAX_LOGIC_RETRIES} échecs)          : {len(abandoned_ids)}")
print(f" - Puzzles en reprise (< {MAX_LOGIC_RETRIES} échecs partiels) : {len(partial_ids)}")
print(f" - Total puzzles exclus de la relance        : {len(solved_ids | abandoned_ids)}")
print("=" * 60)


# call api avec backoff

def call_api_with_backoff(prompt_text : str, puzzle_id):
    for attempt in range(MAX_API_RETRIES):
        try:
            response =  client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt_text}],
                reasoning_effort="high",
                temperature=0.7,
                extra_body={"thinking": {"type": "enabled"}}
            )
            reasoning_text = response.choices[0].message.reasoning_content or ""
            content_text = response.choices[0].message.content or ""
            return reasoning_text,content_text
        except (RateLimitError, APIStatusError, APIConnectionError) as e:
            status_code = getattr(e, "status_code", "Connexion")
            sleep_time = BASE_BACKOFF * (2 ** attempt)
            print(f"Erreur traitement puzzle {puzzle_id}: {e}, Tentative {attempt+1} / {MAX_API_RETRIES}")
            if attempt == MAX_API_RETRIES - 1:
                raise e
            time.sleep(sleep_time)
        except Exception as e:
            print(f"Erreur critique : {e}")
            time.sleep(3)
    return None, None

# charger les puzzles non traites, boucle sur les puzzles du train

with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
     open(OUTPUT_FILE, "a", encoding="utf-8") as f_out, \
     open(REJECTED_FILE, "a", encoding="utf-8") as f_rej:
    for line in f_in:
        if not line.strip():
            continue
        item = json.loads(line)
        puzzle_id = item["id"]
        
        # Ignorer si déjà validé ou déjà abandonné (3 échecs enregistrés)
        if puzzle_id in solved_ids or puzzle_id in abandoned_ids:
            continue
        
        raw_prompt = item["messages"][0]["content"]
        gold_answer = item["messages"][1]["content"]
        # Préparation de l'oracle de référence
        gold_parsed = parse_final(gold_answer)
        gold_oracle = {
            "gem": gold_parsed["gem"],
            "box_truths": {
                b: [gold_parsed["box_idx"][b].get(i) for i in range(1, len(gold_parsed["box_idx"][b]) + 1)]
                for b in BOXES
            }
        }
        # Prompt adapté
        custom_prompt = raw_prompt + "\nUse step by step reasoning, explicit, consistent with the final answer, concise but deductive."
        puzzle_solved = False
        
        # Déterminer à quel essai reprendre (ex: si interrompu au 1er échec, reprendre à l'essai 2)
        prior_failed_attempts = rejected_counts.get(puzzle_id, 0)
        start_attempt = prior_failed_attempts + 1
        
        for logic_attempt in range(start_attempt, MAX_LOGIC_RETRIES + 1):
            reasoning, content = call_api_with_backoff(custom_prompt, puzzle_id)
            if reasoning is None and content is None:
                print(f"[ECHEC RESEAU] Impossible d'obtenir une réponse pour {puzzle_id}")
                break
            # Vérification par l'Oracle
            model_parsed = parse_final(content)
            verdict = grade(model_parsed, gold_oracle)
            if verdict["correct"]:
                # Succès : Enregistrement dans train_cot.jsonl
                full_cot_assistant = content
                new_item = {
                    "id": puzzle_id,
                    "messages": [
                        {"role": "user", "content": raw_prompt},
                        {"role": "assistant", "content": full_cot_assistant}
                    ],
                    "metadata": item.get("metadata", {}),
                }
                f_out.write(json.dumps(new_item, ensure_ascii=False) + "\n")
                f_out.flush()
                solved_ids.add(puzzle_id)
                print(f"[VALIDE] {puzzle_id} (Essai {logic_attempt}/{MAX_LOGIC_RETRIES})")
                puzzle_solved = True
                break
            else:
                # Échec logique : Logger la trace brute dans les rejets
                rejected_log = {
                    "id": puzzle_id,
                    "logic_attempt": logic_attempt,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "verdict": verdict,
                    "reasoning": reasoning,
                    "content": content,
                }
                f_rej.write(json.dumps(rejected_log, ensure_ascii=False) + "\n")
                f_rej.flush()
                rejected_counts[puzzle_id] = rejected_counts.get(puzzle_id, 0) + 1
                print(f"  [REJET] {puzzle_id} (Essai {logic_attempt}/{MAX_LOGIC_RETRIES}) - Parsed={verdict['parse_ok']}, GemCorrect={verdict['gem_correct']}, Strict={verdict['correct']}")
                time.sleep(0.5)
        
        if not puzzle_solved and rejected_counts.get(puzzle_id, 0) >= MAX_LOGIC_RETRIES:
            abandoned_ids.add(puzzle_id)
            print(f"[ABANDON] {puzzle_id} non résolu après {MAX_LOGIC_RETRIES} essais.")
        
        time.sleep(0.2)  # Pause légère inter-puzzles