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
# charger les puzzles deja traites
done_ids = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                done_ids.add(r["id"])

print(f"{len(done_ids)} puzzles deja traites.")


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
        if puzzle_id in done_ids:
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
        for logic_attempt in range(1, MAX_LOGIC_RETRIES + 1):
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
                done_ids.add(puzzle_id)
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
                print(f"  [REJET] {puzzle_id} (Essai {logic_attempt}/{MAX_LOGIC_RETRIES}) - Parsed={verdict['parse_ok']}, GemCorrect={verdict['gem_correct']}, Strict={verdict['correct']}")
                time.sleep(0.5)
        if not puzzle_solved:
            print(f"[ABANDON] {puzzle_id} non résolu après {MAX_LOGIC_RETRIES} essais.")
        time.sleep(0.2)  # Pause légère inter-puzzles