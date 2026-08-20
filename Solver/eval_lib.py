"""
Bibliotheque pure (sans torch) pour l'evaluation des LLM sur les puzzles Parlor.
Contient : extraction du texte par statement, derivation de l'oracle (GOLD),
construction du prompt, parsing de la reponse, notation stricte (Option B).

Testable sans GPU. run_eval.py importe ce module et n'ajoute que la partie modele.
"""
import os
import re
import csv
import sys

sys.path.append(os.path.dirname(__file__))

from models import BOXES  # ["BLUE", "WHITE", "BLACK"]
from evaluate_batch import PUZZLES_REGISTER
from solver import generate_all_worlds, is_world_valid

ROOT = os.path.dirname(os.path.dirname(__file__))
ORIGINAL_CSV = os.path.join(ROOT, "Parlor puzzles.csv")

BOX_LABEL = {"BLUE": "BLUE (left)", "WHITE": "WHITE (middle)", "BLACK": "BLACK (right)"}
BOX_PREFIX = {"BLUE": "B", "WHITE": "W", "BLACK": "K"}

# --------------------------------------------------------------------------
# 1. Extraction du texte par statement depuis le CSV d'origine
#    (le v3 avait fusionne les multi-statements : on re-parse la source)
# --------------------------------------------------------------------------

def _clean_text(s: str) -> str:
    s = re.sub(r"<br>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<color=[^>]*>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"</color>", "", s, flags=re.IGNORECASE)
    s = re.sub(r"</?b>", "", s, flags=re.IGNORECASE)
    s = (s.replace("‘", "'").replace("’", "'")
           .replace("“", '"').replace("”", '"'))
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) >= 2 and s[0] == "'" and s[-1] == "'":
        s = s[1:-1].strip()
    return s


def _split_statements(cell: str):
    """Decoupe une cellule de boite en statements (separateur = double <br>)."""
    if cell is None:
        return []
    cell = cell.strip()
    if len(cell) >= 2 and cell[0] == "'" and cell[-1] == "'":   # enrobage cellule
        cell = cell[1:-1]
    parts = re.split(r"(?:<br>\s*){2,}", cell, flags=re.IGNORECASE)
    return [t for t in (_clean_text(p) for p in parts) if t]


def load_statement_texts():
    """id -> {box: [texte_stmt0, texte_stmt1, ...]}"""
    texts = {}
    with open(ORIGINAL_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)  # header
        pid = 0
        for rec in reader:
            if not rec or all(c.strip() == "" for c in rec):
                continue
            pid += 1
            texts[pid] = {
                "BLUE": _split_statements(rec[1] if len(rec) > 1 else ""),
                "WHITE": _split_statements(rec[2] if len(rec) > 2 else ""),
                "BLACK": _split_statements(rec[3] if len(rec) > 3 else ""),
            }
    return texts


# --------------------------------------------------------------------------
# 2. Oracle : ensemble GOLD (monde unique) + verite terrain
# --------------------------------------------------------------------------

def build_oracle():
    """id -> {'gem': box, 'box_truths': {box: [bool,...]}} pour les seuls GOLD."""
    oracle = {}
    for pid, puzzle in PUZZLES_REGISTER.items():
        worlds = [w for w in generate_all_worlds(puzzle) if is_world_valid(puzzle, w)]
        if len(worlds) == 1:
            w = worlds[0]
            oracle[pid] = {
                "gem": w.gem_box,
                "box_truths": {b: list(w.box_truths[b]) for b in BOXES},
            }
    return oracle


def statement_counts(pid):
    """Nombre de statements formalises par boite pour un puzzle."""
    puzzle = PUZZLES_REGISTER[pid]
    return {b: len(puzzle.box_statements[b]) for b in BOXES}


# --------------------------------------------------------------------------
# 3. Construction du prompt (zero-shot CoT). La section REGLES est load-bearing
#    (invariant Blue Prince) -> a relire par l'utilisateur.
# --------------------------------------------------------------------------

RULES = """You are solving a logic puzzle from the game "Blue Prince".

There are three boxes: BLUE (left), WHITE (middle), BLACK (right). Each box displays one or more statements.

The rules that make the puzzle uniquely solvable:
- EXACTLY ONE of the three boxes contains the gems; the other two are empty.
- AT LEAST ONE box has ALL of its statements TRUE.
- AT LEAST ONE box has ALL of its statements FALSE.
- The remaining box may be all-true, all-false, or a mix.
- A statement's truth value is determined by the actual state of the world (gem location and the truth values of every statement, including self- and cross-references).

Your task: by logical deduction, determine (a) which box contains the gems, and (b) the truth value of EVERY statement."""

def statement_ids(texts_pid):
    """Liste des identifiants de statement d'un puzzle, ex: ['B1','W1','W2','K1']."""
    ids = []
    for b in BOXES:
        for i in range(1, len(texts_pid[b]) + 1):
            ids.append(f"{BOX_PREFIX[b]}{i}")
    return ids


def build_prompt(pid, texts):
    tp = texts[pid]
    lines = [RULES, "", "Here is the puzzle:", ""]
    for b in BOXES:
        stmts = tp[b]
        lines.append(f"{BOX_LABEL[b]} box:")
        if not stmts:
            lines.append("  (no statement)")
        else:
            for i, s in enumerate(stmts, 1):
                lines.append(f"  [{BOX_PREFIX[b]}{i}] {s}")
        lines.append("")

    ids = statement_ids(tp)
    counts = ", ".join(f"{BOX_LABEL[b].split()[0]}={len(tp[b])}" for b in BOXES)
    id_template = "; ".join(f"{sid}=<T/F>" for sid in ids)
    example = "; ".join(f"{sid}=T" for sid in ids)
    lines.append(
        "Think step by step. Then, on the VERY LAST line, output your final answer "
        "in EXACTLY this one-line format (nothing after it):\n\n"
        f"FINAL: gem=<BLUE|WHITE|BLACK>; {id_template}\n\n"
        f"Label EVERY statement id, and ONLY these: {', '.join(ids)} "
        f"(statement counts: {counts}). One T or F per id.\n"
        f"Example (format only): FINAL: gem=WHITE; {example}"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 4. Parsing de la reponse du modele (derniere ligne FINAL:)
# --------------------------------------------------------------------------

def _tok_to_bool(tok):
    t = tok.strip().upper()
    if t in ("T", "TRUE", "1"):
        return True
    if t in ("F", "FALSE", "0"):
        return False
    return None


PREFIX_BOX = {"B": "BLUE", "W": "WHITE", "K": "BLACK"}
STMT_ID_RE = re.compile(r"^([BWK])(\d+)$", re.IGNORECASE)


def _parse_bool_run(v):
    """'T,F' / 'T, F' / 'TF' / 'true,false' -> [bool,...] ; None si un token invalide."""
    v = v.strip()
    if v == "":
        return []
    if "," in v:
        toks = [t for t in (x.strip() for x in v.split(",")) if t != ""]
    else:
        up = v.upper()
        if up in ("T", "F", "TRUE", "FALSE", "1", "0"):
            toks = [v]
        elif all(c in "TF10" for c in up):   # run compact "FF" -> ["F","F"]
            toks = list(v)
        else:
            toks = [v]
    out = []
    for t in toks:
        b = _tok_to_bool(t)
        if b is None:
            return None
        out.append(b)
    return out


def parse_final(text):
    """Parser tolerant. Retourne {'gem': box, 'box_idx': {box: {index: bool}}}
    ou None si aucune reponse exploitable. Accepte le format par identifiant
    (B1=T; W1=F), le format positionnel (BLUE=T,F) et les runs sans virgule."""
    if not text:
        return None
    matches = list(re.finditer(r"FINAL\s*:(.*)", text, flags=re.IGNORECASE))
    if not matches:
        return None
    line = matches[-1].group(1).strip()
    if line:
        line = line.splitlines()[0]
    else:
        remainder = text[matches[-1].end():].splitlines()
        non_empty = [l.strip() for l in remainder if l.strip()]
        if not non_empty:
            return None
        line = non_empty[0]

    gem = None
    box_idx = {b: {} for b in BOXES}
    for seg in line.split(";"):
        seg = seg.strip()
        if "=" not in seg:
            continue
        k, v = seg.split("=", 1)
        k, v = k.strip(), v.strip()
        ku = k.upper()
        if ku in ("GEM", "COIN", "PRIZE", "REWARD"):
            if v.upper() in BOXES:
                gem = v.upper()
        elif ku in BOXES:                      # positionnel : BLUE=T,F
            run = _parse_bool_run(v)
            if run is not None:
                for i, b in enumerate(run, 1):
                    box_idx[ku][i] = b
        else:                                   # par identifiant : B1=T
            m = STMT_ID_RE.match(k)
            if m:
                box = PREFIX_BOX[m.group(1).upper()]
                bb = _tok_to_bool(v)
                if bb is not None:
                    box_idx[box][int(m.group(2))] = bb
    if gem is None:
        return None
    return {"gem": gem, "box_idx": box_idx}


# --------------------------------------------------------------------------
# 5. Notation stricte (Option B) : correct <=> gem ET toutes les verites matchent
# --------------------------------------------------------------------------

def grade(parsed, oracle_entry):
    if parsed is None:
        return {"parse_ok": False, "gem": None, "gem_correct": False,
                "truths_correct": False, "count_ok": False, "correct": False,
                "parsed_truths": None}
    gem = parsed["gem"]
    gem_correct = (gem == oracle_entry["gem"])
    truths_correct = True
    count_ok = True
    recon = {}
    for b in BOXES:
        n = len(oracle_entry["box_truths"][b])
        idx_map = parsed["box_idx"][b]
        got = [idx_map.get(i) for i in range(1, n + 1)]   # None si manquant
        recon[b] = got
        if got != oracle_entry["box_truths"][b]:
            truths_correct = False
        if set(idx_map.keys()) != set(range(1, n + 1)):   # ni manquant ni en trop
            count_ok = False
    return {
        "parse_ok": True,
        "gem": gem,
        "gem_correct": gem_correct,
        "truths_correct": truths_correct,
        "count_ok": count_ok,
        "correct": gem_correct and truths_correct,
        "parsed_truths": recon,
    }


# --------------------------------------------------------------------------
# 6. Verification d'alignement texte <-> formalisation (garde-fou)
# --------------------------------------------------------------------------

def check_alignment(oracle, texts):
    """Signale les GOLD ou le nb de statements texte != nb formalise."""
    mismatches = []
    for pid in sorted(oracle):
        counts_formal = statement_counts(pid)
        for b in BOXES:
            n_text = len(texts[pid][b])
            n_formal = counts_formal[b]
            if n_text != n_formal:
                mismatches.append((pid, b, n_text, n_formal))
    return mismatches


def render_prompt(tp: dict) -> str:
    """tp = {box: [texte_stmt, ...]} -> prompt complet (RULES + puzzle + format)."""
    lines = [RULES, "", "Here is the puzzle:", ""]
    for b in BOXES:
        stmts = tp[b]
        lines.append(f"{BOX_LABEL[b]} box:")
        if not stmts:
            lines.append("  (no statement)")
        else:
            for i, s in enumerate(stmts, 1):
                lines.append(f"  [{BOX_PREFIX[b]}{i}] {s}")
        lines.append("")
    ids = statement_ids(tp)
    counts = ", ".join(f"{BOX_LABEL[b].split()[0]}={len(tp[b])}" for b in BOXES)
    id_template = "; ".join(f"{sid}=<T/F>" for sid in ids)
    example = "; ".join(f"{sid}=T" for sid in ids)
    lines.append(
        "Think step by step. Then, on the VERY LAST line, output your final answer "
        "in EXACTLY this one-line format (nothing after it):\n\n"
        f"FINAL: gem=<BLUE|WHITE|BLACK>; {id_template}\n\n"
        f"Label EVERY statement id, and ONLY these: {', '.join(ids)} "
        f"(statement counts: {counts}). One T or F per id.\n"
        f"Example (format only): FINAL: gem=WHITE; {example}"
    )
    return "\n".join(lines)


def build_prompt(pid, texts):
    return render_prompt(texts[pid])



if __name__ == "__main__":
    # Auto-test a sec de la partie pure
    texts = load_statement_texts()
    oracle = build_oracle()
    print(f"Puzzles formalises      : {len(PUZZLES_REGISTER)}")
    print(f"Puzzles GOLD (oracle)   : {len(oracle)}")
    mm = check_alignment(oracle, texts)
    if mm:
        print(f"\n[!] {len(mm)} desalignements texte/formalisation sur les GOLD :")
        for pid, b, nt, nf in mm:
            print(f"    id {pid} boite {b}: {nt} statements en texte vs {nf} formalises")
    else:
        print("Alignement texte/formalisation : OK sur tous les GOLD")

    ex = sorted(oracle)[0]
    print("\n--- Exemple de prompt (id", ex, ") ---")
    print(build_prompt(ex, texts))
    print("\n--- Oracle attendu ---")
    print(oracle[ex])
