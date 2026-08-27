"""
Query rewriting: uses an LLM to rewrite a ruling's question (everyday language) closer to
the legal language used in the VAT Law's article text. Goal: bridge the everyday-language
vs. legal-language asymmetry at first-stage retrieval (BM25/dense) — see the query
rewriting finding in the project notes.

Model  : a local vLLM server (default Qwen/Qwen3-8B, served on port 8002 via
         docker/serve_qwen3_instruct_vllm.sh) — no external API is used anywhere in this
         repo, everything runs on local GPU.
Source : train.jsonl + test.jsonl (bm25_baseline.load_test(include_train=True)) —
         filter_valid_queries is NOT applied here; boilerplate/out-of-corpus filtering is
         already done separately in downstream scripts, so the rewrite cache should cover
         every query.
Cache  : data/benchmark/rewrites_<model-slug>.json, key format identical to per_query.keys
         (f"{ozelgeTarih}|{baslik}") — directly compatible with bootstrap_compare.py.
         The script resumes: if --out already exists, existing keys are skipped.
Robustness: outputs containing <think> / empty-or-too-short / refusal-pattern text are
         retried twice; if still unsuccessful, that key is NOT written to the cache
         (downstream resolve_query() already falls back to the original question when a
         key is missing from the cache). "madde N" (article N) patterns not present in the
         original question (possible hallucination) are additionally written to
         rewrites_<slug>_flagged.json.

Usage:
    python src/benchmark/generate_rewrites.py
    python src/benchmark/generate_rewrites.py --sample-review 30
"""

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import requests

from bm25_baseline import load_test

DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUT_DIR = DATA_DIR / "benchmark"

PROMPT_VERSION = "v1"

# This prompt is sent to the LLM in Turkish on purpose: the task is rewriting Turkish tax
# questions into Turkish legal language, so both the instructions and the few-shot examples
# stay in Turkish to match the input/output language the model is actually producing.
SYSTEM_PROMPT = """Sen bir KDV (Katma Değer Vergisi) hukuku metin editörüsün. Görevin, mükellefin \
günlük dilde sorduğu bir özelge sorusunu, KDV Kanunu madde metinlerinde geçen soyut hukuki \
terminolojiye yaklaştırarak yeniden yazmak.

Kurallar:
1. Orijinal soruda geçmeyen hiçbir madde numarası, kanun ismi ya da hukuki sonuç EKLEME. \
Sadece sorunun içinde zaten var olan hukuki-kavramsal içeriği yeniden ifade et.
2. Mükellefe özgü somut detayları (şirket/kurum adı, tutar, tarih, belge numarası, GTİP kodu \
vb.) at; sorunun konusunu belirleyen kavramsal çekirdeği (örn. "ithalat", "hurda metal geri \
kazanımı", "istisna talebi") koru.
3. KDV Kanunu'nda geçen terimleri kullan: teslim, hizmet ifası, vergiyi doğuran olay, matrah, \
oran, istisna, muafiyet, tevkifat, iade, mükellef, vergi sorumlusu gibi.
4. Çıktı SADECE yeniden yazılmış soru metni olsun — tek paragraf, açıklama yok, markdown yok, \
tırnak yok.

Örnekler:
Orijinal: "ABC Ltd. Şti. 2023 yılında Almanya'dan hurda bakır ithal etti, bu ithalat için KDV \
ödemesi gerekir mi?"
Yeniden yazım: Yurt dışından hurda metal ithalatında KDV mükellefiyeti ve olası istisna durumu \
nedir?

Orijinal: "Şirketimiz 15.03.2023 tarihinde Y firmasına yazılım hizmeti verdi, fatura KDV'siz \
kesilebilir mi?"
Yeniden yazım: Yazılım hizmeti ifasının KDV'den istisna olup olmadığı ve faturalandırma usulü \
nedir?"""

# Turkish refusal phrasing, matched against the same local LLM's (Turkish-language) output.
REFUSAL_PATTERNS = re.compile(
    r"yardımcı olamam|üzgünüm|maalesef|bu konuda bilgi veremem|as an ai|i cannot",
    re.IGNORECASE,
)
# "madde" = "article" in Turkish; matches article-number mentions in the (Turkish) question/rewrite text.
MADDE_NUMBER_RE = re.compile(r"madde\s*(\d+)", re.IGNORECASE)


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1]


def out_path(model_name: str) -> Path:
    return OUT_DIR / f"rewrites_{model_slug(model_name)}.json"


def flagged_path(model_name: str) -> Path:
    return OUT_DIR / f"rewrites_{model_slug(model_name)}_flagged.json"


def load_records() -> list[dict]:
    records = load_test(include_train=True)
    return [r for r in records if (r.get("soru") or "").strip()]


def load_existing(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("rewrites", {})


def is_bad_output(text: str) -> bool:
    if not text or len(text.strip()) < 10:
        return True
    if "<think>" in text.lower():
        return True
    if REFUSAL_PATTERNS.search(text):
        return True
    return False


def call_llm(session: requests.Session, url: str, model: str, soru: str) -> str | None:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": soru},
        ],
        "temperature": 0.3,
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    for attempt in range(3):
        try:
            resp = session.post(f"{url}/v1/chat/completions", json=payload, timeout=120)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[rewrite] request error (attempt {attempt + 1}/3): {e}")
            continue
        if not is_bad_output(text):
            return text
        print(f"[rewrite] invalid output, retrying (attempt {attempt + 1}/3): {text[:80]!r}")
    return None


def extract_madde_numbers(text: str) -> set[str]:
    return set(MADDE_NUMBER_RE.findall(text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--vllm-url", default="http://localhost:8002")
    parser.add_argument("--vllm-model", default=None, help="defaults to --model")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--sample-review", type=int, default=0,
                         help="Prints N random original/rewrite pairs to the console after generation")
    args = parser.parse_args()
    vllm_model = args.vllm_model or args.model
    out = args.out or out_path(args.model)

    records = load_records()
    existing = load_existing(out)
    todo = [
        r for r in records
        if f"{r['ozelgeTarih']}|{r['baslik']}" not in existing
    ]
    print(f"Total queries: {len(records)}  Already cached: {len(existing)}  To generate: {len(todo)}")

    session = requests.Session()
    rewrites = dict(existing)

    def process(record: dict) -> tuple[str, dict | None]:
        key = f"{record['ozelgeTarih']}|{record['baslik']}"
        rewrite = call_llm(session, args.vllm_url, vllm_model, record["soru"])
        if rewrite is None:
            return key, None
        return key, {"soru": record["soru"], "rewrite": rewrite}

    n_done = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for key, value in ex.map(process, todo):
            n_done += 1
            if value is not None:
                rewrites[key] = value
            if n_done % 20 == 0 or n_done == len(todo):
                print(f"[rewrite] {n_done}/{len(todo)} processed", flush=True)

    n_failed = len(todo) - sum(1 for r in todo if f"{r['ozelgeTarih']}|{r['baslik']}" in rewrites)
    print(f"Generation complete. Failed (not written to cache): {n_failed}/{len(todo)}")

    payload = {
        "model": args.model,
        "prompt_version": PROMPT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rewrites": rewrites,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Output: {out}  ({len(rewrites)} rewrites)")

    flagged = {}
    for key, v in rewrites.items():
        orig_numbers = extract_madde_numbers(v["soru"])
        rewrite_numbers = extract_madde_numbers(v["rewrite"])
        extra = rewrite_numbers - orig_numbers
        if extra:
            flagged[key] = {"soru": v["soru"], "rewrite": v["rewrite"], "extra_madde_numbers": sorted(extra)}
    if flagged:
        fpath = flagged_path(args.model)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(flagged, f, ensure_ascii=False, indent=2)
        print(f"WARNING: {len(flagged)} rewrite(s) contain an article number not present in the original -> {fpath}")
    else:
        print("No hallucination-suspect article numbers found.")

    if args.sample_review:
        sample_keys = random.sample(list(rewrites.keys()), min(args.sample_review, len(rewrites)))
        print(f"\n--- {len(sample_keys)} random samples ---")
        for key in sample_keys:
            v = rewrites[key]
            print(f"\n[{key}]")
            print(f"  ORIGINAL : {v['soru']}")
            print(f"  REWRITE  : {v['rewrite']}")


if __name__ == "__main__":
    main()
