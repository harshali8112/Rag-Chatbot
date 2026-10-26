"""Evaluate the RAG chatbot's retrieval (and optionally its answers).

Run from your project folder, next to rag_pipeline.py and eval_questions.json:

    python evaluate.py --pdf "data/Unit 2 Cloud Computing.pptx.pdf"
    python evaluate.py --pdf "data/Unit 2 Cloud Computing.pptx.pdf" --chunk-size 800 --chunk-overlap 100
    python evaluate.py --pdf "data/Unit 2 Cloud Computing.pptx.pdf" --k 3 --mmr
    python evaluate.py --pdf "data/Unit 2 Cloud Computing.pptx.pdf" --answers
    python evaluate.py --pdf "data/Unit 2 Cloud Computing.pptx.pdf" --answers-only --resume   # continue after a quota error

What it measures (for the in-scope questions only):
  page hit       : at least one retrieved chunk comes from a page that really contains the answer
  keyword hit    : at least one expected key term appears in the retrieved chunks
  keyword cover  : the share of expected key terms found across the retrieved chunks (best for list questions)

Every run adds one row to results_log.csv, so your experiments build a results table for you.
It uses its own index folder (chroma_eval), so your app's chroma_db is never touched.
With --answers it also calls Gemini once per question (this uses API quota) and saves answers.csv
for you to grade by hand, including whether out-of-scope questions were correctly refused.
Progress is saved after every question. If Gemini reports a quota or rate-limit error, the script
retries, then stops cleanly; run it again later with --answers-only --resume and it keeps the answers
(and any grades you typed) and only asks the questions that are still missing.
"""
import argparse
import csv
import datetime
import json
import pathlib
import re
import time

import rag_pipeline as rp

HERE = pathlib.Path(__file__).resolve().parent


def norm(text):
    """Lower-case and collapse whitespace so 'Service\\nProvider' matches 'service provider'."""
    return re.sub(r"\s+", " ", text).lower()


def evaluate_retrieval(questions, k):
    rows = []
    for item in questions:
        docs = rp.retrieve(item["question"], k=k)
        pages_found = sorted({d.metadata["page"] + 1 for d in docs if isinstance(d.metadata.get("page"), int)})
        blob = norm(" ".join(d.page_content for d in docs))
        keywords = item["expected_keywords"]
        found = [kw for kw in keywords if norm(kw) in blob]
        rows.append({
            "id": item["id"],
            "question": item["question"],
            "page_hit": bool(set(pages_found) & set(item["expected_pages"])),
            "keyword_hit": bool(found),
            "keyword_cover": len(found) / len(keywords),
            "missing": [kw for kw in keywords if kw not in found],
            "pages_found": pages_found,
        })
    return rows


ERROR_PREFIX = "Sorry, the language model request failed"
GRADE_COL = "your_grade_1_correct_0_wrong"
COLUMNS = ["id", "type", "question", "chatbot_answer", "reference_answer",
           "error", "said_could_not_find", GRADE_COL]


def is_error(answer):
    return (answer or "").startswith(ERROR_PREFIX)


def load_done(path):
    """Rows from an earlier answers.csv that hold a real answer (works with the older file format too)."""
    done = {}
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                answer = row.get("chatbot_answer") or ""
                if answer and not is_error(answer):
                    done[int(row["id"])] = row
    return done


def run_answers(questions, delay, out_path, resume, retries, retry_wait, max_fail_streak):
    """Ask the real LLM every question and save the answers for manual grading."""
    done = load_done(out_path) if resume else {}
    results, fail_streak = {}, 0

    def blank(item):
        return {"id": item["id"], "type": item["type"], "question": item["question"], "chatbot_answer": "",
                "reference_answer": item["reference_answer"], "error": True,
                "said_could_not_find": False, GRADE_COL: ""}

    def save():
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            for item in questions:
                writer.writerow(results.get(item["id"], blank(item)))

    for item in questions:
        qid = item["id"]
        if qid in done:
            row = done[qid]
            results[qid] = {c: row.get(c, "") for c in COLUMNS}
            results[qid]["error"] = False
            print(f"  kept    {qid:>2}: already answered")
            continue
        if fail_streak >= max_fail_streak:
            results[qid] = blank(item)
            continue

        answer = ""
        for attempt in range(retries + 1):
            answer, _ = rp.generate_rag_answer(item["question"])
            answer = answer or ""
            if not is_error(answer):
                break
            if attempt < retries:
                print(f"  question {qid}: model error, waiting {retry_wait:.0f}s before retry {attempt + 1}/{retries} ...")
                time.sleep(retry_wait)

        failed = is_error(answer)
        fail_streak = fail_streak + 1 if failed else 0
        refused = (not failed) and ("could not find" in answer.lower() or "don't know" in answer.lower())
        results[qid] = {"id": qid, "type": item["type"], "question": item["question"], "chatbot_answer": answer,
                        "reference_answer": item["reference_answer"], "error": failed,
                        "said_could_not_find": refused, GRADE_COL: ""}
        save()  # progress is never lost
        print(f"  {'ERROR   ' if failed else 'answered'}{qid:>2}: {item['question'][:60]}")
        if not failed:
            time.sleep(delay)  # stay under free-tier rate limits

    save()
    total_ok = sum(1 for r in results.values() if r["chatbot_answer"] and not r["error"])
    print(f"\nSaved {out_path}  ({total_ok} of {len(questions)} questions have a real answer)")
    if total_ok < len(questions):
        print("Some questions still have no answer, most likely because of Gemini's free-tier quota.")
        print("Wait a while (a daily quota resets the next day), then run again with:  --answers-only --resume")
        print("Only the missing questions will be asked. Tip: check Google AI Studio for your current limits.")
    else:
        print("Open it, read each answer against the slides, and put 1 (correct) or 0 (wrong) in the last column.")
        print("For out-of-scope questions the right result is said_could_not_find = True.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", required=True, help="PDF to index for this run")
    parser.add_argument("--questions", default=str(HERE / "eval_questions.json"))
    parser.add_argument("--chunk-size", type=int, default=rp.CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=rp.CHUNK_OVERLAP)
    parser.add_argument("--k", type=int, default=rp.TOP_K)
    parser.add_argument("--mmr", action="store_true", help="use MMR (more varied chunks)")
    parser.add_argument("--answers", action="store_true", help="also call Gemini and save answers.csv")
    parser.add_argument("--answers-only", action="store_true",
                        help="skip the retrieval scoring and the log row; just ask Gemini (implies --answers)")
    parser.add_argument("--resume", action="store_true",
                        help="keep real answers (and grades) already in answers.csv; ask only the missing ones")
    parser.add_argument("--delay", type=float, default=6.0, help="seconds between Gemini calls with --answers")
    parser.add_argument("--retries", type=int, default=2, help="retries per question after a model error")
    parser.add_argument("--retry-wait", type=float, default=60.0, help="seconds to wait before a retry")
    parser.add_argument("--max-fail-streak", type=int, default=3,
                        help="stop after this many questions in a row fail (quota probably used up)")
    parser.add_argument("--note", default="", help="a short label for this run, saved in results_log.csv")
    args = parser.parse_args()

    # Settings for this run (the functions in rag_pipeline read these at call time)
    rp.CHUNK_SIZE, rp.CHUNK_OVERLAP, rp.USE_MMR = args.chunk_size, args.chunk_overlap, args.mmr
    rp.PERSIST_DIRECTORY = str(HERE / "chroma_eval")

    questions = json.loads(pathlib.Path(args.questions).read_text(encoding="utf-8"))
    in_scope = [q for q in questions if q["type"] == "in_scope"]

    print(f"Indexing {args.pdf} (chunk size {args.chunk_size}, overlap {args.chunk_overlap}) ...")
    rp.initialize_vector_store(args.pdf)

    if not args.answers_only:
        rows = evaluate_retrieval(in_scope, args.k)
        for r in rows:
            status = "PASS" if r["page_hit"] and r["keyword_hit"] else "FAIL"
            extra = f" missing={r['missing']}" if r["missing"] else ""
            print(f"{status}  Q{r['id']:<2} pages={r['pages_found']}{extra}  {r['question'][:55]}")

        n = len(rows)
        page_hit = 100 * sum(r["page_hit"] for r in rows) / n
        kw_hit = 100 * sum(r["keyword_hit"] for r in rows) / n
        cover = 100 * sum(r["keyword_cover"] for r in rows) / n
        print(f"\nSettings: chunk_size={args.chunk_size} overlap={args.chunk_overlap} k={args.k} mmr={args.mmr}")
        print(f"Questions: {n}")
        print(f"Page hit      : {page_hit:5.1f}%")
        print(f"Keyword hit   : {kw_hit:5.1f}%")
        print(f"Keyword cover : {cover:5.1f}%")

        log = HERE / "results_log.csv"
        new_file = not log.exists()
        with open(log, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["date", "note", "chunk_size", "overlap", "k", "mmr", "questions",
                                 "page_hit_pct", "keyword_hit_pct", "keyword_cover_pct"])
            writer.writerow([datetime.date.today().isoformat(), args.note, args.chunk_size, args.chunk_overlap,
                             args.k, args.mmr, n, round(page_hit, 1), round(kw_hit, 1), round(cover, 1)])
        print(f"Added a row to {log.name}")

    if args.answers or args.answers_only:
        print("\nAsking Gemini the questions (uses API quota) ...")
        run_answers(questions, args.delay, HERE / "answers.csv", args.resume,
                    args.retries, args.retry_wait, args.max_fail_streak)


if __name__ == "__main__":
    main()