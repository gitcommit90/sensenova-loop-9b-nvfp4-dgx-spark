"""Light-weight re-implementation of a SUBSET of the IFEval instruction checkers.

Semantics follow google-research/google-research `instruction_following_eval/instructions.py`
(Apache-2.0, (c) Google LLC), re-written from scratch with these documented deviations:
  * word tokenisation uses regex \\w+ (orig: nltk RegexpTokenizer(r"\\w+") -> identical) and
    capital-word counting uses whitespace/punct split instead of nltk.word_tokenize;
  * `length_constraints:number_sentences` and a few rarely used ids are NOT supported; the eval
    builder only selects prompts whose every instruction id is in SUPPORTED.
  * language detection uses `langdetect` (Apache-2.0) with a fixed seed, like the original.
Only "strict" scoring is implemented (prompt-level: all instructions pass).
"""
from __future__ import annotations

import json
import re
import threading

_LANG_LOCK = threading.Lock()  # langdetect seeds the *global* random module -> not thread-safe

try:
    import langdetect
    langdetect.DetectorFactory.seed = 0
except Exception:  # pragma: no cover
    langdetect = None


def _rel(count: int, relation: str, target: int) -> bool:
    if relation == "less than":
        return count < target
    if relation == "at least":
        return count >= target
    raise ValueError(relation)


def _lang(text: str) -> str | None:
    if langdetect is None:
        return None
    try:
        with _LANG_LOCK:
            langdetect.DetectorFactory.seed = 0
            return langdetect.detect(text)
    except Exception:
        return None


def no_comma(r, **k):
    return "," not in r


def number_highlighted_sections(r, num_highlights, **k):
    n = 0
    for h in re.findall(r"\*[^\n\*]*\*", r):
        if h.strip("*").strip():
            n += 1
    for h in re.findall(r"\*\*[^\n\*]*\*\*", r):
        if h.removeprefix("**").removesuffix("**").strip():
            n += 1
    return n >= num_highlights


def title(r, **k):
    return any(t.lstrip("<").rstrip(">").strip() for t in re.findall(r"<<[^\n]+>>", r))


def number_bullet_lists(r, num_bullets, **k):
    a = re.findall(r"^\s*\*[^\*].*$", r, flags=re.M)
    b = re.findall(r"^\s*-.*$", r, flags=re.M)
    return len(a) + len(b) == num_bullets


def json_format(r, **k):
    v = r.strip()
    v = re.sub(r"^```(?:json|Json|JSON)?\s*", "", v)
    v = re.sub(r"\s*```$", "", v).strip()
    try:
        json.loads(v)
        return True
    except ValueError:
        return False


def multiple_sections(r, section_spliter, num_sections, **k):
    pat = r"\s?" + re.escape(section_spliter.strip()) + r"\s?\d+\s?"
    return len(re.split(pat, r)) - 1 >= num_sections


def constrained_response(r, **k):
    v = r.strip()
    return any(o in v for o in ("My answer is yes.", "My answer is no.", "My answer is maybe."))


def english_lowercase(r, **k):
    return r == r.lower() and _lang(r) == "en"


def english_capital(r, **k):
    return r == r.upper() and _lang(r) == "en"


def capital_word_frequency(r, capital_frequency, capital_relation, **k):
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", r)
    n = sum(1 for w in words if w.isupper())
    return _rel(n, capital_relation, capital_frequency)


def number_words(r, num_words, relation, **k):
    return _rel(len(re.findall(r"\w+", r)), relation, num_words)


def number_paragraphs(r, num_paragraphs, **k):
    paras = re.split(r"\s?\*\*\*\s?", r)
    n = len(paras)
    for i, p in enumerate(paras):
        if not p.strip():
            if i == 0 or i == len(paras) - 1:
                n -= 1
            else:
                return False
    return n == num_paragraphs


def nth_paragraph_first_word(r, num_paragraphs, nth_paragraph, first_word, **k):
    paras = re.split(r"\n\n", r)
    n = sum(1 for p in paras if p.strip())
    paras_ne = [p for p in paras if p.strip()]
    if nth_paragraph > len(paras_ne):
        return False
    p = paras_ne[nth_paragraph - 1].strip()
    if not p:
        return False
    word = p.split()[0].strip()
    word = word.lstrip("'\"").rstrip(".,?!'\"").lower()
    # original strips leading/trailing punctuation char by char
    word = re.sub(r"[^\w]+$", "", re.sub(r"^[^\w]+", "", word))
    return n == num_paragraphs and word == first_word.lower()


def existence(r, keywords, **k):
    return all(re.search(kw, r, flags=re.I) for kw in keywords)


def frequency(r, keyword, frequency, relation, **k):
    return _rel(len(re.findall(keyword, r, flags=re.I)), relation, frequency)


def forbidden_words(r, forbidden_words, **k):
    return not any(re.search(r"\b" + w + r"\b", r, flags=re.I) for w in forbidden_words)


def letter_frequency(r, letter, let_frequency, let_relation, **k):
    return _rel(r.lower().count(letter.lower()), let_relation, let_frequency)


def end_checker(r, end_phrase, **k):
    return r.strip().strip('"').lower().endswith(end_phrase.strip().lower())


def quotation(r, **k):
    v = r.strip()
    return len(v) > 1 and v[0] == '"' and v[-1] == '"'


def postscript(r, postscript_marker, **k):
    v = r.lower()
    m = postscript_marker.strip()
    if m == "P.P.S":
        pat = r"\s*p\.\s?p\.\s?s.*$"
    elif m == "P.S.":
        pat = r"\s*p\.\s?s\..*$"
    else:
        pat = r"\s*" + re.escape(m.lower()) + r".*$"
    return bool(re.findall(pat, v, flags=re.M))


def number_placeholders(r, num_placeholders, **k):
    return len(re.findall(r"\[.*?\]", r)) >= num_placeholders


def repeat_prompt(r, prompt_to_repeat, **k):
    return r.strip().lower().startswith(prompt_to_repeat.strip().lower())


def two_responses(r, **k):
    parts = r.split("******")
    valid = []
    for i, p in enumerate(parts):
        if not p.strip():
            if i != 0 and i != len(parts) - 1:
                return False
        else:
            valid.append(p)
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


def response_language(r, language, **k):
    return _lang(r) == language


SUPPORTED = {
    "punctuation:no_comma": no_comma,
    "detectable_format:number_highlighted_sections": number_highlighted_sections,
    "detectable_format:title": title,
    "detectable_format:number_bullet_lists": number_bullet_lists,
    "detectable_format:json_format": json_format,
    "detectable_format:multiple_sections": multiple_sections,
    "detectable_format:constrained_response": constrained_response,
    "change_case:english_lowercase": english_lowercase,
    "change_case:english_capital": english_capital,
    "change_case:capital_word_frequency": capital_word_frequency,
    "length_constraints:number_words": number_words,
    "length_constraints:number_paragraphs": number_paragraphs,
    "length_constraints:nth_paragraph_first_word": nth_paragraph_first_word,
    "keywords:existence": existence,
    "keywords:frequency": frequency,
    "keywords:forbidden_words": forbidden_words,
    "keywords:letter_frequency": letter_frequency,
    "startend:end_checker": end_checker,
    "startend:quotation": quotation,
    "detectable_content:postscript": postscript,
    "detectable_content:number_placeholders": number_placeholders,
    "combination:repeat_prompt": repeat_prompt,
    "combination:two_responses": two_responses,
    "language:response_language": response_language,
}


def check(response: str, instruction_ids: list[str], kwargs_list: list[dict]) -> list[bool]:
    out = []
    for iid, kw in zip(instruction_ids, kwargs_list):
        kw = {k: v for k, v in (kw or {}).items() if v is not None}
        fn = SUPPORTED[iid]
        try:
            out.append(bool(response.strip()) and bool(fn(response, **kw)))
        except Exception:
            out.append(False)
    return out
