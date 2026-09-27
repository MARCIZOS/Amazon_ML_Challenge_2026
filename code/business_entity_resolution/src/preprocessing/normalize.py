"""Unicode-safe canonicalisation of business names and addresses.

Design rules (from the team role spec and the EDA):
  * Original fields are never modified - every function returns NEW fields.
  * Non-ASCII text is never deleted. Native-script tokens (Devanagari, Tamil,
    Gujarati, ...) are mapped to English with a vocabulary learned from the
    training pairs (see translit_vocab.py) and otherwise transliterated with
    `anyascii` (ISC licence). Accented Latin text is folded (e -> e) only in the
    derived comparison fields.
  * Country is an open set: US / India / France word lists are used when the
    label matches, any other label falls back to trying all of them.

Output fields per record (see docs/DATA_DICTIONARY.md):
  name_clean   full cleaned name (lowercase ascii, legal forms canonicalised)
  name_core    name without legal forms, stop words, honorifics, country words
  name_compact name_core with spaces removed (matches 'deepeshagri.com' to 'Deepesh Agri')
  name_alt     the text before a trade-name marker (t/a, dba, aka, nee ...), cleaned
  name_legal   sorted canonical legal forms found, e.g. "ltd pvt"
  name_key     phonetic skeleton of name_core tokens (robust to typos/translit)
  name_script  dominant script of the ORIGINAL name (LATIN, DEVANAGARI, ...)
  name_translit 1 if native-script text was converted
  addr_clean   full cleaned address with canonical street abbreviations
  addr_core    address words only (no numbers, state, stop words)
  addr_key     phonetic skeleton of addr_core tokens
  addr_nums    sorted unique numbers from the address (leading zeros removed)
  addr_state   canonical admin area, e.g. US-TX, IN-MH, FR-NAQ ('' if none)
  addr_missing 1 if the address is empty / placeholder only
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from functools import lru_cache

try:
    from anyascii import anyascii as _to_ascii
except ImportError:  # pragma: no cover - fallback keeps the module importable
    def _to_ascii(s: str) -> str:
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LEXICON = os.path.normpath(os.path.join(HERE, "..", "..", "configs", "lexicons.json"))
DEFAULT_VOCAB = os.path.normpath(os.path.join(HERE, "..", "..", "configs", "translit_vocab.json"))

NAME_FIELDS = ["name_clean", "name_core", "name_compact", "name_alt", "name_legal", "name_key",
               "name_script", "name_translit"]
ADDR_FIELDS = ["addr_clean", "addr_core", "addr_key", "addr_nums", "addr_state",
               "addr_missing"]
OUTPUT_FIELDS = NAME_FIELDS + ADDR_FIELDS

# ----------------------------------------------------------------------------- regexes
ZERO_WIDTH = re.compile(r"[​-‏‪-‮⁠﻿­]")
URL_RE = re.compile(r"\b(?:https?://\S+|www\.\S+|\S+@\S+\.\S+|[a-z0-9-]+\.(?:com|in|net|org|fr|co|biz|info)\b\S*)",
                    re.I)
TRAILING_ID = re.compile(r"\s[-–—]\s*\d{5,}\s*$")
LONG_NUM = re.compile(r"\b\d{6,}\b")
ELISION = re.compile(r"\b(l|d|qu)['’‘`]", re.I)
APOS = re.compile(r"['’‘`´]")
NON_ALNUM = re.compile(r"[^0-9a-z]+")
LETTER_DIGIT = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])")
DIGITS = re.compile(r"\d+")
ORDINAL = re.compile(r"^\d+(st|nd|rd|th|er|e|eme|re)$")
SEGMENT_SPLIT = re.compile(r"[,;()\[\]{}|]+|\s[-–]\s")
WS = re.compile(r"\s+")
MS_PREFIX = re.compile(r"(?:^|\s)m/s\.?(?=\s)", re.I)

LEET = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b", "9": "g"}


# ----------------------------------------------------------------------------- helpers
def unicode_canonical(s) -> str:
    """NFKC-normalise, drop zero-width/control characters, collapse whitespace.

    Safe for every script: nothing meaningful is removed.
    """
    if s is None:
        return ""
    if isinstance(s, float) and s != s:  # NaN
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = ZERO_WIDTH.sub("", s)
    s = "".join(ch if (ch.isprintable() or ch == " ") else " " for ch in s)
    return WS.sub(" ", s).strip()


def _script_of(ch: str) -> str:
    try:
        return unicodedata.name(ch).split(" ")[0]
    except ValueError:
        return "UNKNOWN"


def dominant_script(s: str) -> str:
    """Most frequent script among the letters of `s` (LATIN if ascii only)."""
    counts = {}
    for ch in s:
        if ch.isalpha():
            sc = "LATIN" if ch.isascii() else _script_of(ch)
            counts[sc] = counts.get(sc, 0) + 1
    if not counts:
        return "NONE"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def has_non_latin(tok: str) -> bool:
    return any(ch.isalpha() and not ch.isascii() and _script_of(ch) != "LATIN" for ch in tok)


def native_key(tok: str) -> str:
    """Key used for the learned transliteration vocabulary."""
    tok = unicode_canonical(tok)
    return tok.strip(".,;:!?()[]{}\"'-_/\\|*#@+<>~`").lower()


@lru_cache(maxsize=2_000_000)
def skeleton(tok: str) -> str:
    """Phonetic consonant skeleton of one ascii token.

    'private'/'praivet'/'praaivett' -> 'prbt'; 'limited'/'limtid' -> 'lmt'.
    Deliberately lossy (d->t, v/w->b, c/q->k) so it acts as a fuzzy key.
    Numbers are returned unchanged.
    """
    if not tok or tok.isdigit():
        return tok
    t = tok.replace("ph", "f").replace("ck", "k").replace("x", "ks")
    t = t.translate(str.maketrans("cqzvwd", "kksbbt"))
    t = re.sub(r"m(?=[kg])", "n", t)
    t = re.sub(r"([bdfgjklmnprstk])h", r"\1", t)
    first, rest = t[0], re.sub(r"[aeiouyh]", "", t[1:])
    out = []
    for ch in first + rest:
        if not out or out[-1] != ch:
            out.append(ch)
    return "".join(out)


@lru_cache(maxsize=1_000_000)
def fix_leet(tok: str) -> str:
    """Undo digit-for-letter substitutions inside a word: '6eneral' -> 'general',
    'we1lness' -> 'wellness', '5w' -> 'sw'. Ordinals ('2nd') and mostly-numeric
    tokens are left alone."""
    if tok.isdigit() or tok.isalpha() or not tok.isalnum():
        return tok
    if ORDINAL.match(tok):
        return tok
    n_dig = sum(c.isdigit() for c in tok)
    if n_dig > 2 or n_dig >= len(tok) or any(c.isdigit() and c not in LEET for c in tok):
        return tok
    return "".join(LEET.get(c, c) for c in tok)


# ----------------------------------------------------------------------------- normalizer
class Normalizer:
    """Holds the lexicons + learned vocabulary and normalises records."""

    def __init__(self, lexicon_path: str | None = None, vocab_path: str | None = None):
        with open(lexicon_path or DEFAULT_LEXICON, encoding="utf-8") as f:
            lx = json.load(f)
        self.vocab = {}
        vp = vocab_path if vocab_path is not None else DEFAULT_VOCAB
        if vp and os.path.exists(vp):
            with open(vp, encoding="utf-8") as f:
                self.vocab = json.load(f).get("vocab", {})

        self.legal = lx["legal"]
        self.legal_multi = {tuple(k.split()): v for k, v in lx["legal_multi"].items()}
        self.legal_multi_max = max((len(k) for k in self.legal_multi), default=1)
        self.name_stop = set(lx["name_stop"])
        self.name_country = set(lx["name_country_words"])
        self.name_abbr = lx["name_abbr"]
        self.addr_abbr = lx["addr_abbr"]
        self.addr_stop = set(lx["addr_stop"])
        self.placeholders = set(p.lower() for p in lx["placeholders"])
        self.country_words = set(lx["country_words"])

        vals = set(self.legal.values()) | set(self.legal_multi.values())
        self.legal_vals = {p for v in vals for p in v.split()}

        markers = sorted(lx["trade_markers"], key=len, reverse=True)
        self.trade_re = re.compile(
            r"(?:^|\s)(" + "|".join(re.escape(m) for m in markers) + r")(?=\s|$)", re.I)

        # admin areas: phrase (normalised) -> canonical code, per country
        self.admin = {"us": {}, "india": {}, "france": {}}
        self.admin_exact_only = {"us": set(), "india": set(), "france": set()}
        for code, name in lx["us_states"].items():
            c = "US-" + code.upper()
            self.admin["us"][self._plain(name)] = c
            self.admin["us"][code] = c
            self.admin_exact_only["us"].add(code)
        for code, names in lx["in_states"].items():
            c = "IN-" + code.upper()
            self.admin["india"][code] = c
            self.admin_exact_only["india"].add(code)
            for n in names:
                self.admin["india"][self._plain(n)] = c
        for code, names in lx["fr_regions"].items():
            c = "FR-" + code.upper()
            for n in names:
                self.admin["france"][self._plain(n)] = c
        for code, deps in lx["fr_departments"].items():
            c = "FR-" + code.upper()
            for d in deps:
                p = self._plain(d)
                self.admin["france"][p] = c
                self.admin_exact_only["france"].add(p)  # 'somme', 'var' ... only as a full segment
        # fuzzy fallback: consonant skeleton of a state name (spaces removed)
        self.admin_skel = {k: {} for k in self.admin}
        for k, v in self.admin.items():
            for phrase, code in v.items():
                if phrase in self.admin_exact_only[k]:
                    continue
                sk = "".join(skeleton(t) for t in phrase.split())
                if len(sk) >= 4:
                    self.admin_skel[k].setdefault(sk, code)
        # multi-word phrases that may be matched inside a segment
        self.admin_phrases = {
            k: sorted([p for p in v if " " in p and p not in self.admin_exact_only[k]],
                      key=len, reverse=True)
            for k, v in self.admin.items()
        }

    # ------------------------------------------------------------------ basic text
    @staticmethod
    def _plain(s: str) -> str:
        s = _to_ascii(s).lower()
        s = APOS.sub("", s)
        return WS.sub(" ", NON_ALNUM.sub(" ", s)).strip()

    def _strip_urls(self, s: str) -> str:
        """Drop URLs/e-mails that decorate a name; if the name IS a domain
        ('deepeshagri.com') keep its label ('deepeshagri') instead."""
        rest = URL_RE.sub(" ", s)
        rest_words = [w for w in NON_ALNUM.sub(" ", rest.lower()).split()
                      if w not in self.name_stop and len(w) > 1]
        if rest_words:
            return rest

        def label(m):
            u = m.group(0)
            if "@" in u:
                u = u.split("@", 1)[1]
            u = re.sub(r"^(https?://)?(www\.)?", "", u, flags=re.I)
            return " " + u.split(".")[0] + " "
        return URL_RE.sub(label, s)

    def _translate_native(self, text: str) -> tuple[str, int]:
        """Replace native-script tokens using the learned vocab, else anyascii."""
        if text.isascii():
            return text, 0
        out, changed = [], 0
        for tok in text.split(" "):
            if has_non_latin(tok):
                changed = 1
                key = native_key(tok)
                eng = self.vocab.get(key)
                out.append(eng if eng else _to_ascii(tok))
            else:
                out.append(tok)
        return " ".join(out), changed

    def _to_tokens(self, text: str) -> list[str]:
        """ascii-fold, lowercase, drop punctuation -> tokens."""
        s = _to_ascii(text).lower()
        s = s.replace("&", " and ").replace("+", " ")
        s = ELISION.sub(r"\1 ", s)
        s = APOS.sub("", s)
        return NON_ALNUM.sub(" ", s).split()

    def _country_key(self, country: str) -> str | None:
        c = self._plain(country or "")
        if c in ("us", "usa", "united states", "united states of america", "america"):
            return "us"
        if c in ("india", "in", "ind", "bharat"):
            return "india"
        if c in ("france", "fr", "fra", "republique francaise"):
            return "france"
        return None  # unknown country: try every lexicon

    # ------------------------------------------------------------------ names
    def _name_tokens(self, text: str) -> tuple[list[str], list[str]]:
        toks = [fix_leet(t) for t in self._to_tokens(text)]
        toks = [self.name_abbr.get(t, t) for t in toks]
        # dedupe immediate repeats ("nq nq anchor")
        dedup = []
        for t in toks:
            if not dedup or dedup[-1] != t or len(t) == 1:
                dedup.append(t)
        # canonicalise multi-token legal forms (l l c -> llc)
        out, legal, i = [], [], 0
        while i < len(dedup):
            hit = None
            for n in range(min(self.legal_multi_max, len(dedup) - i), 1, -1):
                key = tuple(dedup[i:i + n])
                if key in self.legal_multi:
                    hit = (n, self.legal_multi[key])
                    break
            if hit:
                out.extend(hit[1].split())
                legal.extend(hit[1].split())
                i += hit[0]
                continue
            t = dedup[i]
            if t in self.legal:
                out.extend(self.legal[t].split())
                legal.extend(self.legal[t].split())
            else:
                out.append(t)
            i += 1
        return out, legal

    def _core(self, toks: list[str], legal_vals: set) -> list[str]:
        core = [t for t in toks
                if t not in legal_vals and t not in self.name_stop
                and t not in self.name_country and not LONG_NUM.fullmatch(t)]
        return core

    def name(self, raw) -> dict:
        orig = unicode_canonical(raw)
        script = dominant_script(orig)
        s = orig.split("|")[0]                      # "Name | www.site.com"
        s = MS_PREFIX.sub(" ", s)
        s = self._strip_urls(s)
        s = TRAILING_ID.sub(" ", s)
        s, translit = self._translate_native(s)

        alt = ""
        m = None
        for m in self.trade_re.finditer(s):
            pass                                    # last marker wins
        if m is not None:
            alt, s = s[:m.start()], s[m.end():]
            if not s.strip():                       # marker at the end -> keep the left part
                s, alt = alt, ""

        toks, legal = self._name_tokens(s)
        legal_vals = self.legal_vals
        core = self._core(toks, legal_vals)
        if not core:                                # e.g. "Private Limited" only
            core = [t for t in toks if t not in legal_vals] or toks
        alt_core = []
        if alt.strip():
            atoks, _ = self._name_tokens(alt)
            alt_core = self._core(atoks, legal_vals)

        return {
            "name_clean": " ".join(t for t in toks if not LONG_NUM.fullmatch(t)),
            "name_core": " ".join(core),
            "name_compact": "".join(core),
            "name_alt": " ".join(alt_core),
            "name_legal": " ".join(sorted(set(legal))),
            "name_key": " ".join(skeleton(t) for t in core),
            "name_script": script,
            "name_translit": translit,
        }

    # ------------------------------------------------------------------ addresses
    def _find_state(self, segments: list[str], ckey: str | None) -> tuple[str, set]:
        """Return (state_code, indices_of_segments_that_were_the_state)."""
        keys = [ckey] if ckey else ["us", "india", "france"]
        # 1) a whole segment equals a state / code / department
        for i in range(len(segments) - 1, -1, -1):          # states are usually last
            seg = segments[i]
            for k in keys:
                code = self.admin[k].get(seg)
                if code:
                    return code, {i}
        # 2) a multi-word state/region phrase inside a segment
        for i in range(len(segments) - 1, -1, -1):
            seg = " " + segments[i] + " "
            for k in keys:
                for p in self.admin_phrases[k]:
                    if " " + p + " " in seg:
                        segments[i] = WS.sub(" ", seg.replace(" " + p + " ", " ")).strip()
                        return self.admin[k][p], set()
        # 3) fuzzy: a short segment whose skeleton equals a state skeleton
        #    (catches transliterated native-script state names: 'mharastr')
        for i in range(len(segments) - 1, -1, -1):
            toks = segments[i].split()
            if not 1 <= len(toks) <= 3:
                continue
            sk = "".join(skeleton(t) for t in toks)
            for k in keys:
                code = self.admin_skel[k].get(sk)
                if code:
                    return code, {i}
        return "", set()

    def address(self, raw, country="") -> dict:
        orig = unicode_canonical(raw)
        s, _ = self._translate_native(orig)
        ckey = self._country_key(country)

        # split into comma-like segments, drop placeholders
        segs_raw = [x.strip() for x in SEGMENT_SPLIT.split(s)]
        segments = []
        for x in segs_raw:
            if not x or x.lower() in self.placeholders:
                continue
            toks = [t for t in self._to_tokens(x) if t not in self.placeholders]
            if toks:
                segments.append(" ".join(toks))

        state, state_idx = self._find_state(segments, ckey)
        body = " ".join(seg for i, seg in enumerate(segments) if i not in state_idx)

        # numbers (before letter/digit splitting so '0233' -> '233')
        body = LETTER_DIGIT.sub(" ", body)
        toks = body.split()
        toks = [str(int(t)) if t.isdigit() else t for t in toks]
        nums = sorted({t for t in toks if t.isdigit()}, key=lambda x: (len(x), x))
        toks = [self.addr_abbr.get(t, t) for t in toks]

        core = [t for t in toks
                if not t.isdigit() and t not in self.addr_stop
                and t not in self.country_words and len(t) > 1]
        clean = " ".join(toks)
        return {
            "addr_clean": clean,
            "addr_core": " ".join(core),
            "addr_key": " ".join(skeleton(t) for t in core),
            "addr_nums": " ".join(nums),
            "addr_state": state,
            "addr_missing": int(not core and not nums),
        }

    # ------------------------------------------------------------------ record
    def record(self, name, address, country="") -> dict:
        out = self.name(name)
        out.update(self.address(address, country))
        return out


# ----------------------------------------------------------------------------- dataframe API
def normalize_frame(df, normalizer: Normalizer | None = None):
    """Return a copy of `df` with the normalised columns appended.

    Original columns (entity_id, business_name, business_address, country) are
    kept untouched.
    """
    import pandas as pd  # local import keeps the module light for multiprocessing

    nz = normalizer or Normalizer()
    rows = [nz.record(n, a, c) for n, a, c in
            zip(df["business_name"], df["business_address"], df["country"])]
    extra = pd.DataFrame(rows, index=df.index, columns=OUTPUT_FIELDS)
    for c in ("name_translit", "addr_missing"):
        extra[c] = extra[c].astype("int8")
    return pd.concat([df, extra], axis=1)
