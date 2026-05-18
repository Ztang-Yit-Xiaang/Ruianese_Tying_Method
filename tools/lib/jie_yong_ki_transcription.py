#!/usr/bin/env python3
"""Transcription helpers for Jie Yong Ki / 張永愷 and older Ruianese codes."""

from __future__ import annotations

import re
from dataclasses import dataclass


TONE_RE = re.compile(r"^(?P<body>[a-z]+)(?P<tone>[1-8])?$")


@dataclass(frozen=True)
class ConversionResult:
    code: str
    status: str
    note: str


def split_tone(code):
    match = TONE_RE.match(code.strip())
    if not match:
        return code.strip(), ""
    return match.group("body"), match.group("tone") or ""


def candidate_bodies(body):
    candidates = [(body, "unchanged")]

    def add(value, note):
        if value and (value, note) not in candidates:
            candidates.append((value, note))

    if "uoo" in body:
        add(body.replace("uoo", "uoe"), "uoo->uoe")
    if "oo" in body:
        add(body.replace("oo", "oe"), "oo->oe")
    if "eu" in body:
        add(body.replace("eu", "e"), "eu->e")
    if "ii" in body:
        add(body.replace("ii", "i"), "ii->i")
    if body.startswith("dz"):
        add("zz" + body[2:], "dz->zz")
    if body.startswith("fv"):
        add("f" + body[2:], "fv->f")
    if body.startswith("vv"):
        add("v" + body[2:], "vv->v")
    if body.startswith("xy"):
        rest = body[2:]
        xy_map = {
            "ii": "i",
            "ie": "ie",
            "iae": "iae",
            "iao": "iao",
            "ao": "iao",
            "jue": "yue",
            "ju": "yu",
        }
        add("x" + xy_map.get(rest, rest), "xy->x")
    if body.startswith("ny"):
        rest = body[2:]
        ny_map = {
            "ii": "i",
            "ie": "ie",
            "iae": "iae",
            "iao": "iao",
            "jue": "yue",
            "ju": "yu",
            "o": "yo",
        }
        add("nj" + ny_map.get(rest, rest), "ny->nj")
    for old, new in (("qjue", "qyue"), ("jjjue", "jjyue"), ("jju", "jyu"), ("jue", "jyue"), ("jii", "ji"), ("qii", "qi"), ("jjii", "jji")):
        if body == old:
            add(new, f"{old}->{new}")
    return candidates


def expand_candidates(body):
    seen = []
    queue = list(candidate_bodies(body))
    while queue:
        current, note = queue.pop(0)
        if current in seen:
            continue
        seen.append(current)
        for next_value, next_note in candidate_bodies(current):
            if next_value not in seen:
                combined = next_note if note == "unchanged" else f"{note}; {next_note}"
                queue.append((next_value, combined))
    return seen


def convert_code(code, valid_syllables):
    body, tone = split_tone(code)
    if not body:
        return ConversionResult(code, "unresolved", "empty code")
    valid = valid_syllables
    direct = body + tone
    if direct in valid:
        return ConversionResult(direct, "ok", "already valid")

    checked = []
    for candidate in expand_candidates(body):
        converted = candidate + tone
        checked.append(converted)
        if converted in valid:
            return ConversionResult(converted, "converted", f"{code}->{converted}")
    return ConversionResult(code, "unresolved", "no valid mapping; tried " + ", ".join(checked[:12]))
