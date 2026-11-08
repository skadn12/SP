"""Versioned public-selection checks; historical v1 artifacts stay untouched.

This module detects presentation defects. Semantic operation support and answer
reliability still require independent content review. No automatic source,
option, answer or display rewriting is performed here.
"""
from __future__ import annotations
import re

COUNT_SELECTION = re.compile(
    r'\bwhich\s+(?:two|three|four|five|six|[2-6])\b'
    r'|\b(?:select|choose|pick|mark)\s+(?:(?:exactly|the)\s+)?(?:[2-6]|two|three|four|five|six)\b'
    r'|选(?:择|取)?\s*(?:恰好|正好)?\s*[二三四五六2-6]\s*(?:个|项|条)', re.I)
CONTROL_DISCLOSURE = re.compile(
    r'\ball[ -]correct\b|\baudit[ _-]*control\b|\brecord[ _-]*audit[ _-]*control\b'
    r'|\b(?:all|every)\s+(?:six|6)\s+(?:(?:entries|options|statements)\s+)?(?:are|can be|is)\s+(?:correct|supported)\b'
    r'|全部(?:正确|为真)|全对(?:控制)?题', re.I)
PUBLIC_OPERATION_MAP = {'MC_RECORD_AUDIT_CONTROL':'MC_FULL_RECORD_AUDIT'}

def public_operation(operation: str) -> str:
    return PUBLIC_OPERATION_MAP.get(operation, operation)

def inspect_public_selection(question: str, operation: str, question_type: str) -> dict:
    """Inspect public task text only, never factual source text or Gold."""
    flags=[]
    if question_type=='multi_select':
        matches=[m.group() for m in COUNT_SELECTION.finditer(question)]
        if matches: flags.append({'code':'QUESTION_REVEALS_SELECTION_COUNT','matches':matches})
        matches=[m.group() for m in CONTROL_DISCLOSURE.finditer(question+'\n'+operation)]
        if operation in PUBLIC_OPERATION_MAP: matches.append(operation)
        if matches: flags.append({'code':'PUBLIC_CONTROL_DISCLOSURE','matches':matches})
    return {'status':'PASS_PUBLIC_PRESENTATION_ONLY' if not flags else 'FAIL_PUBLIC_PRESENTATION',
            'flags':flags,'semantic_operation_verified':False,'gold_reliability_verified':False}

def require_public_selection(question: str, operation: str, question_type: str) -> dict:
    result=inspect_public_selection(question,operation,question_type)
    if result['flags']: raise ValueError(result)
    return result
