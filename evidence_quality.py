"""Separate recorded statements from questions before retrieval or reuse."""
import re

MEASURED=re.compile(r'\d+(?:\.\d+)?\s*(?:t/s|tok/s|token/s|GiB|GB/s|ms|秒|%|s\b)',re.I)


def body_text(content):
    return content.split('\n',1)[-1] if content.startswith('角色：') else content


def question_only(content):
    body=body_text(content).strip()
    if not body:
        return True
    table=bool(re.search(r'(?m)^\s*\|.*\|',body))
    interrogative=re.compile(r'[?？]|(?:什么|为何|为什么|如何|怎么|哪些|吗[。！!\s]*$|呢[。！!\s]*$)')
    clauses=[c.strip() for c in re.split(r'(?<=[。！？?])|\n',body) if c.strip()]
    if interrogative.search(body) and not table and all(interrogative.search(c) for c in clauses):
        return True
    if MEASURED.search(body) or table:
        return False
    # Chat requests and questions describe information needs, not completed work.
    if content.startswith('角色：user；') and re.match(r'(?:请|帮|能否|能不能|麻烦|把|让|我要|我想|需要|继续|开始)',body):
        return True
    return len(body)<240 and bool(interrogative.search(body))


def factual_sources(sources):
    return [s for s in sources if not question_only(s.get('content',''))]
