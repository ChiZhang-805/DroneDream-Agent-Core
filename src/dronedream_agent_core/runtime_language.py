"""Narrow lexical guards for runtime intent, not a replacement for semantic reasoning.

These rules can veto a misleading keyword match. They cannot authorize actuation
or prove the meaning of arbitrary natural language; ambiguous requests stay held.
"""

from __future__ import annotations

import re

_NEGATION = re.compile(
    r"\b(?:not|never|don't|do not|no|avoid|without)\b|不要|不能|不许|禁止|别|无需"
)
_DEFERRED = re.compile(
    r"\b(?:after|later|then|once|when|if|until)\b|之后|以后|到达|完成.*后|如果|等到"
)
_QUESTION = re.compile(r"\b(?:should|would|could|whether)\b|是否|要不要|能不能")


# 功能：
#   转义动作短语并为英文增加词边界，避免把标识符 recording_status 中的子串当作动作。
# 输入：
#   phrase：内部规则提供的动作短语。
# 输出：
#   pattern：忽略大小写拼写差异后的字面匹配正则。
def _pattern(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase.casefold())
    if phrase.isascii():
        escaped = rf"(?<![a-z0-9_]){escaped}(?![a-z0-9_])"
    pattern = re.compile(escaped)
    return pattern


# 功能：
#   在有界文字中检测短语是否被提及，包括否定提及；结果不能直接作为动作意图。
# 输入：
#   text：最长四千字符的用户文字。
#   phrase：内部规则提供的短语。
# 输出：
#   mentioned：文字中是否存在符合边界的字面匹配。
def phrase_mentioned(text: str, phrase: str) -> bool:
    if not isinstance(text, str) or len(text) > 4000:
        raise ValueError("RUNTIME_LANGUAGE_TEXT_INVALID")
    mentioned = _pattern(phrase).search(text.casefold()) is not None
    return mentioned


# 功能：
#   1. 排除同分句前方的否定、疑问以及后方问号，必要时排除延后条件。
#   2. 此规则只提供保守词法筛选，不证明任意自然语言语义，也不授予控制权限。
# 输入：
#   text：最长四千字符的用户文字。
#   phrase：内部规则提供的动作短语。
#   allow_deferred：是否允许匹配包含未来或条件步骤的表述。
# 输出：
#   present：是否至少存在一个没有被上述规则否决的匹配。
def affirmative_phrase_present(text: str, phrase: str, *, allow_deferred: bool = True) -> bool:
    if not isinstance(text, str) or len(text) > 4000:
        raise ValueError("RUNTIME_LANGUAGE_TEXT_INVALID")
    normalized = text.casefold()
    present = False
    for match in _pattern(phrase).finditer(normalized):
        prefix = re.split(r"[。.!?！？;；,，\n]", normalized[: match.start()])[-1]
        suffix = re.split(r"[。.!！;；,，\n]", normalized[match.end() :])[0]
        if _NEGATION.search(prefix) or _QUESTION.search(prefix) or "?" in suffix or "？" in suffix:
            continue
        # 逗号不能让「到达以后，立即降落」变成立即执行；紧急规则保留前方条件约束。
        if not allow_deferred and (
            _DEFERRED.search(normalized[: match.start()]) or _DEFERRED.search(suffix)
        ):
            continue
        present = True
        break
    return present
