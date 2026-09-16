"""Character-bounded preview assembly; never expand all cell values before clipping."""


class PreviewText:
    # 功能：
    #   为一份文本预览建立字符预算，标题、正文和分隔符使用同一个预算。
    # 输入：
    #   self：新建的预览缓冲器。
    #   limit：允许的字符数，取 1 至 200000 的整数。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, limit: int):
        if type(limit) is not int or not 0 < limit <= 200_000:
            raise ValueError("ATTACHMENT_TEXT_LIMIT_INVALID")
        self.remaining = limit
        self.truncated = False
        self._parts: list[str] = []
        self._started = False

    # 功能：
    #   1. 只保留预算内的字符串片段，不先拼出超长中间字符串。
    #   2. 恰好用完预算不算截断，只有真正丢弃字符时才标记截断。
    # 输入：
    #   self：当前预览缓冲器。
    #   text：本次追加的文字。
    #   separator：已有片段与新片段之间的分隔符。
    # 输出：
    #   accepted：本次内容是否全部保留。
    def append(self, text: str, *, separator: str = "") -> bool:
        if not isinstance(text, str) or not isinstance(separator, str):
            raise ValueError("ATTACHMENT_TEXT_FRAGMENT_INVALID")
        if not self.truncated:
            for fragment in (separator if self._started else "", text):
                kept = fragment[: self.remaining]
                if kept:
                    self._parts.append(kept)
                    self.remaining -= len(kept)
                if len(kept) < len(fragment):
                    self.truncated = True
                    break
            # 空段落也占一个段落位置，但不必为每个空字符串分配列表项。
            self._started = True
        accepted = not self.truncated
        return accepted

    # 功能：
    #   合并已保留的片段，合并结果不会超过初始化时的字符预算。
    # 输入：
    #   self：当前预览缓冲器。
    # 输出：
    #   text：完整的受限预览文字。
    def render(self) -> str:
        text = "".join(self._parts)
        return text
