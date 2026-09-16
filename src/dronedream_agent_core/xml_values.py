"""Non-executing bounded XML parsing shared by attachments and simulation assets."""

from xml.etree import ElementTree


class _BoundedTreeBuilder(ElementTree.TreeBuilder):
    """Reject declarations and structural excess while the parser builds its tree."""

    # 功能：
    #   为单份 XML 文档建立元素与深度计数，避免解析状态在不同文档之间串用。
    # 输入：
    #   maximum_elements：此文档允许的元素总数上限。
    # 输出：
    #   None：不返回业务数据。
    def __init__(self, *, maximum_elements: int):
        super().__init__()
        self.depth = 0
        self.elements = 0
        self.maximum_elements = maximum_elements

    # 功能：
    #   在创建新节点之前检查深度和数量，避免树分配完成后才发现超限。
    # 输入：
    #   tag：当前开始标签名。
    #   attrs：当前元素的属性字典。
    # 输出：
    #   element：树构建器创建的当前元素。
    def start(self, tag, attrs):
        self.depth += 1
        self.elements += 1
        if self.depth > 128 or self.elements > self.maximum_elements:
            raise ValueError("XML_STRUCTURE_LIMIT")
        element = super().start(tag, attrs)
        return element

    # 功能：
    #   在正常完成结束标签处理后减少深度，保留底层解析器的标签匹配检查。
    # 输入：
    #   tag：当前结束标签名。
    # 输出：
    #   result：树构建器关闭的元素。
    def end(self, tag):
        result = super().end(tag)
        self.depth -= 1
        return result

    # 功能：
    #   在解析器识别出文档类型声明时拒绝 DTD，避免依赖原始字节搜索而漏过其他字符编码。
    # 输入：
    #   name：文档类型名。
    #   pubid：外部公共标识符。
    #   system：外部系统标识符。
    # 输出：
    #   None：不返回业务数据。
    def doctype(self, name, pubid, system):
        raise ValueError("XML_DOCTYPE_FORBIDDEN")


# 功能：
#   1. 在字节、元素和深度预算内解析 XML，不读取外部实体或执行仿真插件。
#   2. 仅保证解析边界；资产的引用文件、扩展声明和执行准入仍由调用方单独验证。
# 输入：
#   content：XML 原始字节串。
#   maximum_bytes：允许的原文大小上限。
#   maximum_elements：允许的元素总数上限。
# 输出：
#   root：解析得到的根元素。
def parse_xml(
    content: bytes, *, maximum_bytes: int, maximum_elements: int = 100_000
) -> ElementTree.Element:
    if (
        type(maximum_bytes) is not int
        or not 0 < maximum_bytes <= 64 * 1024 * 1024
        or type(maximum_elements) is not int
        or not 0 < maximum_elements <= 1_000_000
    ):
        raise ValueError("XML_BUDGET_INVALID")
    if type(content) is not bytes or len(content) > maximum_bytes:
        raise ValueError("XML_SIZE_LIMIT")
    root = ElementTree.fromstring(
        content,
        parser=ElementTree.XMLParser(target=_BoundedTreeBuilder(maximum_elements=maximum_elements)),
    )
    return root
