"""Bound a plain ZIP/ZIP64 index before ZipFile allocates its member objects.

Implements the single-volume central-directory layout used by
DroneDream data packages. Self-extractors, directory signatures, ZIP64 extensible
end records and appended payloads are deliberately outside this input profile.
Layouts follow PKWARE APPNOTE sections 4.3.12 and 4.3.14 through 4.3.16:
https://pkware.cachefly.net/webdocs/casestudies/APPNOTE.TXT
"""

import struct
from typing import BinaryIO


# 功能：
#   定位并读取固定长度结构记录，拒绝负偏移和截断数据。
# 输入：
#   source：调用方持有的可定位二进制输入。
#   offset：从输入开头计算的字节偏移。
#   size：调用方已限制的读取字节数。
# 输出：
#   value：长度与 size 一致的结构字节。
def _read_at(source: BinaryIO, offset: int, size: int) -> bytes:
    if offset < 0:
        raise ValueError("ZIP_INDEX_OFFSET_INVALID")
    source.seek(offset)
    value = source.read(size)
    if len(value) != size:
        raise ValueError("ZIP_INDEX_TRUNCATED")
    return value


# 功能：
#   1. 在普通 ZIP 解析分配成员对象前，以有界辅助内存核对目录结构、数量和体积。
#   2. 仅接受单卷普通 ZIP／固定 ZIP64 布局，成功后将输入位置归零。
#   3. 不校验内容、CRC、路径或信任；调用方须继续使用同一快照完成这些校验。
# 输入：
#   source：调用方持有的可定位二进制快照，不由本函数关闭。
#   maximum_members：允许的成员总数，取 1 至 20000 的整数。
#   maximum_index_bytes：中央目录字节预算，最大 32 MiB。
# 输出：
#   actual：实际逐项数出的中央目录成员数。
def validate_zip_index(
    source: BinaryIO, *, maximum_members: int, maximum_index_bytes: int = 32 * 1024 * 1024
) -> int:
    if (
        type(maximum_members) is not int
        or not 0 < maximum_members <= 20_000
        or type(maximum_index_bytes) is not int
        or not 0 < maximum_index_bytes <= 32 * 1024 * 1024
    ):
        raise ValueError("ZIP_INDEX_BUDGET_INVALID")
    source.seek(0, 2)
    file_size = source.tell()
    tail_size = min(file_size, 65535 + 22)
    # ZIP 尾记录最多携带 65535 字节注释，定位时无须读取整个文件。
    tail = _read_at(source, file_size - tail_size, tail_size)
    marker = tail.rfind(b"PK\x05\x06")
    if marker < 0 or len(tail) - marker < 22:
        raise ValueError("ZIP_INDEX_END_MISSING")
    _, disk, directory_disk, disk_count, count, size, offset, comment_length = struct.unpack_from(
        "<4s4H2IH", tail, marker
    )
    end_offset = file_size - tail_size + marker
    if marker + 22 + comment_length != len(tail):
        raise ValueError("ZIP_INDEX_TRAILING_DATA")
    if disk != 0 or directory_disk != 0:
        raise ValueError("ZIP_INDEX_MULTIVOLUME_FORBIDDEN")
    directory_end = end_offset
    locator = _read_at(source, end_offset - 20, 20) if end_offset >= 20 else b""
    if locator[:4] == b"PK\x06\x07":
        _, record_disk, record_offset, volumes = struct.unpack("<4sIQI", locator)
        if record_disk != 0 or volumes != 1:
            raise ValueError("ZIP_INDEX_MULTIVOLUME_FORBIDDEN")
        if record_offset + 56 != end_offset - 20:
            raise ValueError("ZIP_INDEX_ZIP64_LAYOUT_INVALID")
        record = struct.unpack("<4sQ2H2I4Q", _read_at(source, record_offset, 56))
        (
            magic,
            record_size,
            _,
            _,
            disk64,
            directory_disk64,
            disk_count64,
            count64,
            size64,
            offset64,
        ) = record
        if magic != b"PK\x06\x06" or record_size != 44:
            raise ValueError("ZIP_INDEX_ZIP64_LAYOUT_INVALID")
        if disk64 != 0 or directory_disk64 != 0 or disk_count64 != count64:
            raise ValueError("ZIP_INDEX_MULTIVOLUME_FORBIDDEN")
        # 普通尾记录可使用 ZIP64 哨兵，但不能另报一个与扩展记录矛盾的值。
        if any(
            old not in (sentinel, new)
            for old, new, sentinel in (
                (disk_count, disk_count64, 0xFFFF),
                (count, count64, 0xFFFF),
                (size, size64, 0xFFFFFFFF),
                (offset, offset64, 0xFFFFFFFF),
            )
        ):
            raise ValueError("ZIP_INDEX_ZIP64_CONFLICT")
        count, size, offset = count64, size64, offset64
        directory_end = record_offset
    elif disk_count != count or count == 0xFFFF or size == 0xFFFFFFFF or offset == 0xFFFFFFFF:
        raise ValueError("ZIP_INDEX_ZIP64_OR_VOLUME_INVALID")
    if not 1 <= count <= maximum_members or not 0 < size <= maximum_index_bytes:
        raise ValueError("ZIP_INDEX_LIMIT_EXCEEDED")
    if offset + size != directory_end or offset <= 0:
        raise ValueError("ZIP_INDEX_LAYOUT_INVALID")
    if _read_at(source, 0, 4) != b"PK\x03\x04":
        raise ValueError("ZIP_INDEX_PREFIX_FORBIDDEN")
    cursor, actual = offset, 0
    # 不能只相信尾记录的数量声明：每项变长记录也必须完整落在目录边界内。
    while cursor < directory_end:
        actual += 1
        if actual > count or actual > maximum_members:
            raise ValueError("ZIP_INDEX_COUNT_MISMATCH")
        if directory_end - cursor < 46:
            raise ValueError("ZIP_INDEX_TRUNCATED")
        header = _read_at(source, cursor, 46)
        if header[:4] != b"PK\x01\x02":
            raise ValueError("ZIP_INDEX_HEADER_INVALID")
        name_size, extra_size, comment_size, start_disk = struct.unpack_from("<4H", header, 28)
        if start_disk != 0 or name_size == 0:
            raise ValueError("ZIP_INDEX_MEMBER_INVALID")
        cursor += 46 + name_size + extra_size + comment_size
        if cursor > directory_end:
            raise ValueError("ZIP_INDEX_TRUNCATED")
    if actual != count:
        raise ValueError("ZIP_INDEX_COUNT_MISMATCH")
    source.seek(0)
    return actual
