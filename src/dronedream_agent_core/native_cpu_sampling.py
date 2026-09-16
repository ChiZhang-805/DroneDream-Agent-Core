"""Opt-in Linux software sampling of an explicitly owned simulator thread.

No ptrace, signals, suspension, priority changes or kernel/other-process samples.
The bounded diagnostic is not flight qualification and never feeds a controller.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import platform
import struct
import threading
from collections import Counter
from pathlib import Path


# 功能：
#   逐条解析完整 perf 采样记录，保留指令地址和显式丢样数；截断记录不能补成有效样本。
# 输入：
#   buffer：最多一 MiB 的实际字节缓冲。
# 输出：
#   ips：采样记录中的指令地址列表。
#   lost：内核记录报告的丢样总数。
def decode_samples(buffer: bytes) -> tuple[list[int], int]:
    if not isinstance(buffer, (bytes, bytearray)) or len(buffer) > 1024 * 1024:
        raise ValueError("NATIVE_PROFILE_BUFFER_INVALID")
    ips, lost, offset = [], 0, 0
    while offset < len(buffer):
        if len(buffer) - offset < 8:
            raise ValueError("NATIVE_PROFILE_INCOMPLETE_HEADER")
        kind, _, size = struct.unpack_from("IHH", buffer, offset)
        if size < 8 or offset + size > len(buffer):
            raise ValueError("NATIVE_PROFILE_INVALID_RECORD_SIZE")
        if kind == 9:  # PERF_RECORD_SAMPLE: IP followed by PID/TID.
            if size != 24:
                raise ValueError("NATIVE_PROFILE_INVALID_SAMPLE")
            ips.append(struct.unpack_from("Q", buffer, offset + 8)[0])
        elif kind == 2:  # PERF_RECORD_LOST: id followed by lost count.
            if size != 24:
                raise ValueError("NATIVE_PROFILE_INVALID_LOSS_RECORD")
            lost += struct.unpack_from("Q", buffer, offset + 16)[0]
        offset += size
    return ips, lost


# 功能：
#   将最多三十二个热点地址关联到实际可执行映射偏移，不猜测函数名或丢弃未知地址。
# 输入：
#   ips：最多六万五千五百三十六个无符号六十四位采样地址。
#   maps：最多二 Mi 字符的 proc 映射文本。
# 输出：
#   output：按采样次数排序的热点地址、计数及可选镜像偏移列表。
def symbolize_mappings(ips: list[int], maps: str) -> list[dict]:
    if (not isinstance(ips, (list, tuple)) or len(ips) > 65536
            or any(type(ip) is not int or not 0 <= ip < 2**64 for ip in ips)
            or not isinstance(maps, str) or len(maps) > 2 * 1024 * 1024):
        raise ValueError("NATIVE_PROFILE_MAPPINGS_INVALID")
    mappings = []
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or "x" not in fields[1]:
            continue
        try:
            start, end = (int(value, 16) for value in fields[0].split("-"))
            offset = int(fields[2], 16)
        except ValueError as exc:
            raise ValueError("NATIVE_PROFILE_MAPPING_INVALID") from exc
        if not 0 <= start < end <= 2**64 or not 0 <= offset < 2**64:
            raise ValueError("NATIVE_PROFILE_MAPPING_INVALID")
        mappings.append((start, end, offset, fields[5]))
    output = []
    for ip, count in Counter(ips).most_common(32):
        item = {"instruction_pointer": hex(ip), "samples": count}
        for start, end, offset, image in mappings:
            if start <= ip < end:
                item.update(image=image, file_offset=hex(ip - start + offset))
                break
        output.append(item)
    return output


# 功能：
#   对调用方已登记的 Linux x86-64 模拟器线程进行可停止的软件用户态采样，不控制进程。
# 输入：
#   pid：调用方声明拥有的父进程标识。
#   tid：该进程任务目录中的线程标识。
#   stop：线程停止事件，正常采样等待上限为一百毫秒，不承诺端到端操作系统硬期限。
# 输出：
#   result：可用性、采样及丢失计数、镜像偏移，或有限的失败原因。
def sample_owned_thread(pid: int, tid: int, stop) -> dict:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        return {"available": False, "reason": "unsupported-platform"}
    if (type(pid) is not int or type(tid) is not int or not 0 < pid <= 2**31 - 1
            or not 0 < tid <= 2**31 - 1 or not isinstance(stop, threading.Event)):
        raise ValueError("NATIVE_PROFILE_INVALID_PROCESS")
    if not Path(f"/proc/{pid}/task/{tid}").is_dir():
        return {"available": False, "reason": "owned-thread-exited"}
    attr = bytearray(128)
    # Software CPU-clock, IP|TID, exclude kernel/hypervisor. Event starts disabled.
    struct.pack_into("IIQQQQQ", attr, 0, 1, 128, 0, 2_000_000, 3, 0,
                     1 | (1 << 5) | (1 << 6))
    native = (ctypes.c_char * 128).from_buffer(attr)
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.syscall(298, ctypes.byref(native), tid, -1, -1, 0)
    if fd < 0:
        return {"available": False, "reason": "perf-event-unavailable",
                "errno": ctypes.get_errno()}
    try:
        import fcntl

        page = mmap.PAGESIZE
        with mmap.mmap(fd, page * 17, flags=mmap.MAP_SHARED,
                       prot=mmap.PROT_READ | mmap.PROT_WRITE) as ring:
            fcntl.ioctl(fd, 0x2400, 0)  # PERF_EVENT_IOC_ENABLE
            stop.wait(.1)
            fcntl.ioctl(fd, 0x2401, 0)  # PERF_EVENT_IOC_DISABLE; stable ring from here.
            head, tail, data_offset, data_size = struct.unpack_from("QQQQ", ring, 1024)
            if (data_offset != page or data_size != page * 16
                    or head < tail or head - tail > data_size):
                return {"available": False, "reason": "sample-buffer-overrun"}
            start = tail % data_size
            length = head - tail
            first = min(length, data_size - start)
            content = (ring[data_offset + start:data_offset + start + first]
                       + ring[data_offset:data_offset + length - first])
            ips, lost = decode_samples(content)
            with Path(f"/proc/{pid}/maps").open(encoding="utf-8") as stream:
                mappings = stream.read(2 * 1024 * 1024 + 1)
            if len(mappings) > 2 * 1024 * 1024:
                raise ValueError("NATIVE_PROFILE_MAPS_TOO_LARGE")
            result = {"available": True, "tid": tid, "sample_count": len(ips),
                      "lost_samples": lost, "hot_instructions": symbolize_mappings(ips, mappings),
                      "method": "software-user-cpu-ip; 500Hz; at-most-100ms"}
            return result
    except (OSError, ValueError) as error:
        return {"available": False, "reason": type(error).__name__}
    finally:
        os.close(fd)
