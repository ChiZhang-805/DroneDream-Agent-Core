import struct

import pytest

from dronedream_agent_core.native_cpu_sampling import decode_samples, symbolize_mappings


# 功能：
#   解析完整采样与丢样记录，重复地址保留真实采样次数，丢失数量单独报告。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_software_profile_preserves_ip_samples_and_explicit_losses():
    sample = struct.pack("IHHQII", 9, 0, 24, 0x1034, 1, 2)
    loss = struct.pack("IHHQQ", 2, 0, 24, 99, 4)
    assert decode_samples(sample + sample + loss) == ([0x1034, 0x1034], 4)


# 功能：
#   截断头部、错误记录长度和不完整丢样记录都必须拒绝，不能补造缺失信息。
# 输入：
#   raw：非法的采样字节记录。
# 输出：
#   None：不返回业务数据。
@pytest.mark.parametrize("raw", [b"short", struct.pack("IHH", 9, 0, 4),
    struct.pack("IHHQ", 9, 0, 16, 0), struct.pack("IHH", 2, 0, 8)])
def test_profile_does_not_invent_missing_records(raw):
    with pytest.raises(ValueError, match="NATIVE_PROFILE_"):
        decode_samples(raw)


# 功能：
#   热点偏移由实际可执行映射推导，带空格的镜像名保持完整，非执行区地址仍标为未知。
# 输入：
#   无。
# 输出：
#   None：不返回业务数据。
def test_instruction_offsets_use_actual_executable_mapping_and_keep_unknown_ips():
    result = symbolize_mappings([0x1034, 0x1034, 0x9999],
        "1000-2000 r-xp 00000300 00:01 3 /lib/example with spaces.so\n"
        "9000-a000 rw-p 00000000 00:00 0 [heap]\n")
    assert result[0] == {"instruction_pointer": "0x1034", "samples": 2,
                         "image": "/lib/example with spaces.so", "file_offset": "0x334"}
    assert result[1] == {"instruction_pointer": "0x9999", "samples": 1}
