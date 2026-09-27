"""Explicit simulation-only payload measurement modes, never flight admission."""


# 功能：
#   1. 限定负载测量使用已接纳研究模型或明确的确定性教师，两种控制权限不能混用。
#   2. 教师必须记录原始学习观测；所有测量均要求多模态记录，不能带正式飞行资格。
# 输入：
#   provider：当前局部控制提供方；teacher、recording：教师权限与原始观测记录开关。
#   model_authority、model_packages、admission、qualification：模型控制权限及证据是否存在。
#   multimodal、fallback：是否记录多模态数据以及是否配置其他回退控制方。
# 输出：
#   mode：通过契约检查的教师或研究模型测量模式名称。
def payload_collection_mode(*, provider, teacher, recording, model_authority, model_packages,
                            admission, qualification, multimodal, fallback):
    flags = (teacher, recording, model_authority, model_packages, admission,
             qualification, multimodal, fallback)
    if any(type(flag) is not bool for flag in flags):
        raise ValueError('PAYLOAD_COLLECTION_FLAGS_INVALID')
    if not multimodal:
        raise ValueError('development payload collection requires multimodal recording')
    if qualification:
        raise ValueError('development payload collection requires simulation admission only')
    if teacher:
        if (provider is not None or not recording or model_authority or model_packages
                or admission or fallback):
            raise ValueError('PAYLOAD_COLLECTION_TEACHER_AUTHORITY_MIXED')
        mode = 'recorded-simulation-teacher'
    else:
        if provider != 'local-policy':
            raise ValueError('development payload collection requires local-policy')
        if not admission or not model_packages or not model_authority or fallback:
            raise ValueError('PAYLOAD_COLLECTION_MODEL_AUTHORITY_INCOMPLETE')
        mode = 'simulation-admitted-model'
    return mode
