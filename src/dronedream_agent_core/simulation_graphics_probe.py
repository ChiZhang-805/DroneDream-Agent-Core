"""Read the actual renderer of a disposable EGL context; no flight or driver installation."""
import ctypes as c
import json


# 功能：
#   创建独立 EGL 上下文并读取实际 GL_RENDERER，最后释放资源。
# 输入：
#   当前进程显式渲染环境。
# 输出：
#   JSON 渲染器名称；不可用时异常退出。
def main():
    egl = c.CDLL('libEGL.so.1')
    gl = c.CDLL('libGL.so.1')
    signatures = {
        'eglGetDisplay': (c.c_void_p, [c.c_void_p]),
        'eglInitialize': (c.c_uint, [c.c_void_p, c.POINTER(c.c_int), c.POINTER(c.c_int)]),
        'eglBindAPI': (c.c_uint, [c.c_uint]),
        'eglChooseConfig': (c.c_uint, [c.c_void_p, c.POINTER(c.c_int), c.POINTER(c.c_void_p), c.c_int, c.POINTER(c.c_int)]),
        'eglCreateContext': (c.c_void_p, [c.c_void_p, c.c_void_p, c.c_void_p, c.POINTER(c.c_int)]),
        'eglMakeCurrent': (c.c_uint, [c.c_void_p, c.c_void_p, c.c_void_p, c.c_void_p]),
        'eglDestroyContext': (c.c_uint, [c.c_void_p, c.c_void_p]),
        'eglTerminate': (c.c_uint, [c.c_void_p]),
    }
    for name, (restype, argtypes) in signatures.items():
        getattr(egl, name).restype, getattr(egl, name).argtypes = restype, argtypes
    display = egl.eglGetDisplay(None)
    major, minor = c.c_int(), c.c_int()
    assert egl.eglInitialize(display, c.byref(major), c.byref(minor)), 'EGL_INITIALIZE_FAILED'
    context = None
    try:
        assert egl.eglBindAPI(0x30A2)
        config, count = c.c_void_p(), c.c_int()
        attrs = (c.c_int * 5)(0x3033, 1, 0x3040, 8, 0x3038)
        assert egl.eglChooseConfig(display, attrs, c.byref(config), 1, c.byref(count)) and count.value
        context = egl.eglCreateContext(display, config, None, (c.c_int * 1)(0x3038))
        assert context and egl.eglMakeCurrent(display, None, None, context), 'EGL_CONTEXT_FAILED'
        gl.glGetString.restype, gl.glGetString.argtypes = c.c_char_p, [c.c_uint]
        print(json.dumps({'renderer': gl.glGetString(0x1F01).decode(), 'vendor': gl.glGetString(0x1F00).decode()}), flush=True)
    finally:
        egl.eglMakeCurrent(display, None, None, None)
        if context:
            egl.eglDestroyContext(display, context)
        egl.eglTerminate(display)


if __name__ == '__main__':
    main()
