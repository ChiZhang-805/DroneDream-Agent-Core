#!/usr/bin/env python3
"""Read the actual EGL/OpenGL renderer in a short-lived diagnostic process.

No world, vehicle, model weights or runtime installation is changed. Run with
the same display/driver environment as a proposed Gazebo server. A renderer
name is capability evidence, not a measured sensor latency qualification.
"""

import argparse
import ctypes as c
import json


def probe(*, gl_major: int = 4, gl_minor: int = 3) -> dict:
    if (gl_major, gl_minor) not in {(3, 3), (4, 3), (4, 5)}:
        raise ValueError("unsupported diagnostic OpenGL version")
    egl = c.CDLL("libEGL.so.1")
    gl = c.CDLL("libGL.so.1")
    pointer, integer = c.c_void_p, c.c_int
    signatures = {
        "eglGetDisplay": (pointer, [pointer]),
        "eglInitialize": (integer, [pointer, c.POINTER(integer), c.POINTER(integer)]),
        "eglChooseConfig": (integer, [pointer, c.POINTER(integer), c.POINTER(pointer),
                                      integer, c.POINTER(integer)]),
        "eglBindAPI": (integer, [c.c_uint]),
        "eglCreatePbufferSurface": (pointer, [pointer, pointer, c.POINTER(integer)]),
        "eglCreateContext": (pointer, [pointer, pointer, pointer, c.POINTER(integer)]),
        "eglMakeCurrent": (integer, [pointer, pointer, pointer, pointer]),
        "eglDestroyContext": (integer, [pointer, pointer]),
        "eglDestroySurface": (integer, [pointer, pointer]),
        "eglTerminate": (integer, [pointer]),
        "eglGetError": (c.c_uint, []),
    }
    for name, (result, arguments) in signatures.items():
        method = getattr(egl, name)
        method.restype, method.argtypes = result, arguments

    def checked(value, operation):
        if not value:
            raise RuntimeError(f"{operation}:EGL_ERROR_{egl.eglGetError():04x}")
        return value

    display = checked(egl.eglGetDisplay(None), "get-display")
    major, minor = integer(), integer()
    checked(egl.eglInitialize(display, c.byref(major), c.byref(minor)), "initialize")
    context = surface = None
    try:
        # EGL_SURFACE_TYPE=PBUFFER_BIT, RENDERABLE_TYPE=OPENGL_BIT, RGB8.
        attributes = (integer * 11)(0x3033, 1, 0x3040, 8, 0x3024, 8,
                                    0x3023, 8, 0x3022, 8, 0x3038)
        config, count = pointer(), integer()
        checked(egl.eglChooseConfig(display, attributes, c.byref(config), 1,
                                    c.byref(count)), "choose-config")
        checked(count.value, "config-count")
        checked(egl.eglBindAPI(0x30A2), "bind-opengl")
        size = (integer * 5)(0x3057, 1, 0x3056, 1, 0x3038)
        surface = checked(egl.eglCreatePbufferSurface(display, config, size), "pbuffer")
        # Request the stated capability; never set a Mesa version override.
        version = (integer * 7)(0x3098, gl_major, 0x30FB, gl_minor, 0x30FD, 1, 0x3038)
        context = checked(egl.eglCreateContext(display, config, None, version), "context")
        checked(egl.eglMakeCurrent(display, surface, surface, context), "make-current")
        gl.glGetString.restype, gl.glGetString.argtypes = c.c_char_p, [c.c_uint]
        result = {name: (gl.glGetString(code) or b"").decode("utf-8", "replace")
                  for name, code in (("vendor", 0x1F00), ("renderer", 0x1F01),
                                     ("version", 0x1F02))}
        if not all(result.values()):
            raise RuntimeError("OPENGL_IDENTITY_UNAVAILABLE")
        return {"egl_version": f"{major.value}.{minor.value}", **result}
    finally:
        egl.eglMakeCurrent(display, None, None, None)
        if context:
            egl.eglDestroyContext(display, context)
        if surface:
            egl.eglDestroySurface(display, surface)
        egl.eglTerminate(display)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gl-version", choices=("3.3", "4.3", "4.5"), default="4.3")
    version = tuple(int(value) for value in parser.parse_args().gl_version.split("."))
    print(json.dumps(probe(gl_major=version[0], gl_minor=version[1]), sort_keys=True))
