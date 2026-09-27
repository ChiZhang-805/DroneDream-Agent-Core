// Exact bounded occupancy scan reduction. No device, filesystem or flight authority.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using Key = std::array<int64_t, 3>;
struct Hash {
    // 功能：
    //   计算哈希桶位置；完整三轴相等检查仍决定是否为同一体素。
    // 输入：
    //   key：三轴整数键。
    // 输出：
    //   hash：无符号哈希值。
    size_t operator()(const Key& key) const noexcept {
        uint64_t hash = 1469598103934665603ULL;
        for (auto value : key) { hash ^= static_cast<uint64_t>(value); hash *= 1099511628211ULL; }
        return static_cast<size_t>(hash);
    }
};
struct Ray { std::array<double, 3> origin, endpoint; int steps; double strength; bool hit; };
struct Entry { Key key; double strength; };
using Index = std::unordered_map<Key, size_t, Hash>;
constexpr size_t LIMIT = 2000000;

// 功能：
//   按首次出现顺序记录最大证据，只改变调用私有容器；哈希碰撞不会合并不同键。
// 输入：
//   key、strength：已有体素及强度；index、entries：本次调用私有容器。
// 输出：
//   无。
static void merge(const Key& key, double strength, Index& index, std::vector<Entry>& entries) {
    auto found = index.find(key);
    if (found == index.end()) {
        if (entries.size() >= LIMIT) throw std::runtime_error("METRIC_SCAN_EVIDENCE_BUDGET_EXCEEDED");
        index.emplace(key, entries.size());
        entries.push_back({key, strength});
    } else if (strength > entries[found->second].strength) {
        entries[found->second].strength = strength;
    }
}

// 功能：
//   复制并严格验证有限三元组，拒绝隐式对象转换；失败时保留 Python 转换异常。
// 输入：
//   value：Python 三元组；result：接收独立 double 数组的引用。
// 输出：
//   valid：三轴是否全部有效。
static bool triple(PyObject* value, std::array<double, 3>& result) {
    if (!PyTuple_CheckExact(value) || PyTuple_GET_SIZE(value) != 3) return false;
    for (int i = 0; i < 3; ++i) {
        auto* item = PyTuple_GET_ITEM(value, i);
        if (!PyFloat_CheckExact(item) && !PyLong_CheckExact(item)) return false;
        result[i] = PyFloat_AsDouble(item);
        if (PyErr_Occurred() || !std::isfinite(result[i])) return false;
    }
    return true;
}

// 功能：
//   生成最终去重后的 Python 证据元组，命中键从自由空间中移除；分配失败返回 nullptr。
// 输入：
//   entries：首次出现顺序的证据；excluded：需要排除的命中索引。
// 输出：
//   result：键与强度组成的元组。
static PyObject* pack(const std::vector<Entry>& entries, const Index* excluded) {
    size_t count = 0;
    for (const auto& entry : entries) if (!excluded || !excluded->count(entry.key)) ++count;
    PyObject* result = PyTuple_New(static_cast<Py_ssize_t>(count));
    if (!result) return nullptr;
    size_t offset = 0;
    for (const auto& entry : entries) {
        if (excluded && excluded->count(entry.key)) continue;
        PyObject* item = Py_BuildValue("((LLL)d)", static_cast<long long>(entry.key[0]),
            static_cast<long long>(entry.key[1]), static_cast<long long>(entry.key[2]), entry.strength);
        if (!item) { Py_DECREF(result); return nullptr; }
        // 这两层是本函数新建的精确 tuple，只包含精确 int/float，不可能形成引用环。
        // 在下一次分配前取消环扫描，避免逐帧数万纯值对象反复触发无用的循环遍历。
        // 不关闭全局 GC，不处理调用者容器，也不取消任何可变地图或用户对象的跟踪。
        auto* key = PyTuple_GET_ITEM(item, 0);
        if (PyObject_GC_IsTracked(key)) PyObject_GC_UnTrack(key);
        if (PyObject_GC_IsTracked(item)) PyObject_GC_UnTrack(item);
        PyTuple_SET_ITEM(result, static_cast<Py_ssize_t>(offset++), item);
    }
    if (PyObject_GC_IsTracked(result)) PyObject_GC_UnTrack(result);
    return result;
}

// 功能：
//   有界采样整帧并在本地内存归约最大占用/自由证据；不修改地图，错误不返回部分更新。
// 输入：
//   args：冻结射线列表、地图最小坐标、正分辨率。
// 输出：
//   result：命中及自由证据二元组。
static PyObject* integrate(PyObject*, PyObject* args) {
    PyObject *raw, *raw_minimum, *raw_resolution;
    if (!PyArg_ParseTuple(args, "OOO", &raw, &raw_minimum, &raw_resolution)) return nullptr;
    // 不接受 bool 或执行用户对象的 __float__，确保复制期间不会被隐式回调改写射线。
    if (!PyFloat_CheckExact(raw_resolution) && !PyLong_CheckExact(raw_resolution)) {
        PyErr_SetString(PyExc_ValueError, "METRIC_NATIVE_RESOLUTION_INVALID");
        return nullptr;
    }
    const double resolution = PyFloat_AsDouble(raw_resolution);
    if (PyErr_Occurred()) return nullptr;
    std::array<double, 3> minimum;
    if (!PyList_CheckExact(raw) || PyList_GET_SIZE(raw) < 1 || PyList_GET_SIZE(raw) > 250000
        || !triple(raw_minimum, minimum) || !std::isfinite(resolution) || resolution <= 0.) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "METRIC_NATIVE_INPUT_INVALID");
        return nullptr;
    }
    std::vector<Ray> rays;
    Index occupied, free;
    std::vector<Entry> hits, frees;
    try {
        rays.reserve(static_cast<size_t>(PyList_GET_SIZE(raw)));
        size_t work = 0;
        for (Py_ssize_t i = 0; i < PyList_GET_SIZE(raw); ++i) {
            auto* row = PyList_GET_ITEM(raw, i);
            Ray ray;
            if (!PyTuple_CheckExact(row) || PyTuple_GET_SIZE(row) != 5
                || !triple(PyTuple_GET_ITEM(row, 0), ray.origin)
                || !triple(PyTuple_GET_ITEM(row, 1), ray.endpoint)
                || !PyLong_CheckExact(PyTuple_GET_ITEM(row, 2))
                || !PyFloat_CheckExact(PyTuple_GET_ITEM(row, 3))
                || !PyBool_Check(PyTuple_GET_ITEM(row, 4)))
                throw std::runtime_error("METRIC_NATIVE_RAY_INVALID");
            long steps = PyLong_AsLong(PyTuple_GET_ITEM(row, 2));
            if (PyErr_Occurred() || steps < 1 || steps > static_cast<long>(LIMIT))
                throw std::runtime_error("METRIC_SCAN_TRAVERSAL_BUDGET_EXCEEDED");
            ray.steps = static_cast<int>(steps);
            ray.strength = PyFloat_AS_DOUBLE(PyTuple_GET_ITEM(row, 3));
            if (!std::isfinite(ray.strength) || ray.strength < 0. || ray.strength > 5.)
                throw std::runtime_error("METRIC_NATIVE_STRENGTH_INVALID");
            ray.hit = PyTuple_GET_ITEM(row, 4) == Py_True;
            work += static_cast<size_t>(steps) + 1;
            if (work > LIMIT) throw std::runtime_error("METRIC_SCAN_TRAVERSAL_BUDGET_EXCEEDED");
            rays.push_back(ray);
        }
    } catch (const std::bad_alloc&) { return PyErr_NoMemory(); }
      catch (const std::exception& error) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, error.what());
        return nullptr;
    }
    // 所有 Python 输入已复制。纯计算释放 GIL，不阻塞同时进行的安全发布和传感器接收。
    const char* error = nullptr;
    Py_BEGIN_ALLOW_THREADS
    try {
        const double inverse = 1. / resolution;
        for (const auto& ray : rays) {
            Key previous{};
            bool has_previous = false;
            for (int step = 0; step <= ray.steps; ++step) {
                const double ratio = static_cast<double>(step) / ray.steps;
                Key key;
                for (int axis = 0; axis < 3; ++axis) {
                    // 编译禁用 FMA/fast-math，保持原 NumPy 分步乘加、减法及倒数乘法。
                    double value = (ray.endpoint[axis] - ray.origin[axis]) * ratio;
                    value += ray.origin[axis];
                    value -= minimum[axis];
                    value *= inverse;
                    if (!std::isfinite(value) || value <= -4611686018427387904.0 || value >= 4611686018427387904.0)
                        throw std::runtime_error("METRIC_NATIVE_COORDINATE_INVALID");
                    key[axis] = static_cast<int64_t>(std::floor(value));
                }
                if (!has_previous || key != previous) {
                    // 延后一格，最后的去重键只有在非命中时才能作为自由空间。
                    if (has_previous) merge(previous, ray.strength, free, frees);
                    previous = key;
                    has_previous = true;
                }
            }
            if (ray.hit) merge(previous, ray.strength, occupied, hits);
            else merge(previous, ray.strength, free, frees);
            if (hits.size() + frees.size() > LIMIT)
                throw std::runtime_error("METRIC_SCAN_EVIDENCE_BUDGET_EXCEEDED");
        }
    } catch (const std::bad_alloc&) { error = "METRIC_NATIVE_ALLOCATION_FAILED"; }
      catch (const std::exception&) { error = "METRIC_NATIVE_REDUCTION_FAILED"; }
    Py_END_ALLOW_THREADS
    if (error) { PyErr_SetString(PyExc_ValueError, error); return nullptr; }
    PyObject* first = pack(hits, nullptr);
    if (!first) return nullptr;
    PyObject* second = pack(frees, &occupied);
    if (!second) { Py_DECREF(first); return nullptr; }
    PyObject* result = PyTuple_Pack(2, first, second);
    Py_DECREF(first); Py_DECREF(second);
    // 两个子元组已完全初始化为无环纯值图；返回值仍由正常引用计数释放。
    if (result && PyObject_GC_IsTracked(result)) PyObject_GC_UnTrack(result);
    return result;
}

// 功能：检查有界精确整数，拒绝 bool 和隐式数值转换；输入：Python 值及界；输出：整数。
static bool bounded_integer(PyObject* value, long low, long high, long& result) {
    if (!PyLong_CheckExact(value)) return false;
    result = PyLong_AsLong(value);
    return !PyErr_Occurred() && low <= result && result <= high;
}

struct DepthTile { long row = -1, col = -1; double distance = 0.; bool hit = false; };

// 功能：单次释放 GIL 后检查所有原始深度像素，保持最近径向命中及行优先并列规则。
// 输入：不可变小端 float32 字节、像素布局、内参、量程和明确的无回波语义。
// 输出：原始像素位置/距离/命中元组及有效像素数；不补洞、不更改时间、不提供飞行许可。
static PyObject* project_depth(PyObject*, PyObject* args) {
    if (!PyTuple_CheckExact(args) || PyTuple_GET_SIZE(args) != 9) {
        PyErr_SetString(PyExc_ValueError, "DEPTH_NATIVE_ARGUMENTS_INVALID"); return nullptr;
    }
    auto* data = PyTuple_GET_ITEM(args, 0);
    auto* optics = PyTuple_GET_ITEM(args, 5);
    auto* mode = PyTuple_GET_ITEM(args, 8);
    long width, height, step, stride;
    double intrinsics[4], minimum, maximum;
    if (!PyBytes_CheckExact(data)
        || !bounded_integer(PyTuple_GET_ITEM(args, 1), 2, 8192, width)
        || !bounded_integer(PyTuple_GET_ITEM(args, 2), 2, 8192, height)
        || !bounded_integer(PyTuple_GET_ITEM(args, 3), 8, 98304, step)
        || !bounded_integer(PyTuple_GET_ITEM(args, 4), 1, 8192, stride)
        || !PyTuple_CheckExact(optics) || PyTuple_GET_SIZE(optics) != 4 || !PyBool_Check(mode)) {
        if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "DEPTH_NATIVE_LAYOUT_INVALID");
        return nullptr;
    }
    for (int i = 0; i < 6; ++i) {
        auto* value = i < 4 ? PyTuple_GET_ITEM(optics, i) : PyTuple_GET_ITEM(args, i + 2);
        if (!PyFloat_CheckExact(value) && !PyLong_CheckExact(value)) {
            PyErr_SetString(PyExc_ValueError, "DEPTH_NATIVE_CALIBRATION_INVALID"); return nullptr;
        }
        double parsed = PyFloat_AsDouble(value);
        if (PyErr_Occurred() || !std::isfinite(parsed)) {
            if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "DEPTH_NATIVE_CALIBRATION_INVALID");
            return nullptr;
        }
        if (i < 4) intrinsics[i] = parsed;
        else if (i == 4) minimum = parsed;
        else maximum = parsed;
    }
    const long rows = (height + stride - 1) / stride, cols = (width + stride - 1) / stride;
    if (width * height > 4194304 || rows * cols > 250000 || step < width * 4
        || step > width * 4 + 65536 || static_cast<int64_t>(step) * height > 33554432
        || PyBytes_GET_SIZE(data) != static_cast<int64_t>(step) * height
        || intrinsics[0] <= 0. || intrinsics[1] <= 0.
        || intrinsics[2] < 0. || intrinsics[2] >= width
        || intrinsics[3] < 0. || intrinsics[3] >= height
        || minimum < 0. || maximum <= minimum || maximum > 1000.) {
        PyErr_SetString(PyExc_ValueError, "DEPTH_NATIVE_CALIBRATION_INVALID"); return nullptr;
    }
    double hy = std::max(intrinsics[2], width - 1 - intrinsics[2]) / intrinsics[0];
    double vz = std::max(intrinsics[3], height - 1 - intrinsics[3]) / intrinsics[1];
    if (!std::isfinite(maximum * maximum * (1. + hy * hy + vz * vz))) {
        PyErr_SetString(PyExc_ValueError, "DEPTH_NATIVE_GEOMETRY_OVERFLOW"); return nullptr;
    }
    const auto* bytes = reinterpret_cast<const unsigned char*>(PyBytes_AS_STRING(data));
    const bool far_clip = mode == Py_True;
    std::vector<DepthTile> tiles;
    long valid_count = 0;
    bool allocation_failed = false;
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559);
    Py_BEGIN_ALLOW_THREADS
    try {
        tiles.reserve(static_cast<size_t>(rows * cols));
        for (long top = 0; top < height; top += stride) {
            for (long left = 0; left < width; left += stride) {
                DepthTile tile, free;
                double best = std::numeric_limits<double>::infinity();
                long free_cost = std::numeric_limits<long>::max();
                for (long row = top; row < std::min(height, top + stride); ++row) {
                    const double z = (intrinsics[3] - row) / intrinsics[1];
                    for (long col = left; col < std::min(width, left + stride); ++col) {
                        const auto* pixel = bytes + static_cast<size_t>(row) * step + col * 4;
                        uint32_t bits = uint32_t(pixel[0]) | (uint32_t(pixel[1]) << 8)
                            | (uint32_t(pixel[2]) << 16) | (uint32_t(pixel[3]) << 24);
                        float axial;
                        std::memcpy(&axial, &bits, sizeof(axial));
                        const double depth = axial;
                        const bool finite = std::isfinite(depth);
                        bool valid = finite && depth >= minimum && depth <= maximum + 1e-5;
                        if (far_clip) valid = valid || depth == std::numeric_limits<double>::infinity();
                        else valid = valid && depth < maximum - 1e-5;
                        if (!valid) continue;
                        ++valid_count;
                        const double y = (intrinsics[2] - col) / intrinsics[0];
                        const double measured = finite ? std::min(depth, maximum) : 0.;
                        const double radial = measured * measured * (1. + y * y + z * z);
                        if (finite && radial < maximum * maximum && depth < maximum - 1e-5
                            && radial < best) {
                            best = radial;
                            tile = {row, col, std::sqrt(radial), true};
                        }
                        const long cost = std::abs(row % stride - stride / 2)
                            + std::abs(col % stride - stride / 2);
                        if (cost < free_cost) {
                            free_cost = cost;
                            free = {row, col, maximum, false};
                        }
                    }
                }
                if (!tile.hit) tile = free;
                if (tile.row >= 0) tiles.push_back(tile);
            }
        }
    } catch (const std::bad_alloc&) { allocation_failed = true; }
    Py_END_ALLOW_THREADS
    if (allocation_failed) return PyErr_NoMemory();
    PyObject* samples = PyTuple_New(static_cast<Py_ssize_t>(tiles.size()));
    if (!samples) return nullptr;
    for (size_t i = 0; i < tiles.size(); ++i) {
        const auto& tile = tiles[i];
        auto* sample = Py_BuildValue("lldO", tile.row, tile.col, tile.distance,
                                    tile.hit ? Py_True : Py_False);
        if (!sample) { Py_DECREF(samples); return nullptr; }
        PyTuple_SET_ITEM(samples, static_cast<Py_ssize_t>(i), sample);
    }
    return Py_BuildValue("Nl", samples, valid_count);
}

static PyMethodDef methods[] = {
    {"integrate", integrate, METH_VARARGS, "Exact bounded scan reduction."},
    {"project_depth", project_depth, METH_VARARGS, "All-pixel nearest-radial depth reduction."},
    {nullptr, nullptr, 0, nullptr}};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_dronedream_metric_scan", nullptr, -1, methods};

// 功能：
//   注册无外部状态的本地计算模块，声明固定的输入输出协议。
// 输入：
//   无。
// 输出：
//   result：扩展模块或初始化失败状态。
PyMODINIT_FUNC PyInit__dronedream_metric_scan() {
    PyObject* result = PyModule_Create(&module);
    if (result && PyModule_AddStringConstant(result, "CONTRACT", "exact-scan-max-hit-first-v1") < 0) {
        Py_DECREF(result); return nullptr;
    }
    if (result && PyModule_AddStringConstant(result, "DEPTH_CONTRACT", "all-pixels-nearest-radial-v1") < 0) {
        Py_DECREF(result); return nullptr;
    }
    return result;
}
