#pragma once
#include <openssl/evp.h>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace dronedream::cache {
inline constexpr std::uintmax_t MaxFileBytes = 64 * 1024 * 1024;
inline constexpr std::uintmax_t MaxTotalBytes = 256 * 1024 * 1024;
struct FileCloser {
  // 功能：
  //   在成功或异常离开作用域时关闭本次拥有的 C 文件流。
  // 输入：
  //   stream：由当前 unique_ptr 管理的非空文件流。
  // 输出：
  //   void：释放文件描述符，不接管其他路径。
  void operator()(FILE *stream) const noexcept { std::fclose(stream); }
};
// 功能：
//   校验缓存身份摘要的规范表示，不接受截断值或非十六进制文本。
// 输入：
//   s：候选 SHA-256 文本。
// 输出：
//   valid：恰好 64 个小写十六进制字符时为真。
inline bool Digest(const std::string &s) {
  return s.size() == 64 && s.find_first_not_of("0123456789abcdef") == std::string::npos;
}
// 功能：
//   转义 XML 属性中的元字符并拒绝不允许的控制字符，避免诊断文本改写回执结构。
// 输入：
//   s：待写入回执的文本。
// 输出：
//   out：可安全嵌入属性的转义文本。
inline std::string Xml(const std::string &s) {
  std::string out;
  for (unsigned char c : s) {
    if (c == '&') out += "&amp;";
    else if (c == '<') out += "&lt;";
    else if (c == '>') out += "&gt;";
    else if (c == '"') out += "&quot;";
    else if (c == '\'') out += "&apos;";
    else if (c < 32 && c != '\n' && c != '\t' && c != '\r')
      throw std::runtime_error("invalid XML character");
    else out += static_cast<char>(c);
  }
  return out;
}
class Hash {
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> context{EVP_MD_CTX_new(), EVP_MD_CTX_free};
  bool finished = false;
 public:
  // 功能：
  //   初始化独立 SHA-256 上下文，任何 OpenSSL 初始化失败都不能生成伪摘要。
  // 输入：
  //   无。
  // 输出：
  //   当前实例：已初始化的增量摘要对象。
  Hash() {
    if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
      throw std::runtime_error("SHA256 initialization failed");
  }
  // 功能：
  //   累计实际读取字节，结束后的摘要对象不能继续写入。
  // 输入：
  //   bytes：待散列数据起始地址。
  //   count：可读取字节数。
  // 输出：
  //   void：更新当前摘要状态。
  void Update(const char *bytes, std::size_t count) {
    if (finished || (!bytes && count) || EVP_DigestUpdate(context.get(), bytes, count) != 1)
      throw std::runtime_error("SHA256 update failed");
  }
  // 功能：
  //   单次结束摘要并校验输出长度，拒绝重复结束后复用无效上下文。
  // 输入：
  //   无。
  // 输出：
  //   digest：32 字节摘要的小写十六进制表示。
  std::string Finish() {
    unsigned char bytes[EVP_MAX_MD_SIZE];
    unsigned int count = 0;
    if (finished) throw std::runtime_error("SHA256 already finalized");
    finished = true;
    if (EVP_DigestFinal_ex(context.get(), bytes, &count) != 1 || count != 32)
      throw std::runtime_error("SHA256 finalization failed");
    std::ostringstream out;
    for (unsigned int i = 0; i < count; ++i)
      out << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(bytes[i]);
    const auto digest = out.str();
    return digest;
  }
};
// 功能：
//   对内存中已冻结的字节计算摘要，不重新打开可被替换的输入路径。
// 输入：
//   s：冻结输入字节。
// 输出：
//   digest：输入的 SHA-256。
inline std::string Sha(const std::string &s) {
  Hash hash;
  hash.Update(s.data(), s.size());
  const auto digest = hash.Finish();
  return digest;
}
// 功能：
//   检查非空普通缓存文件及尺寸预算；实际加载仍需从同一已打开描述符验证内容。
// 输入：
//   path：缓存文件路径。
// 输出：
//   size：当前文件字节数。
inline std::uintmax_t Size(const std::filesystem::path &path) {
  if (std::filesystem::is_symlink(path) || !std::filesystem::is_regular_file(path))
    throw std::runtime_error("cache must be a regular non-symlink file");
  auto size = std::filesystem::file_size(path);
  if (size == 0 || size > MaxFileBytes) throw std::runtime_error("cache file size invalid");
  return size;
}
// 功能：
//   单次有界读取普通非链接文件；用打开的身份和前后元数据拒绝替换、增长及不完整读取。
// 输入：
//   path：缓存文件路径。
// 输出：
//   content：经过读取一致性校验的独立字节快照。
inline std::string ReadFile(const std::filesystem::path &path) {
  const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (fd < 0) throw std::runtime_error("cache open failed");
  FILE *raw = ::fdopen(fd, "rb");
  if (!raw) { ::close(fd); throw std::runtime_error("cache stream open failed"); }
  std::unique_ptr<FILE, FileCloser> input(raw);
  struct stat before{}, after{}, named{};
  if (::fstat(fd, &before) || !S_ISREG(before.st_mode) || before.st_size <= 0 ||
      static_cast<std::uintmax_t>(before.st_size) > MaxFileBytes)
    throw std::runtime_error("cache file size or type invalid");
  std::string content;
  content.reserve(static_cast<std::size_t>(before.st_size));
  std::array<char, 65536> bytes;
  while (const auto count = std::fread(bytes.data(), 1, bytes.size(), input.get())) {
    if (content.size() + count > static_cast<std::uintmax_t>(before.st_size))
      throw std::runtime_error("cache grew while reading");
    content.append(bytes.data(), count);
  }
  if (std::ferror(input.get()) || content.size() != static_cast<std::uintmax_t>(before.st_size) ||
      ::fstat(fd, &after) || ::lstat(path.c_str(), &named) || !S_ISREG(named.st_mode) ||
      before.st_dev != named.st_dev || before.st_ino != named.st_ino ||
      before.st_size != after.st_size || before.st_mtim.tv_sec != after.st_mtim.tv_sec ||
      before.st_mtim.tv_nsec != after.st_mtim.tv_nsec ||
      before.st_ctim.tv_sec != after.st_ctim.tv_sec || before.st_ctim.tv_nsec != after.st_ctim.tv_nsec)
    throw std::runtime_error("cache changed or read incomplete");
  return content;
}
// 功能：
//   对单次读取的缓存快照散列，拒绝读取过程中的文件漂移。
// 输入：
//   path：缓存路径。
// 输出：
//   digest：快照的 SHA-256。
inline std::string FileSha(const std::filesystem::path &path) {
  const auto digest = Sha(ReadFile(path));
  return digest;
}
// 功能：
//   独占创建完整回执再原子发布，既有目标或暂存属于其他写入者时绝不覆盖。
// 输入：
//   path：当前独立运行目录中的回执目标。
//   text：完整回执文本。
// 输出：
//   void：发布成功无返回数据，冲突或写入错误抛出异常。
inline void WriteNew(const std::filesystem::path &path, const std::string &text) {
  if (text.empty() || text.size() > MaxFileBytes)
    throw std::runtime_error("cache receipt size invalid");
  const auto temporary = path.string() + ".pending";
  const int fd = ::open(temporary.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (fd < 0) throw std::runtime_error("cache receipt temporary unavailable");
  struct stat identity{};
  if (::fstat(fd, &identity)) {
    ::close(fd);
    // 无法证明名称归属时保留暂存，不能误删其他写入者的文件。
    throw std::runtime_error("cache receipt identity unavailable");
  }
  // 功能：
  //   只清理仍指向本次独占创建文件的暂存名，保留随后占用该名称的其他文件。
  // 输入：
  //   无；捕获本次暂存名及打开时的设备和 inode。
  // 输出：
  //   void：本次暂存名存在且归属一致时移除该名称。
  const auto cleanup = [&] {
    struct stat current{};
    if (::lstat(temporary.c_str(), &current) == 0 && current.st_dev == identity.st_dev &&
        current.st_ino == identity.st_ino) ::unlink(temporary.c_str());
  };
  FILE *raw = ::fdopen(fd, "wb");
  if (!raw) {
    cleanup();
    ::close(fd);
    throw std::runtime_error("cache receipt stream open failed");
  }
  std::unique_ptr<FILE, FileCloser> output(raw);
  try {
    if (std::fwrite(text.data(), 1, text.size(), output.get()) != text.size() ||
        std::fflush(output.get()) || ::fsync(fd))
      throw std::runtime_error("cache receipt write failed");
    struct stat current{};
    if (::lstat(temporary.c_str(), &current) || !S_ISREG(current.st_mode) ||
        current.st_dev != identity.st_dev || current.st_ino != identity.st_ino)
      throw std::runtime_error("cache receipt temporary replaced");
    std::filesystem::create_hard_link(temporary, path);
  } catch (...) { cleanup(); throw; }
  cleanup();
}
}
