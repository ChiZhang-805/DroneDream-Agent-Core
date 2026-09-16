#include "cache_contract.hpp"
#include <iostream>
// 功能：
//   测试摘要生命周期、XML 转义及独占缓存回执发布，不把基础契约测试说成视觉验收。
// 输入：
//   无；临时文件仅创建在 CTest 的构建工作目录。
// 输出：
//   exit_code：全部契约通过为 0，非零指明失败检查。
int main() {
  using namespace dronedream::cache;
  if (Sha("abc") != "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" ||
      !Digest(Sha("abc")) || Digest(std::string(64, 'z')) || Digest("abc") ||
      Xml("<&\"'>") != "&lt;&amp;&quot;&apos;&gt;") return 1;
  bool rejected = false;
  try { Xml(std::string(1, '\x01')); } catch (const std::runtime_error &) { rejected = true; }
  if (!rejected) return 2;
  Hash hash;
  hash.Update("abc", 3);
  hash.Finish();
  rejected = false;
  try { hash.Finish(); } catch (const std::runtime_error &) { rejected = true; }
  if (!rejected) return 3;
  rejected = false;
  try { hash.Update("x", 1); } catch (const std::runtime_error &) { rejected = true; }
  if (!rejected) return 4;
  char pattern[] = "cache-test-XXXXXX";
  const auto created = ::mkdtemp(pattern);
  if (!created) return 5;
  const std::filesystem::path directory(created), receipt = directory / "ready.xml";
  WriteNew(receipt, "first");
  rejected = false;
  try { WriteNew(receipt, "second"); } catch (const std::exception &) { rejected = true; }
  if (!rejected || ReadFile(receipt) != "first" || std::filesystem::exists(receipt.string() + ".pending")) return 6;
  const auto blocked = directory / "blocked.xml";
  std::filesystem::create_symlink("unowned", blocked.string() + ".pending");
  rejected = false;
  try { WriteNew(blocked, "payload"); } catch (const std::exception &) { rejected = true; }
  if (!rejected || !std::filesystem::is_symlink(blocked.string() + ".pending") ||
      std::filesystem::exists(directory / "unowned")) return 7;
  rejected = false;
  std::filesystem::create_symlink("ready.xml", directory / "linked");
  try { ReadFile(directory / "linked"); } catch (const std::exception &) { rejected = true; }
  if (!rejected) return 8;
  // 仅删除本测试刚创建的具体文件和空目录，不递归清理构建或缓存目录。
  std::filesystem::remove(directory / "linked");
  std::filesystem::remove(blocked.string() + ".pending");
  std::filesystem::remove(receipt);
  std::filesystem::remove(directory);
  std::cout << "cache primitive contract passed; not a rendering qualification\n";
  const int exit_code = 0;
  return exit_code;
}
