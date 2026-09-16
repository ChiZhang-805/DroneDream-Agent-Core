"""Packaged sidecar entry: dispatch isolated previews before importing the HTTP app."""

import sys


# 功能：
#   1. 精确匹配 PDF 工作者参数时，仅启动隔离解析入口，避免递归启动 HTTP 服务。
#   2. 其他调用交给正常服务入口；延迟导入避免预览子进程初始化整个应用。
# 输入：
#   无显式参数；sys.argv 提供打包程序的命令行参数。
# 输出：
#   None：不返回业务数据。
def main() -> None:
    if sys.argv[1:] == ["--attachment-pdf-worker"]:
        from dronedream_agent_plugins._attachment_pdf_worker import main as worker_main

        worker_main()
    else:
        from dronedream_agent_app.server import main as server_main

        server_main()


if __name__ == "__main__":
    main()
