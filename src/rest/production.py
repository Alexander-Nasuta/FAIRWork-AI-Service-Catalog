import os

from flaskx_endpoints import import_endpoints
from utils.logger import log


def main() -> None:
    from waitress import serve
    log.info("starting flask app...")
    app = import_endpoints()
    serve(app, host="0.0.0.0", port=8080)


if __name__ == '__main__':
    main()
