def test_small_banner():
    from utils.wzl_banner import wzl_banner
    print(wzl_banner)


def test_big_banner():
    from utils.fairwork_banner import banner_color
    from utils.fairwork_banner import banner_sw
    print(banner_sw)
    print(banner_color)


def test_logger():
    from utils.logger import log
    log.info("info msg")
    log.warning("warning msg")
    log.error("error msg")
    log.critical("critical msg")
    log.debug("debug msg")
