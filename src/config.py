import logging

format_log = ("%(asctime)s | %(levelname)s | %(message)s")
logging.basicConfig(
    format = format_log,
    level = logging.INFO,
    datefmt = "%Y-%m-%dT%H:%M:%S"
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)