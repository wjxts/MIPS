
import logging 

def set_logger_for_local_file(logger: logging.Logger):
    logger.setLevel(logging.INFO)

    # 创建一个日志处理器（此处是 StreamHandler，输出到命令行/控制台）
    console_handler = logging.StreamHandler()

    # 设置处理器的日志级别
    console_handler.setLevel(logging.INFO)

    # 创建一个格式器（Formatter）：决定日志的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 将格式器添加到处理器
    console_handler.setFormatter(formatter)

    # 将处理器添加到 logger
    logger.addHandler(console_handler)
    logger.addHandler(console_handler)
    logger.info("Finish setting logger for local file")
    