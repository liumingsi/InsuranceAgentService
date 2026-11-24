from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# 1.创建checkpointer专用连接池
checkpoint_pool = AsyncConnectionPool(
    conninfo=settings.db.checkpoint_url,
    min_size=1,
    max_size=5,
    max_idle=300.0,
    timeout=30.0,                                   # 取连接超时
    check=AsyncConnectionPool.check_connection,     # 取连接前健康检查 ★关键
    kwargs={
        "autocommit": True,
        "prepare_threshold": None,                  # 真正禁用预准备（原 0 是反的）
        "row_factory": dict_row,
        "keepalives": 1,                            # TCP 保活 ★关键
        "keepalives_idle": 60,
        "keepalives_interval": 10,
        "keepalives_count": 5,
    },
    open=False,
)

# 2.初始化连接
async def init_checkpointer() -> AsyncPostgresSaver:
    """初始化checkpointer"""

    # 1.初始化连接池
    await checkpoint_pool.open()  # 初始化连接
    await checkpoint_pool.wait()  # 等待连接池就绪

    # 2.创建checkpointer
    checkpointer = AsyncPostgresSaver(checkpoint_pool)
    await checkpointer.setup()
    logger.info("Checkpointer初始化成功~✅️")
    return checkpointer


# 3.关闭连接
async def close_checkpointer() -> None:
    """关闭Checkpointer连接池"""
    await checkpoint_pool.close()
    logger.info("Checkpointer连接池已关闭")