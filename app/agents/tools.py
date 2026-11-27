import json
from decimal import Decimal

from langchain.tools import tool
from fastapi.encoders import jsonable_encoder

from app.infra.database import AsyncSessionFactory
from app.modules.product.models import Product
from app.modules.product.service import ProductService


@tool
async def query_candidate_products(
    categories: list[str],
    premium_min: Decimal | None = None,
    limit_per_category: int = 5
):
    """
    根据险种和保费条件查询可用于推荐的候选保险产品。当用户咨询具体保险产品或需要保险产品推荐时使用。

    Args:
        categories: 产品分类列表，可选值为 medical、critical_illness、life、accident。
        premium_min: 最低保费上限，可选参数，只返回最低保费小于该值的产品。
        limit_per_category: 每个险种最多返回的产品数量，可选参数，默认5。
    """

    async with AsyncSessionFactory() as session:
        # 1.初始化Service
        service = ProductService(session)
        # 2.调用Service，得到候选产品
        products: list[Product] = await service.list_candidates(
            categories,
            premium_min,
            limit_per_category
        )
        # 3.返回给AI，最好把Product处理成json返回，json就是字符串
        return jsonable_encoder(products)

@tool
async def retrieve_insurance_clause(
    query: str,
    k: int,
    product_id: int
):
    """查询指定保险产品的条款原文及可追溯来源。

    Args:
        query: 需要从保险条款中查询的问题。
        k: 返回的文档的数量
        product_id: 保险产品ID。
    """

    # 1.导入Retriever
    from app.rag import retriever

    # 2.检索
    # Return compact, ID-addressable source data for both the model and UI.
    sources = await retriever.retrieve(query, k, product_id)
    return json.dumps(sources, ensure_ascii=False)

@tool
def ask_user(stage: str, reason: str) -> str:
    """向人类求助。stage 用于标识当前所处的信息收集阶段。"""
    return ""  # 不会执行，由中间件中断