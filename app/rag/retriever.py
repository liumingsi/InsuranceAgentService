from langchain_milvus import Milvus
from pymilvus import Function, FunctionType

from app.infra.database import AsyncSessionFactory
from .repository import ParentChunkRepository


def create_cross_encoder_ranker(query: str):
    return Function(
        name="dashscope_semantic_ranker",  # ranker名词，唯一即可
        input_field_names=["text"],  # 原始文档字段
        function_type=FunctionType.RERANK,  # ranker类型，这里是固定值
        params={
            "reranker": "model",  # rerank类型，基于模型rerank，也就是cross-encoder
            "provider": "ali",  # rerank模型提供者
            "model_name": "qwen3.7-text-rerank",  # rerank模型名称
            "queries": [query],  # 查询条件
            "max_client_batch_size": 5,  # 向模型发送请求时的批处理限制
        },
    )


class ParentChunkRetriever:

    def __init__(self, vector_store: Milvus):
        self._vector_store = vector_store

    async def retrieve(self, query: str, k: int, product_id: int):
        # Keep the tool context and source card list compact.
        k = min(max(k, 1), 4)

        # 1.混合检索，召回子块 list[Document]
        child_chunks = await self._vector_store.asimilarity_search(
            query,
            k=k,  # 返回得分最高的K条
            fetch_k=10,  # 初筛时返回的文档的数量
            expr=f"product_id == {product_id}",
            reranker=create_cross_encoder_ranker(query)
        )

        # 2.从子块中拿到parent_id，并且去重
        parent_ids = list(
            dict.fromkeys(
                child.metadata.get("parent_id") for child in child_chunks
            )
        )
        if not parent_ids:
            return []

        # 3.查询父块
        async with AsyncSessionFactory() as session:
            repo = ParentChunkRepository(session)
            parents = await repo.query_by_ids(parent_ids)


        # 4.把parents按照检索回来的顺序排序

        # 4.1.把parents变成一个字典，key是parent_id，值是parent
        parent_id_dict = {
            str(parent.id): parent
            for parent in parents
        }

        # 4.2.遍历parent_ids，逐一找到parent，放到列表中
        parent_chunks = [
            parent_id_dict.get(id)
            for id in parent_ids
        ]

        # Return the matching child excerpt with its parent citation metadata.
        # The parent may contain an entire long section; the child is the short
        # passage that actually matched the query.
        child_by_parent_id = {}
        for child in child_chunks:
            parent_id = child.metadata.get("parent_id")
            if parent_id is not None:
                child_by_parent_id.setdefault(str(parent_id), child)

        results = []
        for parent in parent_chunks:
            if parent is None:
                continue

            section_path = parent.section_path or []
            child = child_by_parent_id.get(str(parent.id))
            excerpt = child.page_content if child is not None else parent.content

            # Child chunks include the section heading for retrieval context.
            # Keep the heading as metadata and return only the source passage.
            section_header = "\n".join(section_path)
            if section_header and excerpt.startswith(section_header):
                excerpt = excerpt[len(section_header):].lstrip("\n")

            results.append(
                {
                    "citation_id": f"S{len(results) + 1}",
                    "clause_name": parent.clause_name,
                    "section_path": section_path,
                    "source_excerpt": excerpt,
                }
            )

        return results


