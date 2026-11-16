"""Example agent loop; connect a real model and the Day 6 retriever via adapters.

Adapters are intentionally not provided: Day 6 has no fixed database schema and
model providers differ. Never use fake adapters to report benchmark quality.
"""
from typing import Any, Protocol
from baseline import pack_context


class Controller(Protocol):
    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        """Call a real model with a search tool; return normalized action.

        Search: {'action': 'search', 'query': '...'}.
        Final: {'action': 'answer', 'answer': '...', 'citations': ['chunk-id']}.
        Citations must also appear as [chunk-id] in the answer text.
        """
        ...


class Retriever(Protocol):
    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        """Query the real Day 6 vector store with a compatible query embedding.

        Return each chunk's id, text, source_id, source, page, metadata and score.
        The query embedding model and dimension must match the document index.
        """
        ...


def run_agent(question: str, controller: Controller, retriever: Retriever, *,
              filters: dict[str, Any] | None = None, max_searches: int = 4,
              top_k: int = 3, budget: int = 800,
              context_packer: Any = None,
              all_chunks_by_id: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Return answer, citation IDs and trace; no benchmark gold enters this loop.

    A final answer without valid citations, an empty context, and a search limit
    lead to abstention. This only checks citation IDs, not semantic grounding.
    Provider/retriever errors propagate; no silent mock fallback.
    """
    if not question.strip() or max_searches < 1 or top_k < 1 or budget < 1:
        raise ValueError('question, max_searches, top_k hoặc budget không hợp lệ')
    history: list[dict[str, Any]] = []
    chunks: dict[str, dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for _ in range(max_searches + 1):
        action = controller.decide(question, history)
        if action.get('action') == 'search':
            if sum(e['event'] == 'search' for e in events) >= max_searches:
                return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [], 'events': events + [{'event': 'search_limit'}]}
            query = action.get('query')
            if not isinstance(query, str) or not query.strip():
                raise ValueError('search query phải là chuỗi không rỗng')
            results = retriever.search(query, filters=filters, top_k=top_k)
            if len(results) > top_k or len({r['id'] for r in results}) != len(results):
                raise ValueError('retriever phải trả tối đa top_k chunk ID duy nhất')
            candidates.extend(results)
            if context_packer is not None:
                context, ids = context_packer(candidates, budget=budget, latest_results=results, events=events)
            else:
                context, ids = pack_context(candidates, budget=budget)
            cand_map = {c['id']: c for c in candidates}
            if all_chunks_by_id:
                cand_map = {**all_chunks_by_id, **cand_map}
            chunks = {cid: cand_map[cid] for cid in ids if cid in cand_map}
            search_event: dict[str, Any] = {
                'event': 'search', 'query': query, 'filter': filters,
                'results': results, 'context': context, 'context_ids': ids,
            }
            if 'tool' in action:
                search_event['tool'] = action['tool']
            if 'rewritten_query' in action:
                search_event['rewritten_query'] = action['rewritten_query']
            if 'thought' in action:
                search_event['thought'] = action['thought']
            events.append(search_event)
            # Send only the latest cumulative context; avoid multiplying the budget.
            history = [{'tool': 'search', 'queries': [e['query'] for e in events if e['event'] == 'search'],
                        'context': context, 'context_ids': ids}]
        elif action.get('action') == 'answer':
            answer, citations = action.get('answer'), action.get('citations')
            if not isinstance(answer, str) or not isinstance(citations, list) or any(not isinstance(i, str) for i in citations):
                raise ValueError('answer/citations không hợp lệ')
            valid = bool(answer.strip() and citations and all(i in chunks and f'[{i}]' in answer for i in citations))
            if valid:
                ans_event: dict[str, Any] = {'event': 'answer'}
                if 'thought' in action:
                    ans_event['thought'] = action['thought']
                return {'status': 'answered', 'answer': answer, 'citations': citations, 'events': events + [ans_event]}
            abstain_event: dict[str, Any] = {'event': 'invalid_or_missing_citations'}
            if action.get('abstain_reason'):
                abstain_event['abstain_reason'] = action['abstain_reason']
            return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [], 'events': events + [abstain_event]}
        else:
            raise ValueError('action phải là search hoặc answer')
    return {'status': 'abstain', 'answer': 'Không đủ bằng chứng để trả lời.', 'citations': [], 'events': events}
