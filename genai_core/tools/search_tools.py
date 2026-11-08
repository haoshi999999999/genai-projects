from ddgs import DDGS


def search_web(query: str):

    try:
        results = DDGS().text(
            query,
            max_results=5
        )

        if not results:
            return f"No search results found for: {query}"

        formatted_results = []

        for index, result in enumerate(results, start=1):

            formatted_results.append(
                f"""
Result {index}
Title: {result.get('title', 'No title')}
URL: {result.get('href', 'No URL')}
Content: {result.get('body', 'No content')}
"""
            )

        return "\n".join(formatted_results)

    except Exception as exc:
        return f"Web search failed: {str(exc)}"
