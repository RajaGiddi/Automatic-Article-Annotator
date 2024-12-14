from Bio import Entrez

def fetch_pubmed_title_abstract(query, email, max_results=10):
    """
    Fetches the titles and abstracts of articles from PubMed.

    Parameters:
        query (str): The search query string.
        email (str): Email address required by NCBI for API usage.
        max_results (int): Maximum number of articles to fetch.

    Returns:
        list of dict: A list of dictionaries containing titles and abstracts.
    """
    Entrez.email = email

    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()

        pmids = record["IdList"]

        results = []

        if not pmids:
            return "No results found."

        handle = Entrez.efetch(db="pubmed", id=pmids, rettype="abstract", retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        for article in records["PubmedArticle"]:
            try:
                title = article["MedlineCitation"]["Article"]["ArticleTitle"]
                abstract = article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"][0]
                results.append({"title": title, "abstract": abstract})
            except KeyError:
                continue

        return results

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    search_query = "math"
    user_email = "your_email@example.com"
    articles = fetch_pubmed_title_abstract(search_query, user_email, max_results=5)

    for idx, article in enumerate(articles):
        print(f"Article {idx + 1}:")
        print(f"Title: {article['title']}")
        print(f"Abstract: {article['abstract']}\n")
