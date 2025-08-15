# AIzaSyB3pgq-K9fTvokMglsfMIcL-cmAAL8cJqQ  <- Google api


#search engine id:   63eecdc622a494e5a

# https://cse.google.com/cse?cx=63eecdc622a494e5a


# custom search API key :   AIzaSyByaoC5KJdMww_gYE2nP01hXwmA7SgLlCM





import requests
import trafilatura

# Your API credentials
API_KEY = "AIzaSyByaoC5KJdMww_gYE2nP01hXwmA7SgLlCM"
SEARCH_ENGINE_ID = "63eecdc622a494e5a"

def google_search(query, num_results=2):
    """
    Perform Google Programmable Search and return top result URLs.
    """
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "q": query,
        "num": num_results
    }
    response = requests.get(url, params=params)
    data = response.json()
    results = []
    if "items" in data:
        for item in data["items"]:
            results.append(item["link"])
    return results

def fetch_article_text(url):
    """
    Fetch and extract main content text from a URL.
    """
    downloaded = trafilatura.fetch_url(url)
    if downloaded:
        text = trafilatura.extract(downloaded)
        return text or ""
    return ""

def get_combined_articles(query, num_results=6):
    """
    Get top N article URLs and combine their text into one string.
    """
    urls = google_search(query, num_results)
    combined_text = ""
    for i, url in enumerate(urls, start=1):
        print(f"[{i}] Fetching: {url}")
        text = fetch_article_text(url)
        if text:
            combined_text += f"\n--- ARTICLE {i} ---\n{text}\n"
    return combined_text

if __name__ == "__main__":
    topic = "Operation Sindoor"
    combined_content = get_combined_articles(topic, num_results=6)
    print("\nFINAL COMBINED TEXT:\n", combined_content)
