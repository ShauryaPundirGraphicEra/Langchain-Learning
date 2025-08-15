document.getElementById("askBtn").addEventListener("click", async () => {
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  let url = tab.url;
  let question = document.getElementById("question").value;

  let response = await fetch(`http://localhost:8000/ask?url=${encodeURIComponent(url)}&question=${encodeURIComponent(question)}`);
  let data = await response.json();

  document.getElementById("answer").innerText = data.answer;
});
