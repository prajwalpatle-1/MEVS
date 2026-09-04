const btn = document.getElementById('summarize');
const output = document.getElementById('output');
const spinner = document.getElementById('spinner');
const statusText = document.getElementById('statusText');
const videoUrl = document.getElementById('videoUrl');
const transcript = document.getElementById('transcript');

chrome.tabs.query({ active: true, currentWindow: true }).then(([tab]) => {
  if (tab?.url?.includes('youtube.com/watch')) videoUrl.value = tab.url;
});

async function setLoading(loading) {
  if (loading) {
    btn.disabled = true;
    spinner.classList.add('visible');
    statusText.textContent = 'Processing...';
  } else {
    btn.disabled = false;
    spinner.classList.remove('visible');
    statusText.textContent = 'Idle';
  }
}

btn.addEventListener('click', async () => {
  const url = videoUrl.value.trim();
  const pastedTranscript = transcript.value.trim();
  if (!url && !pastedTranscript) {
    statusText.textContent = 'Add a link or transcript';
    return;
  }
  output.textContent = '';
  await setLoading(true);
  try {
    const res = await fetch('http://localhost:8000/summarize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, transcript: pastedTranscript })
    });
    if (!res.ok) {
      const errText = await res.text();
      output.textContent = `Server error: ${res.status} ${res.statusText}\n${errText}`;
    } else {
      const data = await res.json();
      if (typeof data.markdown === 'string' && data.markdown.trim()) {
        output.textContent = data.markdown;
        output.scrollIntoView({ block: 'start' });
      } else {
        output.textContent = 'The server returned an empty summary.';
      }
    }
  } catch (err) {
    output.textContent = 'Network error: ' + (err.message || err);
  } finally {
    await setLoading(false);
  }
});
