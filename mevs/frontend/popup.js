const btn = document.getElementById('summarize');
const output = document.getElementById('output');
const spinner = document.getElementById('spinner');
const statusText = document.getElementById('statusText');

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
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const url = tab?.url;
  if (!url) return;
  output.textContent = '';
  await setLoading(true);
  try {
    const res = await fetch('http://localhost:8000/summarize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    if (!res.ok) {
      const errText = await res.text();
      output.textContent = `Server error: ${res.status} ${res.statusText}\n${errText}`;
    } else {
      const data = await res.json();
      output.textContent = data.markdown || JSON.stringify(data, null, 2);
    }
  } catch (err) {
    output.textContent = 'Network error: ' + (err.message || err);
  } finally {
    await setLoading(false);
  }
});
