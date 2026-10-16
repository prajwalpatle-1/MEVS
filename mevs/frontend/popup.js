/** 
 * ==========================================
 * 1. DOM ELEMENTS & GLOBAL STATE
 * ==========================================
 */
const btn = document.getElementById('summarize');
const output = document.getElementById('output');
const spinner = document.getElementById('spinner');
const statusText = document.getElementById('statusText');
const videoUrl = document.getElementById('videoUrl');
const transcript = document.getElementById('transcript');
const downloadOptions = document.getElementById('downloadOptions');
const apiBaseUrl = 'http://localhost:8000';

// Track if the extension is currently generating a summary
let isProcessing = false; 
let generatedMarkdown = '';

/** 
 * ==========================================
 * 2. TAB & URL MANAGEMENT (SIDE PANEL LOGIC)
 * ==========================================
 */
async function updateUrlFromActiveTab() {
  if (isProcessing) return; // Ignore tab switches if a summary is currently running
  
  try {
    const tabs = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
    if (tabs && tabs.length > 0) {
      const tab = tabs[0];
      if (tab.url && tab.url.includes('youtube.com/watch')) {
        videoUrl.value = tab.url;
      }
    }
  } catch (err) {
    console.error("Failed to read tab URL:", err);
  }
}

// Check the URL immediately when the side panel first opens
updateUrlFromActiveTab();

// Listen for when the user switches to a different tab
chrome.tabs.onActivated.addListener(() => {
  updateUrlFromActiveTab();
});

// Listen for when the current tab's URL changes (e.g., clicking a new video)
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.url || changeInfo.status === 'complete') {
    updateUrlFromActiveTab();
  }
});


/** 
 * ==========================================
 * 3. UI STATE MANAGEMENT
 * ==========================================
 */
async function setLoading(loading) {
  isProcessing = loading; 
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


/** 
 * ==========================================
 * 4. MAIN SUMMARIZE LOGIC
 * ==========================================
 */
btn.addEventListener('click', async () => {
  const url = videoUrl.value.trim();
  const pastedTranscript = transcript.value.trim();
  
  if (!url && !pastedTranscript) {
    statusText.textContent = 'Add a link or transcript';
    return;
  }
  
  // Reset UI for new summary
  output.textContent = '';
  generatedMarkdown = '';
  downloadOptions.style.display = 'none'; 
  await setLoading(true);
  
  try {
    const res = await fetch(`${apiBaseUrl}/summarize`, {
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
        generatedMarkdown = data.markdown;
        renderSummary(data.markdown);
        output.scrollIntoView({ block: 'start' });
        
        // Show download options once summary is successfully generated
        downloadOptions.style.display = 'block'; 
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


async function downloadSummary(format) {
  if (!generatedMarkdown) {
    statusText.textContent = 'Create a summary first';
    return;
  }

  const extension = format === 'docx' ? 'docx' : 'pdf';
  statusText.textContent = `Preparing ${extension.toUpperCase()}...`;

  try {
    const response = await fetch(`${apiBaseUrl}/export/${format}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ markdown: generatedMarkdown })
    });
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(errorText || `Export failed (${response.status})`);
    }

    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = objectUrl;
    link.download = `MEVS-summary.${extension}`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    statusText.textContent = `${extension.toUpperCase()} downloaded`;
  } catch (err) {
    console.error('Summary export failed:', err);
    statusText.textContent = `Export failed: ${err.message || err}`;
  }
}

document.getElementById('downloadDocx')?.addEventListener('click', () => {
  downloadSummary('docx');
});

document.getElementById('downloadPdf')?.addEventListener('click', () => {
  downloadSummary('pdf');
});

function renderSummary(markdown) {
  output.replaceChildren();
  let activeList = null;

  for (const rawLine of markdown.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line) {
      activeList = null;
      continue;
    }

    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    if (heading) {
      const element = document.createElement(`h${heading[1].length}`);
      element.textContent = stripInlineMarkdown(heading[2]);
      output.appendChild(element);
      activeList = null;
      continue;
    }

    const bullet = line.match(/^[-*]\s+(.+)$/);
    if (bullet) {
      if (!activeList) {
        activeList = document.createElement('ul');
        output.appendChild(activeList);
      }
      const item = document.createElement('li');
      item.textContent = stripInlineMarkdown(bullet[1]);
      activeList.appendChild(item);
      continue;
    }

    const image = line.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
    if (image) {
      const source = resolveSummaryImage(image[2]);
      if (source) {
        const figure = document.createElement('figure');
        figure.className = 'summary-figure';
        const img = document.createElement('img');
        img.src = source;
        img.alt = image[1] || 'Video keyframe';
        img.loading = 'lazy';
        img.onerror = () => {
          figure.replaceChildren();
          const caption = document.createElement('figcaption');
          caption.textContent = 'Keyframe image is unavailable.';
          figure.appendChild(caption);
        };
        figure.appendChild(img);
        if (image[1]) {
          const caption = document.createElement('figcaption');
          caption.textContent = image[1];
          figure.appendChild(caption);
        }
        output.appendChild(figure);
      }
      activeList = null;
      continue;
    }

    const paragraph = document.createElement('p');
    paragraph.textContent = stripInlineMarkdown(line);
    output.appendChild(paragraph);
    activeList = null;
  }
}

function resolveSummaryImage(reference) {
  if (/^data:image\/(?:png|jpe?g|webp);base64,/i.test(reference)) {
    return reference;
  }

  const normalized = reference.replace(/\\/g, '/');
  const match = normalized.match(/^images\/([\w.-]+\.(?:jpe?g|png|webp))$/i);
  if (!match || match[1].includes('..')) return null;
  return `${apiBaseUrl}/images/${encodeURIComponent(match[1])}`;
}

function stripInlineMarkdown(text) {
  return text
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/[*_`]/g, '');
}