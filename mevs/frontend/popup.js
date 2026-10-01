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

// Track if the extension is currently generating a summary
let isProcessing = false; 

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
  downloadOptions.style.display = 'none'; 
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


/** 
 * ==========================================
 * 5. DOWNLOAD LOGIC (PDF & WORD)
 * ==========================================
 */
document.getElementById('downloadPdf')?.addEventListener('click', async () => {
  const size = document.getElementById('summarySize').value;
  const summaryText = output.textContent; 
  
  statusText.textContent = "Fetching related image...";
  
  // TODO: Extract a better topic dynamically based on the video title
  const topic = "Educational Video Topic"; 
  
  // 1. Fetch image from Google
  const imageUrl = await fetchTopicImage(topic);
  let base64Img = null;
  if (imageUrl) {
    base64Img = await getBase64ImageFromUrl(imageUrl);
  }
  
  statusText.textContent = "Generating PDF...";

  try {
    // 2. Initialize jsPDF
    if (!window.jspdf) throw new Error("jsPDF library is not loaded.");
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    
    let currentHeight = 30;

    // 3. Add Title
    doc.setFontSize(16);
    doc.text(topic, 20, 20);
    
    // 4. Add Image if found
    if (base64Img) {
      doc.addImage(base64Img, 'JPEG', 20, currentHeight, 100, 60);
      currentHeight += 70; // move text cursor down past the image
    }
    
    // 5. Add Summary Info Header
    doc.setFontSize(14);
    doc.text(`Summary Type: ${size.toUpperCase()}`, 20, currentHeight);
    currentHeight += 10;
    
    // 6. Add Summary Text (Split to fit page width)
    doc.setFontSize(11);
    const splitText = doc.splitTextToSize(summaryText, 170);
    doc.text(splitText, 20, currentHeight);
    
    // 7. Save file and update UI
    doc.save(`MEVS_${size}_Summary.pdf`);
    statusText.textContent = "PDF Downloaded!";
    
    // Reset status text after 3 seconds
    setTimeout(() => {
      if (!isProcessing) statusText.textContent = "Idle";
    }, 3000);

  } catch (err) {
    console.error(err);
    statusText.textContent = "Error creating PDF";
  }
});

// Download Word Logic (Placeholder)
document.getElementById('downloadWord')?.addEventListener('click', () => {
  alert("Word download functionality requires the 'docx' library implementation.");
});


/** 
 * ==========================================
 * 6. HELPER FUNCTIONS
 * ==========================================
 */
async function fetchTopicImage(topic) {
  // IMPORTANT: Replace these with your actual Google Cloud keys
  const API_KEY = 'YOUR_GOOGLE_API_KEY';
  const SEARCH_ENGINE_ID = 'YOUR_CX_ID';
  const url = `https://www.googleapis.com/customsearch/v1?q=${encodeURIComponent(topic)}&cx=${SEARCH_ENGINE_ID}&searchType=image&key=${API_KEY}&num=1`;

  try {
    const response = await fetch(url);
    const data = await response.json();
    if (data.items && data.items.length > 0) {
      return data.items[0].link;
    }
    return null;
  } catch (error) {
    console.error("Image fetch failed", error);
    return null;
  }
}

async function getBase64ImageFromUrl(imageUrl) {
  try {
    const res = await fetch(imageUrl);
    const blob = await res.blob();
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  } catch (error) {
    console.error("Failed to convert image to Base64", error);
    return null;
  }
}