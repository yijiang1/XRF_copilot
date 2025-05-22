function hideWatermark() {
  // Try multiple selector approaches
  const selectors = [
    "#chainlit-copilot",
    ".cl-copilot-container",
    "[data-testid='copilot-container']",
    // Add any other potential selectors
  ];

  for (const selector of selectors) {
    const elements = document.querySelectorAll(selector);
    
    elements.forEach(element => {
      // Try to access shadow DOM if it exists
      if (element.shadowRoot) {
        const watermarks = element.shadowRoot.querySelectorAll("a.watermark, .watermark, [class*='watermark']");
        watermarks.forEach(watermark => {
          watermark.style.display = "none";
          watermark.style.visibility = "hidden";
          watermark.remove(); // Try to remove it completely
        });
      }
      
      // Also check for watermarks in the regular DOM
      const directWatermarks = element.querySelectorAll("a.watermark, .watermark, [class*='watermark']");
      directWatermarks.forEach(watermark => {
        watermark.style.display = "none";
        watermark.style.visibility = "hidden";
        watermark.remove(); // Try to remove it completely
      });
    });
  }

  // Add CSS to hide watermarks globally
  const style = document.createElement('style');
  style.textContent = `
    a.watermark, .watermark, [class*='watermark'] {
      display: none !important;
      visibility: hidden !important;
      opacity: 0 !important;
      pointer-events: none !important;
    }
  `;
  document.head.appendChild(style);
}

// More aggressive approach with mutation observer for the entire document
function setupGlobalObserver() {
  const observer = new MutationObserver((mutations) => {
    let shouldCheck = false;
    
    for (const mutation of mutations) {
      if (mutation.addedNodes.length > 0) {
        shouldCheck = true;
        break;
      }
    }
    
    if (shouldCheck) {
      hideWatermark();
    }
  });
  
  observer.observe(document.body, { 
    childList: true, 
    subtree: true 
  });
}

// Run on page load
document.addEventListener("DOMContentLoaded", function() {
  // Try immediately
  hideWatermark();
  
  // Setup global observer
  setupGlobalObserver();
  
  // Try again after delays to catch late-loading elements
  setTimeout(hideWatermark, 1000);
  setTimeout(hideWatermark, 3000);
  
  // Periodically check
  setInterval(hideWatermark, 5000);
});

// Also run the script immediately in case the DOM is already loaded
if (document.readyState === "complete" || document.readyState === "interactive") {
  hideWatermark();
  setTimeout(setupGlobalObserver, 0);
}
