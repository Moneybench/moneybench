// Function to fetch and render markdown content
async function fetchAndRenderMarkdown(markdownPath) {
    try {
        const response = await fetch(markdownPath);
        if (!response.ok) {
            throw new Error(`Failed to fetch markdown: ${response.status}`);
        }
        const markdownText = await response.text();
        document.getElementById('content').innerHTML = marked.parse(markdownText);
        document.getElementById('content').classList.remove('loading');
    } catch (error) {
        console.error('Error fetching markdown:', error);
        document.getElementById('content').innerHTML = `<p>Error loading content: ${error.message}</p>`;
        document.getElementById('content').classList.remove('loading');
    }
} 