// Configure marked.js for GitHub-style markdown with syntax highlighting
marked.setOptions({
    gfm: true,
    breaks: true,
    headerIds: true,
    highlight: function(code, language) {
        if (language && hljs.getLanguage(language)) {
            try {
                return hljs.highlight(code, { language }).value;
            } catch (err) {
                console.error(err);
            }
        }
        return hljs.highlightAuto(code).value;
    }
});

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
        
        // Apply syntax highlighting to code blocks
        document.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightElement(block);
        });
        
        // Add active class to current page
        highlightCurrentPage();
    } catch (error) {
        console.error('Error fetching markdown:', error);
        document.getElementById('content').innerHTML = `<p>Error loading content: ${error.message}</p>`;
        document.getElementById('content').classList.remove('loading');
    }
}

// Function to highlight the current page in the navigation
function highlightCurrentPage() {
    const currentPage = window.location.pathname.split('/').pop();
    const navLinks = document.querySelectorAll('nav a');
    
    navLinks.forEach(link => {
        const linkHref = link.getAttribute('href');
        if (currentPage === linkHref || 
            (currentPage === '' && linkHref === 'index.html')) {
            link.classList.add('active');
        }
    });
}

// Simple direct theme toggle function
function toggleTheme() {
    const html = document.documentElement;
    const sunIcon = document.getElementById('sun-icon');
    const moonIcon = document.getElementById('moon-icon');
    const themeText = document.getElementById('theme-text');
    
    // Check current theme
    const isDarkTheme = html.classList.contains('dark-theme');
    
    // Toggle theme
    if (isDarkTheme) {
        // Switch to light theme
        html.classList.remove('dark-theme');
        sunIcon.style.display = 'none';
        moonIcon.style.display = 'block';
        themeText.textContent = 'Dark Mode';
        localStorage.setItem('theme', 'light');
        console.log('Switched to light theme');
    } else {
        // Switch to dark theme
        html.classList.add('dark-theme');
        sunIcon.style.display = 'block';
        moonIcon.style.display = 'none';
        themeText.textContent = 'Light Mode';
        localStorage.setItem('theme', 'dark');
        console.log('Switched to dark theme');
    }
}

// Initialize theme based on saved preference
function initTheme() {
    const html = document.documentElement;
    const sunIcon = document.getElementById('sun-icon');
    const moonIcon = document.getElementById('moon-icon');
    const themeText = document.getElementById('theme-text');
    const themeToggle = document.getElementById('theme-toggle');
    
    if (!themeToggle || !sunIcon || !moonIcon || !themeText) {
        console.error('Theme elements not found');
        return;
    }
    
    // Apply saved theme if exists
    const savedTheme = localStorage.getItem('theme');
    
    // Default to dark theme if no preference is saved
    if (!savedTheme || savedTheme === 'dark') {
        html.classList.add('dark-theme');
        sunIcon.style.display = 'block';
        moonIcon.style.display = 'none';
        themeText.textContent = 'Light Mode';
        localStorage.setItem('theme', 'dark');
        console.log('Applied dark theme (default or saved)');
    } else if (savedTheme === 'light') {
        html.classList.remove('dark-theme');
        sunIcon.style.display = 'none';
        moonIcon.style.display = 'block';
        themeText.textContent = 'Dark Mode';
        console.log('Applied saved light theme');
    }
    
    // Add click event directly
    themeToggle.onclick = toggleTheme;
    console.log('Theme toggle click handler attached');
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded - initializing theme');
    initTheme();
});

// Also initialize if script is loaded after DOM is ready
if (document.readyState === 'complete' || document.readyState === 'interactive') {
    console.log('Document already loaded - initializing theme immediately');
    setTimeout(initTheme, 100);
} 