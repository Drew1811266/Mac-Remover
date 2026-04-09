function initTheme() {
  const savedTheme = localStorage.getItem('theme') || 'light';
  document.body.setAttribute('theme-mode', savedTheme);
  return savedTheme;
}

function toggleTheme() {
  const body = document.body;
  const currentTheme = body.getAttribute('theme-mode');
  const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
  body.setAttribute('theme-mode', newTheme);
  localStorage.setItem('theme', newTheme);
  return newTheme;
}

function getTheme() {
  return document.body.getAttribute('theme-mode') || 'light';
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { initTheme, toggleTheme, getTheme };
}
