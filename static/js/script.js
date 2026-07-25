/**
 * script.js - Brain Tumor Detection System
 * Global JavaScript: navbar, flash messages, animations, utilities
 */

'use strict';

// ─── Navbar ────────────────────────────────────────────────────────────────
const navbar = document.getElementById('navbar');
const navToggle = document.getElementById('navToggle');
const navLinks = document.getElementById('navLinks');

// Sticky navbar shadow on scroll
window.addEventListener('scroll', () => {
  if (navbar) {
    navbar.classList.toggle('scrolled', window.scrollY > 10);
  }
});

// Mobile menu toggle
if (navToggle && navLinks) {
  navToggle.addEventListener('click', () => {
    navLinks.classList.toggle('open');
    navToggle.classList.toggle('open');
  });

  // Close menu when a link is clicked
  navLinks.querySelectorAll('a, button.nav-link').forEach(item => {
    item.addEventListener('click', () => {
      navLinks.classList.remove('open');
      navToggle.classList.remove('open');
    });
  });

  // Close menu when clicking outside
  document.addEventListener('click', (e) => {
    if (!navbar.contains(e.target)) {
      navLinks.classList.remove('open');
      navToggle.classList.remove('open');
    }
  });
}

// ─── Flash Message Auto-Dismiss ────────────────────────────────────────────
document.querySelectorAll('.flash').forEach(flash => {
  setTimeout(() => {
    flash.style.opacity = '0';
    flash.style.transform = 'translateX(100%)';
    flash.style.transition = 'all 0.4s ease';
    setTimeout(() => flash.remove(), 400);
  }, 5000);
});

// ─── Password Toggle ────────────────────────────────────────────────────────
function togglePassword(inputId) {
  const input = document.getElementById(inputId);
  if (!input) return;
  input.type = input.type === 'password' ? 'text' : 'password';
}

// ─── Smooth Scroll for anchor links ─────────────────────────────────────────
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', function (e) {
    const target = document.querySelector(this.getAttribute('href'));
    if (target) {
      e.preventDefault();
      const offset = 80; // navbar height
      const top = target.getBoundingClientRect().top + window.scrollY - offset;
      window.scrollTo({ top, behavior: 'smooth' });
    }
  });
});

// ─── Animated Counter for dashboard stats ──────────────────────────────────
function animateCounters() {
  document.querySelectorAll('.stat-num[data-count]').forEach(el => {
    const target = parseInt(el.dataset.count, 10);
    if (isNaN(target)) return;
    let current = 0;
    const step = Math.max(1, Math.ceil(target / 40));
    const interval = setInterval(() => {
      current = Math.min(current + step, target);
      el.textContent = current;
      if (current >= target) clearInterval(interval);
    }, 30);
  });
}

// ─── Intersection Observer for animations ──────────────────────────────────
const observerOptions = { threshold: 0.1, rootMargin: '0px 0px -40px 0px' };

const fadeObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('visible');
      fadeObserver.unobserve(entry.target);
    }
  });
}, observerOptions);

// Animate cards on scroll
document.querySelectorAll('.feature-card, .step-card, .team-card, .class-card, .objective-card, .dataset-card').forEach(el => {
  el.style.opacity = '0';
  el.style.transform = 'translateY(20px)';
  el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
  fadeObserver.observe(el);
});

// Trigger stat counter animation when dashboard stats are visible
const statsGrid = document.querySelector('.stats-grid');
if (statsGrid) {
  const statsObserver = new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting) {
      animateCounters();
      statsObserver.disconnect();
    }
  }, { threshold: 0.3 });
  statsObserver.observe(statsGrid);
}

// Add visible class handler
document.head.insertAdjacentHTML('beforeend', `
  <style>
    .visible {
      opacity: 1 !important;
      transform: translateY(0) !important;
    }
  </style>
`);

// ─── Hero scan animation ─────────────────────────────────────────────────────
const heroScanAnim = document.getElementById('heroScanAnim');
if (heroScanAnim) {
  const states = ['Analyzing...', 'Processing...', 'Glioma Detected', 'Confidence: 92%'];
  const colors = ['#1a73e8', '#6c63ff', '#dc2626', '#16a34a'];
  let i = 0;
  setInterval(() => {
    i = (i + 1) % states.length;
    heroScanAnim.textContent = states[i];
    heroScanAnim.style.color = colors[i];
  }, 2000);
}

// ─── Active nav link highlight (based on scroll) ────────────────────────────
const sections = document.querySelectorAll('section[id]');
if (sections.length > 0) {
  const navScrollObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.getAttribute('id');
        document.querySelectorAll('.nav-link').forEach(link => {
          link.classList.remove('active');
          if (link.getAttribute('href') === `#${id}`) {
            link.classList.add('active');
          }
        });
      }
    });
  }, { threshold: 0.5 });
  sections.forEach(s => navScrollObserver.observe(s));
}

// ─── Table row hover enhancement ────────────────────────────────────────────
document.querySelectorAll('.data-table tbody tr').forEach(row => {
  row.style.transition = 'background 0.15s ease';
});

// ─── Confirmation dialog utility ─────────────────────────────────────────────
function confirmAction(message, callback) {
  if (window.confirm(message)) callback();
}

// ─── Copy to clipboard utility ───────────────────────────────────────────────
function copyToClipboard(text) {
  navigator.clipboard.writeText(text).then(() => {
    showToast('Copied to clipboard!', 'success');
  }).catch(() => {
    showToast('Failed to copy.', 'error');
  });
}

// ─── Toast notification ──────────────────────────────────────────────────────
function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `flash flash-${type}`;
  toast.innerHTML = `<span>${message}</span><button class="flash-close" onclick="this.parentElement.remove()">×</button>`;

  let container = document.getElementById('flashContainer');
  if (!container) {
    container = document.createElement('div');
    container.id = 'flashContainer';
    container.className = 'flash-container';
    document.body.appendChild(container);
  }
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(100%)';
    toast.style.transition = 'all 0.4s ease';
    setTimeout(() => toast.remove(), 400);
  }, 4000);
}

// ─── Session timeout warning ─────────────────────────────────────────────────
(function sessionTimeoutWarning() {
  const WARNING_BEFORE = 5 * 60 * 1000;  // 5 minutes before timeout
  const SESSION_DURATION = 2 * 60 * 60 * 1000; // 2 hours (matches Flask config)

  // Only run if user appears to be logged in (check for dashboard/upload links)
  const isLoggedIn = document.querySelector('a[href*="dashboard"], a[href*="upload"]');
  if (!isLoggedIn) return;

  setTimeout(() => {
    showToast('⚠️ Your session will expire in 5 minutes. Please save your work.', 'warning');
  }, SESSION_DURATION - WARNING_BEFORE);
})();

// ─── Image lazy loading ────────────────────────────────────────────────────
document.querySelectorAll('img[data-src]').forEach(img => {
  const imgObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.src = entry.target.dataset.src;
        imgObserver.unobserve(entry.target);
      }
    });
  });
  imgObserver.observe(img);
});

// ─── Form validation helper ───────────────────────────────────────────────
function validateEmail(email) {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

function showFieldError(fieldId, errorId, message) {
  const field = document.getElementById(fieldId);
  const error = document.getElementById(errorId);
  if (field) field.style.borderColor = 'var(--danger)';
  if (error) error.textContent = message;
}

function clearFieldError(fieldId, errorId) {
  const field = document.getElementById(fieldId);
  const error = document.getElementById(errorId);
  if (field) field.style.borderColor = '';
  if (error) error.textContent = '';
}

// Real-time email validation on register page
const emailInput = document.getElementById('email');
if (emailInput) {
  emailInput.addEventListener('blur', function () {
    if (this.value && !validateEmail(this.value)) {
      showFieldError('email', 'emailError', 'Please enter a valid email address.');
    } else {
      clearFieldError('email', 'emailError');
    }
  });
}

// ─── Navbar hamburger animation ───────────────────────────────────────────
if (navToggle) {
  navToggle.addEventListener('click', function () {
    const spans = this.querySelectorAll('span');
    this.classList.toggle('open');
    if (this.classList.contains('open')) {
      spans[0].style.transform = 'rotate(45deg) translate(5px, 5px)';
      spans[1].style.opacity = '0';
      spans[2].style.transform = 'rotate(-45deg) translate(5px, -5px)';
    } else {
      spans[0].style.transform = '';
      spans[1].style.opacity = '';
      spans[2].style.transform = '';
    }
  });
}

// ─── Print support ────────────────────────────────────────────────────────
function printPage() {
  window.print();
}

// ─── Initialize on DOM ready ──────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Animate stat counters immediately if already visible (above fold)
  if (document.querySelector('.stat-num[data-count]')) {
    animateCounters();
  }

  // Add loading="lazy" to all non-critical images
  document.querySelectorAll('img:not([loading])').forEach(img => {
    if (!img.closest('.hero') && !img.closest('.navbar')) {
      img.setAttribute('loading', 'lazy');
    }
  });
});
