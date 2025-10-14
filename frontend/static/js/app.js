// Vibe-Sync Frontend JavaScript

const API_BASE = window.location.origin;

// State
let currentMood = null;
let spotifyConnected = false;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    setupEventListeners();
    loadStats();
});

// Event Listeners
function setupEventListeners() {
    // Mood cards
    document.querySelectorAll('.mood-card').forEach(card => {
        card.addEventListener('click', () => {
            const mood = card.dataset.mood;
            selectMood(mood);
        });
    });

    // Back button
    document.getElementById('back-btn')?.addEventListener('click', () => {
        showSection('mood');
    });

    // Spotify setup form
    document.getElementById('spotify-setup-form')?.addEventListener('submit', async (e) => {
        e.preventDefault();
        await setupSpotify();
    });
}

// API Functions
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();

        spotifyConnected = data.spotify_connected;

        updateStatus(data.status === 'healthy', spotifyConnected);

        if (!spotifyConnected) {
            document.getElementById('setup-section').classList.remove('hidden');
        }
    } catch (error) {
        console.error('Health check failed:', error);
        updateStatus(false, false);
    }
}

function updateStatus(healthy, connected) {
    const indicator = document.getElementById('status-indicator');
    const text = document.getElementById('status-text');

    if (healthy && connected) {
        indicator.classList.add('connected');
        text.textContent = 'Connected to Spotify';
    } else if (healthy) {
        text.textContent = 'Ready (No Spotify)';
    } else {
        text.textContent = 'Disconnected';
    }
}

async function setupSpotify() {
    const clientId = document.getElementById('client-id').value;
    const clientSecret = document.getElementById('client-secret').value;

    if (!clientId || !clientSecret) {
        alert('Please enter both Client ID and Client Secret');
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/api/setup-spotify`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                client_id: clientId,
                client_secret: clientSecret
            })
        });

        const data = await response.json();

        if (response.ok) {
            alert('✅ Spotify connected successfully!');
            document.getElementById('setup-section').classList.add('hidden');
            spotifyConnected = true;
            updateStatus(true, true);
            loadStats();
        } else {
            alert('❌ Failed to connect: ' + data.detail);
        }
    } catch (error) {
        console.error('Setup error:', error);
        alert('❌ Connection error. Check console for details.');
    }
}

async function selectMood(mood) {
    currentMood = mood;
    showSection('results');
    showLoading(true);

    try {
        const response = await fetch(`${API_BASE}/api/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                mood: mood,
                limit: 50
            })
        });

        const data = await response.json();

        if (response.ok) {
            displayRecommendations(data.recommendations);
        } else {
            alert('❌ Error: ' + data.detail);
            showSection('mood');
        }
    } catch (error) {
        console.error('Recommendation error:', error);
        alert('❌ Failed to get recommendations. Please try again.');
        showSection('mood');
    } finally {
        showLoading(false);
    }
}

async function loadStats() {
    try {
        const response = await fetch(`${API_BASE}/api/stats`);
        const data = await response.json();

        document.getElementById('stat-tracks').textContent =
            data.total_tracks > 0 ? data.total_tracks.toLocaleString() : '-';
        document.getElementById('stat-moods').textContent = data.moods_available;

        // Update system status based on mood classifier readiness
        const systemElement = document.getElementById('stat-system') || document.getElementById('statAccuracy');
        if (systemElement) {
            systemElement.textContent = data.mood_classifier_ready ? 'Ready' : 'Setting up...';
        }
    } catch (error) {
        console.error('Stats error:', error);
    }
}

// UI Functions
function showSection(section) {
    document.getElementById('mood-section').classList.toggle('hidden', section !== 'mood');
    document.getElementById('results-section').classList.toggle('hidden', section !== 'results');
}

function showLoading(show) {
    document.getElementById('loading').classList.toggle('hidden', !show);
    document.getElementById('recommendations').classList.toggle('hidden', show);
}

function displayRecommendations(tracks) {
    const container = document.getElementById('recommendations');
    container.innerHTML = '';

    if (!tracks || tracks.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No recommendations found.</p>';
        return;
    }

    tracks.forEach((track, index) => {
        const card = createTrackCard(track, index + 1);
        container.appendChild(card);
    });
}

function createTrackCard(track, rank) {
    const card = document.createElement('div');
    card.className = 'track-card';

    const score = Math.round(track.score * 100);

    // Create mood scores display
    let moodScoresHtml = '';
    if (track.mood_scores) {
        const moodEmojis = {
            'workout': '🏋️',
            'chill': '😌',
            'party': '🎉',
            'focus': '📚',
            'sleep': '😴'
        };

        moodScoresHtml = '<div class="mood-compatibility">';
        for (const [mood, score] of Object.entries(track.mood_scores)) {
            const percentage = Math.round(score * 100);
            const emoji = moodEmojis[mood] || '🎵';
            moodScoresHtml += `<div class="mood-score" title="${mood}: ${percentage}%">
                <span class="mood-emoji">${emoji}</span>
                <div class="mood-bar">
                    <div class="mood-fill" style="width: ${percentage}%"></div>
                </div>
                <span class="mood-percent">${percentage}%</span>
            </div>`;
        }
        moodScoresHtml += '</div>';
    }

    card.innerHTML = `
        <div class="track-rank">${rank}</div>
        <div class="track-image">
            ${track.image ? `<img src="${track.image}" alt="${track.name}">` : ''}
        </div>
        <div class="track-info">
            <div class="track-name">${track.name}</div>
            <div class="track-artist">${track.artist}</div>
            <div class="track-score">
                <div class="score-bar">
                    <div class="score-fill" style="width: ${score}%"></div>
                </div>
                <span class="score-text">${score}% match</span>
            </div>
            ${moodScoresHtml}
            ${track.spotify_url ? `<a href="${track.spotify_url}" target="_blank" class="track-link">Open in Spotify →</a>` : ''}
        </div>
    `;

    return card;
}

// Utility
function formatDuration(ms) {
    const minutes = Math.floor(ms / 60000);
    const seconds = Math.floor((ms % 60000) / 1000);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}
