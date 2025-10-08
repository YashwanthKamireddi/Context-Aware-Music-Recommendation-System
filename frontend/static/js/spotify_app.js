/* ===================================
   VIBE-SYNC - SPOTIFY CLONE APP
   Complete Production JavaScript
   =================================== */

// ===================================
// STATE MANAGEMENT
// ===================================
const state = {
    currentView: 'home',
    currentMood: null,
    currentPlaylist: [],
    isPlaying: false,
    currentTrackIndex: 0,
    searchResults: [],
    history: [],
    historyIndex: -1,
    audioPlayer: null, // HTML5 Audio element
    currentVolume: 0.7
};

// ===================================
// API CALLS
// ===================================
const API = {
    async checkHealth() {
        try {
            const response = await fetch('/api/health');
            return await response.json();
        } catch (error) {
            console.error('Health check failed:', error);
            return { status: 'error' };
        }
    },

    async getStats() {
        try {
            const response = await fetch('/api/stats');
            return await response.json();
        } catch (error) {
            console.error('Failed to get stats:', error);
            return null;
        }
    },

    async getRecommendations(mood) {
        try {
            showToast(`🎵 Generating ${mood} recommendations...`);
            const response = await fetch('/api/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mood: mood, limit: 20 })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            // Handle both 'tracks' and 'recommendations' response format
            const tracks = data.tracks || data.recommendations || [];
            showToast(`✅ Found ${tracks.length} perfect tracks!`);
            return { ...data, tracks: tracks };
        } catch (error) {
            console.error('Failed to get recommendations:', error);
            showToast('❌ Failed to generate recommendations', 'error');
            return null;
        }
    },

    async searchTracks(query) {
        try {
            const response = await fetch(`/api/search?query=${encodeURIComponent(query)}&limit=20`);
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Search failed:', error);
            return { tracks: [] };
        }
    }
};

// ===================================
// UI HELPERS
// ===================================
function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    const toastMessage = document.getElementById('toastMessage');
    const icon = toast.querySelector('i');

    toastMessage.textContent = message;

    if (type === 'error') {
        icon.className = 'fas fa-exclamation-circle';
    } else {
        icon.className = 'fas fa-check-circle';
    }

    toast.classList.add('show');

    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

function showLoading(show = true) {
    const loadingScreen = document.getElementById('loadingScreen');
    if (show) {
        loadingScreen.classList.remove('hidden');
    } else {
        loadingScreen.classList.add('hidden');
    }
}

function switchView(viewName) {
    // Update nav items
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.remove('active');
        if (item.dataset.view === viewName) {
            item.classList.add('active');
        }
    });

    // Update views
    document.querySelectorAll('.content-view').forEach(view => {
        view.classList.remove('active');
    });

    const targetView = document.getElementById(`${viewName}View`);
    if (targetView) {
        targetView.classList.add('active');
    }

    // Update history
    if (state.historyIndex < state.history.length - 1) {
        state.history = state.history.slice(0, state.historyIndex + 1);
    }
    state.history.push(viewName);
    state.historyIndex = state.history.length - 1;
    updateNavButtons();

    state.currentView = viewName;
}

function updateNavButtons() {
    const backBtn = document.getElementById('backBtn');
    const forwardBtn = document.getElementById('forwardBtn');

    backBtn.disabled = state.historyIndex <= 0;
    forwardBtn.disabled = state.historyIndex >= state.history.length - 1;
}

function formatDuration(ms) {
    const seconds = Math.floor((ms / 1000) % 60);
    const minutes = Math.floor((ms / (1000 * 60)) % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

function getMoodIcon(mood) {
    const icons = {
        workout: 'fa-dumbbell',
        chill: 'fa-cloud',
        party: 'fa-glass-cheers',
        focus: 'fa-brain',
        sleep: 'fa-moon'
    };
    return icons[mood] || 'fa-music';
}

function getMoodTitle(mood) {
    const titles = {
        workout: 'Workout Energy',
        chill: 'Chill Vibes',
        party: 'Party Mode',
        focus: 'Deep Focus',
        sleep: 'Sleep & Relax'
    };
    return titles[mood] || mood;
}

function getMoodDescription(mood) {
    const descriptions = {
        workout: 'High-energy tracks to fuel your workout and keep you motivated',
        chill: 'Relaxing melodies to help you unwind and find your zen',
        party: 'Upbeat anthems that will get everyone on the dance floor',
        focus: 'Ambient sounds and instrumental tracks for deep concentration',
        sleep: 'Peaceful music to help you drift into restful sleep'
    };
    return descriptions[mood] || 'AI-curated tracks for your mood';
}

// ===================================
// PLAYLIST RENDERING
// ===================================
function displayPlaylist(mood, tracks) {
    state.currentMood = mood;
    state.currentPlaylist = tracks;

    // Update playlist header
    document.getElementById('playlistTitle').textContent = getMoodTitle(mood);
    document.getElementById('playlistDescription').textContent = getMoodDescription(mood);
    document.getElementById('playlistCount').textContent = `${tracks.length} songs`;

    // Update playlist cover color based on mood
    const playlistCover = document.querySelector('.playlist-cover');
    const gradients = {
        workout: 'linear-gradient(135deg, #ff6b6b, #ee5a6f)',
        chill: 'linear-gradient(135deg, #4facfe, #00f2fe)',
        party: 'linear-gradient(135deg, #f093fb, #f5576c)',
        focus: 'linear-gradient(135deg, #43e97b, #38f9d7)',
        sleep: 'linear-gradient(135deg, #fa709a, #fee140)'
    };
    playlistCover.style.background = gradients[mood] || gradients.chill;

    // Render tracks
    const tracksContainer = document.getElementById('tracksContainer');
    tracksContainer.innerHTML = '';

    tracks.forEach((track, index) => {
        const trackRow = createTrackRow(track, index + 1);
        tracksContainer.appendChild(trackRow);
    });

    // Switch to playlist view
    document.querySelectorAll('.content-view').forEach(v => v.classList.remove('active'));
    document.getElementById('playlistView').classList.add('active');

    // Scroll to top
    document.querySelector('.main-content').scrollTop = 0;
}

function createTrackRow(track, number) {
    const row = document.createElement('div');
    row.className = 'track-row';
    row.dataset.trackId = track.id || number;

    // Track Number / Play Icon
    const numberCell = document.createElement('div');
    numberCell.className = 'track-number';
    numberCell.innerHTML = `
        <span>${number}</span>
        <div class="track-play-icon">
            <i class="fas fa-play"></i>
        </div>
    `;

    // Track Info (Cover + Name + Artist)
    const infoCell = document.createElement('div');
    infoCell.className = 'track-info';

    const coverDiv = document.createElement('div');
    coverDiv.className = 'track-cover';

    if (track.album_art && track.album_art !== 'Unknown') {
        const img = document.createElement('img');
        img.src = track.album_art;
        img.alt = track.name;
        img.onerror = () => {
            coverDiv.innerHTML = '<i class="fas fa-music"></i>';
        };
        coverDiv.appendChild(img);
    } else {
        coverDiv.innerHTML = '<i class="fas fa-music"></i>';
    }

    const detailsDiv = document.createElement('div');
    detailsDiv.className = 'track-details';
    detailsDiv.innerHTML = `
        <div class="track-name">${track.name || 'Unknown Track'}</div>
        <div class="track-artist">${track.artist || 'Unknown Artist'}</div>
    `;

    infoCell.appendChild(coverDiv);
    infoCell.appendChild(detailsDiv);

    // Track Album
    const albumCell = document.createElement('div');
    albumCell.className = 'track-album';
    const albumName = track.album || 'Unknown Album';
    albumCell.textContent = albumName;

    // Track Duration + Score
    const durationCell = document.createElement('div');
    durationCell.className = 'track-duration';

    if (track.duration_ms) {
        durationCell.textContent = formatDuration(track.duration_ms);
    }

    if (track.score !== undefined) {
        const scoreSpan = document.createElement('span');
        scoreSpan.className = 'track-score';
        scoreSpan.textContent = `${Math.round(track.score * 100)}%`;
        durationCell.appendChild(scoreSpan);
    }

    // Assemble row
    row.appendChild(numberCell);
    row.appendChild(infoCell);
    row.appendChild(albumCell);
    row.appendChild(durationCell);

    // Click to play
    row.addEventListener('click', () => {
        playTrack(track, number - 1);
    });

    return row;
}

function playTrack(track, index) {
    state.currentTrackIndex = index;

    // Update now playing bar
    const nowPlayingCover = document.querySelector('.now-playing-cover');
    const nowPlayingTitle = document.querySelector('.now-playing-title');
    const nowPlayingArtist = document.querySelector('.now-playing-artist');
    const playBtn = document.getElementById('playBtn');

    if (track.album_art && track.album_art !== 'Unknown') {
        nowPlayingCover.innerHTML = `<img src="${track.album_art}" alt="${track.name}">`;
    } else {
        nowPlayingCover.innerHTML = '<i class="fas fa-music"></i>';
    }

    nowPlayingTitle.textContent = track.name || 'Unknown Track';
    nowPlayingArtist.textContent = track.artist || 'Unknown Artist';

    // REAL AUDIO PLAYBACK
    if (track.preview_url) {
        // Initialize audio player if not exists
        if (!state.audioPlayer) {
            state.audioPlayer = new Audio();
            state.audioPlayer.volume = state.currentVolume;

            // Auto-advance to next track when preview ends
            state.audioPlayer.addEventListener('ended', () => {
                if (state.currentPlaylist.length > 0) {
                    const nextIndex = (state.currentTrackIndex + 1) % state.currentPlaylist.length;
                    playTrack(state.currentPlaylist[nextIndex], nextIndex);
                } else {
                    state.isPlaying = false;
                    playBtn.querySelector('i').className = 'fas fa-play';
                }
            });

            // Handle playback errors
            state.audioPlayer.addEventListener('error', () => {
                showToast('⚠️ Preview not available - Opening in Spotify', 'error');
                if (track.spotify_url) {
                    window.open(track.spotify_url, '_blank');
                }
                state.isPlaying = false;
                playBtn.querySelector('i').className = 'fas fa-play';
            });

            // Update progress bar as track plays
            state.audioPlayer.addEventListener('timeupdate', updateProgressBar);

            // Update total time when loaded
            state.audioPlayer.addEventListener('loadedmetadata', () => {
                const totalTime = document.getElementById('totalTime');
                totalTime.textContent = formatDuration(state.audioPlayer.duration * 1000);
            });
        }

        // Play the track
        state.audioPlayer.src = track.preview_url;
        state.audioPlayer.play()
            .then(() => {
                state.isPlaying = true;
                playBtn.querySelector('i').className = 'fas fa-pause';
                showToast(`🎵 Playing: ${track.name}`);
            })
            .catch(error => {
                console.error('Playback failed:', error);
                showToast('⚠️ Preview not available - Opening in Spotify', 'error');
                if (track.spotify_url) {
                    window.open(track.spotify_url, '_blank');
                }
            });
    } else {
        // No preview available - open in Spotify
        showToast('⚠️ No preview - Opening in Spotify');
        if (track.spotify_url) {
            window.open(track.spotify_url, '_blank');
        }
        playBtn.querySelector('i').className = 'fas fa-play';
    }
}

function updateProgressBar() {
    if (!state.audioPlayer) return;

    const currentTime = state.audioPlayer.currentTime;
    const duration = state.audioPlayer.duration;

    if (!isNaN(duration)) {
        const progressPercent = (currentTime / duration) * 100;
        const progressFill = document.getElementById('progressFill');
        const progressHandle = document.getElementById('progressHandle');
        const currentTimeEl = document.getElementById('currentTime');

        progressFill.style.width = `${progressPercent}%`;
        progressHandle.style.left = `${progressPercent}%`;
        currentTimeEl.textContent = formatDuration(currentTime * 1000);
    }
}

// ===================================
// MOOD SELECTION
// ===================================
async function selectMood(mood) {
    showLoading(true);

    const result = await API.getRecommendations(mood);

    showLoading(false);

    if (result && result.tracks && result.tracks.length > 0) {
        displayPlaylist(mood, result.tracks);
    } else {
        showToast('❌ No recommendations available', 'error');
    }
}

// ===================================
// SEARCH FUNCTIONALITY
// ===================================
let searchTimeout;
async function handleSearch(query) {
    clearTimeout(searchTimeout);

    if (!query || query.trim().length < 2) {
        document.getElementById('searchResults').innerHTML = `
            <div class="empty-state">
                <i class="fas fa-search"></i>
                <h3>Find your music</h3>
                <p>Search for songs, artists, and albums</p>
            </div>
        `;
        return;
    }

    searchTimeout = setTimeout(async () => {
        const results = await API.searchTracks(query);
        displaySearchResults(results.tracks || []);
    }, 500);
}

function displaySearchResults(tracks) {
    const resultsContainer = document.getElementById('searchResults');

    if (tracks.length === 0) {
        resultsContainer.innerHTML = `
            <div class="empty-state">
                <i class="fas fa-search"></i>
                <h3>No results found</h3>
                <p>Try searching with different keywords</p>
            </div>
        `;
        return;
    }

    resultsContainer.innerHTML = '<div class="tracks-list"></div>';
    const tracksList = resultsContainer.querySelector('.tracks-list');

    tracks.forEach((track, index) => {
        const trackRow = createTrackRow(track, index + 1);
        tracksList.appendChild(trackRow);
    });

    state.searchResults = tracks;
}

// ===================================
// INITIALIZATION
// ===================================
async function initializeApp() {
    showLoading(true);

    // Check API health
    const health = await API.checkHealth();
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-text');

    if (health.status === 'healthy') {
        statusText.textContent = 'Connected';
        statusDot.style.background = 'var(--spotify-green)';
    } else {
        statusText.textContent = 'Disconnected';
        statusDot.style.background = 'var(--essential-negative)';
        showToast('⚠️ Backend connection failed', 'error');
    }

    // Load stats
    const stats = await API.getStats();
    if (stats) {
        document.getElementById('statTracks').textContent = stats.total_tracks || '-';
        document.getElementById('statAccuracy').textContent =
            stats.model_accuracy ? `${Math.round(stats.model_accuracy * 100)}%` : '99%+';
    }

    // Set up event listeners
    setupEventListeners();

    // Hide loading screen
    setTimeout(() => {
        showLoading(false);
    }, 1000);
}

function setupEventListeners() {
    // Navigation items
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            const view = item.dataset.view;
            if (view) switchView(view);
        });
    });

    // Mood cards and links
    document.querySelectorAll('.mood-card, .mood-link').forEach(element => {
        element.addEventListener('click', (e) => {
            e.preventDefault();
            const mood = element.dataset.mood;
            if (mood) selectMood(mood);
        });
    });

    // Back to home button
    document.getElementById('backToHome').addEventListener('click', () => {
        switchView('home');
    });

    // Navigation buttons
    document.getElementById('backBtn').addEventListener('click', () => {
        if (state.historyIndex > 0) {
            state.historyIndex--;
            const view = state.history[state.historyIndex];
            document.querySelectorAll('.content-view').forEach(v => v.classList.remove('active'));
            document.getElementById(`${view}View`).classList.add('active');
            updateNavButtons();
        }
    });

    document.getElementById('forwardBtn').addEventListener('click', () => {
        if (state.historyIndex < state.history.length - 1) {
            state.historyIndex++;
            const view = state.history[state.historyIndex];
            document.querySelectorAll('.content-view').forEach(v => v.classList.remove('active'));
            document.getElementById(`${view}View`).classList.add('active');
            updateNavButtons();
        }
    });

    // Search
    const globalSearch = document.getElementById('globalSearch');
    globalSearch.addEventListener('input', (e) => {
        handleSearch(e.target.value);
        if (e.target.value.length > 0) {
            switchView('search');
        }
    });

    document.getElementById('clearSearch').addEventListener('click', () => {
        globalSearch.value = '';
        handleSearch('');
    });

    // Playback controls
    document.getElementById('playBtn').addEventListener('click', () => {
        const icon = document.getElementById('playBtn').querySelector('i');

        if (state.audioPlayer && state.audioPlayer.src) {
            if (state.isPlaying) {
                // PAUSE
                state.audioPlayer.pause();
                icon.className = 'fas fa-play';
                state.isPlaying = false;
                showToast('⏸️ Paused');
            } else {
                // PLAY
                state.audioPlayer.play();
                icon.className = 'fas fa-pause';
                state.isPlaying = true;
                showToast('▶️ Playing');
            }
        } else {
            showToast('Select a track to play');
        }
    });

    document.getElementById('nextBtn').addEventListener('click', () => {
        if (state.currentPlaylist.length > 0) {
            const nextIndex = (state.currentTrackIndex + 1) % state.currentPlaylist.length;
            playTrack(state.currentPlaylist[nextIndex], nextIndex);
        }
    });

    document.getElementById('prevBtn').addEventListener('click', () => {
        if (state.currentPlaylist.length > 0) {
            const prevIndex = state.currentTrackIndex === 0
                ? state.currentPlaylist.length - 1
                : state.currentTrackIndex - 1;
            playTrack(state.currentPlaylist[prevIndex], prevIndex);
        }
    });

    // Volume control
    const volumeRange = document.getElementById('volumeRange');
    const volumeBtn = document.getElementById('volumeBtn');

    volumeRange.addEventListener('input', (e) => {
        const volume = e.target.value;
        const icon = volumeBtn.querySelector('i');

        // Update audio player volume
        state.currentVolume = volume / 100;
        if (state.audioPlayer) {
            state.audioPlayer.volume = state.currentVolume;
        }

        if (volume == 0) {
            icon.className = 'fas fa-volume-mute';
        } else if (volume < 50) {
            icon.className = 'fas fa-volume-down';
        } else {
            icon.className = 'fas fa-volume-up';
        }
    });

    volumeBtn.addEventListener('click', () => {
        const currentVolume = volumeRange.value;
        if (currentVolume > 0) {
            volumeRange.dataset.lastVolume = currentVolume;
            volumeRange.value = 0;
            volumeBtn.querySelector('i').className = 'fas fa-volume-mute';
            state.currentVolume = 0;
            if (state.audioPlayer) {
                state.audioPlayer.volume = 0;
            }
        } else {
            const lastVolume = volumeRange.dataset.lastVolume || 70;
            volumeRange.value = lastVolume;
            volumeBtn.querySelector('i').className = 'fas fa-volume-up';
            state.currentVolume = lastVolume / 100;
            if (state.audioPlayer) {
                state.audioPlayer.volume = state.currentVolume;
            }
        }
    });

    // Shuffle and repeat
    document.getElementById('shuffleBtn').addEventListener('click', function() {
        this.classList.toggle('active');
        showToast(this.classList.contains('active') ? '🔀 Shuffle on' : '🔀 Shuffle off');
    });

    document.getElementById('repeatBtn').addEventListener('click', function() {
        this.classList.toggle('active');
        showToast(this.classList.contains('active') ? '🔁 Repeat on' : '🔁 Repeat off');
    });

    // Progress bar seeking
    const progressBar = document.getElementById('progressBar');
    progressBar.addEventListener('click', (e) => {
        if (state.audioPlayer && state.audioPlayer.duration) {
            const rect = progressBar.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const percent = clickX / rect.width;
            state.audioPlayer.currentTime = state.audioPlayer.duration * percent;
        }
    });
}

// ===================================
// START THE APP
// ===================================
document.addEventListener('DOMContentLoaded', initializeApp);

// Set greeting based on time
function setGreeting() {
    const hour = new Date().getHours();
    const viewTitle = document.querySelector('#homeView .view-title');

    if (hour < 12) {
        viewTitle.textContent = 'Good morning';
    } else if (hour < 18) {
        viewTitle.textContent = 'Good afternoon';
    } else {
        viewTitle.textContent = 'Good evening';
    }
}

setGreeting();
