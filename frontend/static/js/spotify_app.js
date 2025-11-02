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
    currentVolume: 0.7,
    shuffleEnabled: false,
    repeatMode: 'off', // off | all | one
    libraryPlaylists: []
};

function updateVolumeSliderUI(rangeElement) {
    if (!rangeElement) return;
    const min = Number(rangeElement.min || 0);
    const max = Number(rangeElement.max || 100);
    const value = Number(rangeElement.value);
    const percentage = ((value - min) / (max - min)) * 100;
    rangeElement.style.background = `linear-gradient(90deg, rgba(255,255,255,0.95) ${percentage}%, rgba(255,255,255,0.2) ${percentage}%)`;
}

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
                body: JSON.stringify({ mood: mood, limit: 50 })
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
            const response = await fetch(`/api/search?query=${encodeURIComponent(query)}&limit=50`);
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
    if (state.currentView === viewName) {
        if (viewName === 'library') {
            renderLibrary();
        }
        return;
    }

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

    if (viewName === 'library') {
        renderLibrary();
    }
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

function getMoodGradient(mood) {
    const gradients = {
        workout: 'linear-gradient(135deg, #ff6b6b, #ee5a6f)',
        chill: 'linear-gradient(135deg, #4facfe, #00f2fe)',
        party: 'linear-gradient(135deg, #f093fb, #f5576c)',
        focus: 'linear-gradient(135deg, #43e97b, #38f9d7)',
        sleep: 'linear-gradient(135deg, #fa709a, #fee140)'
    };
    return gradients[mood] || gradients.chill;
}

function createPlaylistKey(mood, tracks) {
    const ids = tracks.map(track => track.id || track.name || '').join('|');
    return `${mood || 'unknown'}:${ids}`;
}

function isCurrentPlaylistSaved() {
    if (!state.currentPlaylist.length) return false;
    const key = createPlaylistKey(state.currentMood, state.currentPlaylist);
    return state.libraryPlaylists.some(playlist => playlist.key === key);
}

function updateSaveButtonState() {
    const saveBtn = document.querySelector('.btn-save');
    if (!saveBtn) return;

    const icon = saveBtn.querySelector('i');
    const isSaved = isCurrentPlaylistSaved();

    if (isSaved) {
        saveBtn.classList.add('active');
        if (icon) icon.className = 'fas fa-heart';
    } else {
        saveBtn.classList.remove('active');
        if (icon) icon.className = 'far fa-heart';
    }
}

function persistLibrary() {
    try {
        localStorage.setItem('vibesync_library_v1', JSON.stringify(state.libraryPlaylists));
    } catch (error) {
        console.warn('Failed to persist library:', error);
    }
}

function loadLibraryFromStorage() {
    try {
        const raw = localStorage.getItem('vibesync_library_v1');
        if (!raw) return;
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed)) {
            state.libraryPlaylists = parsed;
        }
    } catch (error) {
        console.warn('Failed to load library:', error);
        state.libraryPlaylists = [];
    }
}

function formatRelativeDate(isoString) {
    if (!isoString) return '';
    const date = new Date(isoString);
    if (Number.isNaN(date.getTime())) return '';

    const now = new Date();
    const diffMs = now - date;
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays <= 0) return 'Saved today';
    if (diffDays === 1) return 'Saved yesterday';
    if (diffDays < 7) return `Saved ${diffDays} days ago`;

    const options = { year: 'numeric', month: 'short', day: 'numeric' };
    return `Saved ${date.toLocaleDateString(undefined, options)}`;
}

function renderLibrary() {
    const container = document.getElementById('libraryContainer');
    const emptyState = document.getElementById('libraryEmptyState');

    if (!container || !emptyState) return;

    container.innerHTML = '';

    if (!state.libraryPlaylists.length) {
        container.classList.add('hidden');
        emptyState.classList.remove('hidden');
        return;
    }

    container.classList.remove('hidden');
    emptyState.classList.add('hidden');

    state.libraryPlaylists
        .slice()
        .sort((a, b) => new Date(b.savedAt) - new Date(a.savedAt))
        .forEach(playlist => {
            const card = document.createElement('div');
            card.className = 'library-card';
            card.dataset.id = playlist.id;
            card.innerHTML = `
                <div class="library-card-bg" style="background: ${getMoodGradient(playlist.mood)}"></div>
                <div class="library-card-content">
                    <div class="library-card-heading">
                        <span class="library-card-type">${getMoodTitle(playlist.mood)}</span>
                        <span class="library-card-meta">${formatRelativeDate(playlist.savedAt)}</span>
                    </div>
                    <h3 class="library-card-title">${playlist.title}</h3>
                    <p class="library-card-description">${playlist.description}</p>
                    <div class="library-card-footer">
                        <span class="library-card-count">${playlist.tracks.length} songs</span>
                        <div class="library-card-actions">
                            <button class="library-card-btn" data-action="play" data-id="${playlist.id}">
                                <i class="fas fa-play"></i>
                                Play
                            </button>
                            <button class="library-card-btn" data-action="delete" data-id="${playlist.id}">
                                <i class="fas fa-trash"></i>
                                Remove
                            </button>
                        </div>
                    </div>
                </div>
            `;
            container.appendChild(card);
        });
}

function getNextTrackIndex(direction = 1) {
    const total = state.currentPlaylist.length;
    if (!total) return -1;

    if (direction === 1) {
        if (state.repeatMode === 'one') {
            return state.currentTrackIndex;
        }

        if (state.shuffleEnabled) {
            const candidates = state.currentPlaylist
                .map((_, idx) => idx)
                .filter(idx => idx !== state.currentTrackIndex);

            if (candidates.length === 0) {
                return state.repeatMode === 'off' ? -1 : state.currentTrackIndex;
            }
            const randomIndex = Math.floor(Math.random() * candidates.length);
            return candidates[randomIndex];
        }

        const nextIndex = state.currentTrackIndex + 1;
        if (nextIndex < total) return nextIndex;
        return state.repeatMode === 'all' ? 0 : -1;
    }

    // direction === -1
    if (state.shuffleEnabled) {
        const candidates = state.currentPlaylist
            .map((_, idx) => idx)
            .filter(idx => idx !== state.currentTrackIndex);

        if (candidates.length === 0) {
            return state.repeatMode === 'off' ? -1 : state.currentTrackIndex;
        }
        const randomIndex = Math.floor(Math.random() * candidates.length);
        return candidates[randomIndex];
    }

    const prevIndex = state.currentTrackIndex - 1;
    if (prevIndex >= 0) return prevIndex;
    return state.repeatMode === 'all' ? total - 1 : 0;
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
    playlistCover.style.background = getMoodGradient(mood);

    // Render tracks
    const tracksContainer = document.getElementById('tracksContainer');
    tracksContainer.innerHTML = '';

    tracks.forEach((track, index) => {
        const trackRow = createTrackRow(track, index + 1);
        tracksContainer.appendChild(trackRow);
    });

    updateSaveButtonState();

    // Switch to playlist view with history support
    switchView('playlist');

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

    resetProgressBar();

    // REAL AUDIO PLAYBACK
    if (track.preview_url) {
        // Initialize audio player if not exists
        if (!state.audioPlayer) {
            state.audioPlayer = new Audio();
            state.audioPlayer.volume = state.currentVolume;

            // Auto-advance to next track when preview ends
            state.audioPlayer.addEventListener('ended', () => {
                if (!state.currentPlaylist.length) {
                    state.isPlaying = false;
                    playBtn.querySelector('i').className = 'fas fa-play';
                    resetProgressBar();
                    return;
                }

                const nextIndex = getNextTrackIndex(1);
                if (nextIndex === -1) {
                    state.isPlaying = false;
                    playBtn.querySelector('i').className = 'fas fa-play';
                    resetProgressBar();
                    return;
                }

                playTrack(state.currentPlaylist[nextIndex], nextIndex);
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

function resetProgressBar() {
    const progressFill = document.getElementById('progressFill');
    const progressHandle = document.getElementById('progressHandle');
    const currentTimeEl = document.getElementById('currentTime');

    if (progressFill) progressFill.style.width = '0%';
    if (progressHandle) progressHandle.style.left = '0%';
    if (currentTimeEl) currentTimeEl.textContent = '0:00';
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

    loadLibraryFromStorage();
    renderLibrary();
    updateSaveButtonState();

    state.history = ['home'];
    state.historyIndex = 0;
    state.currentView = 'home';
    updateNavButtons();

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

    // Play All button
    document.querySelector('.btn-play-all')?.addEventListener('click', () => {
        if (state.currentPlaylist.length > 0) {
            playTrack(state.currentPlaylist[0], 0);
        } else {
            showToast('No tracks to play', 'error');
        }
    });

    const saveBtn = document.querySelector('.btn-save');
    if (saveBtn) {
        saveBtn.addEventListener('click', () => {
            if (!state.currentPlaylist.length) {
                showToast('Generate a playlist first', 'error');
                return;
            }

            const key = createPlaylistKey(state.currentMood, state.currentPlaylist);
            const existingIndex = state.libraryPlaylists.findIndex(item => item.key === key);

            if (existingIndex >= 0) {
                state.libraryPlaylists.splice(existingIndex, 1);
                showToast('Removed from Your Library');
            } else {
                const playlistSnapshot = state.currentPlaylist.map(track => JSON.parse(JSON.stringify(track)));
                state.libraryPlaylists.push({
                    id: `pl-${Date.now()}`,
                    key,
                    mood: state.currentMood,
                    title: getMoodTitle(state.currentMood),
                    description: getMoodDescription(state.currentMood),
                    tracks: playlistSnapshot,
                    savedAt: new Date().toISOString()
                });
                showToast('Saved to Your Library');
            }

            persistLibrary();
            renderLibrary();
            updateSaveButtonState();
        });
    }

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

            document.querySelectorAll('.nav-item').forEach(item => {
                item.classList.toggle('active', item.dataset.view === view);
            });

            state.currentView = view;
            updateNavButtons();

            if (view === 'library') {
                renderLibrary();
            } else if (view === 'playlist') {
                updateSaveButtonState();
            }
        }
    });

    document.getElementById('forwardBtn').addEventListener('click', () => {
        if (state.historyIndex < state.history.length - 1) {
            state.historyIndex++;
            const view = state.history[state.historyIndex];
            document.querySelectorAll('.content-view').forEach(v => v.classList.remove('active'));
            document.getElementById(`${view}View`).classList.add('active');
            document.querySelectorAll('.nav-item').forEach(item => {
                item.classList.toggle('active', item.dataset.view === view);
            });

            state.currentView = view;
            updateNavButtons();

            if (view === 'library') {
                renderLibrary();
            } else if (view === 'playlist') {
                updateSaveButtonState();
            }
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
            const nextIndex = getNextTrackIndex(1);
            if (nextIndex === -1) {
                showToast('End of playlist');
                return;
            }
            playTrack(state.currentPlaylist[nextIndex], nextIndex);
        }
    });

    document.getElementById('prevBtn').addEventListener('click', () => {
        if (state.currentPlaylist.length > 0) {
            const prevIndex = getNextTrackIndex(-1);
            if (prevIndex === -1) {
                showToast('Start of playlist');
                return;
            }
            playTrack(state.currentPlaylist[prevIndex], prevIndex);
        }
    });

    // Volume control
    const volumeRange = document.getElementById('volumeRange');
    const volumeBtn = document.getElementById('volumeBtn');

    updateVolumeSliderUI(volumeRange);

    volumeRange.addEventListener('input', (e) => {
        const volume = e.target.value;
        const icon = volumeBtn.querySelector('i');

        // Update audio player volume
        state.currentVolume = volume / 100;
        if (state.audioPlayer) {
            state.audioPlayer.volume = state.currentVolume;
        }

        updateVolumeSliderUI(volumeRange);

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
            updateVolumeSliderUI(volumeRange);
        } else {
            const lastVolume = volumeRange.dataset.lastVolume || 70;
            volumeRange.value = lastVolume;
            volumeBtn.querySelector('i').className = 'fas fa-volume-up';
            state.currentVolume = lastVolume / 100;
            if (state.audioPlayer) {
                state.audioPlayer.volume = state.currentVolume;
            }
            updateVolumeSliderUI(volumeRange);
        }
    });

    // Shuffle and repeat
    const shuffleBtn = document.getElementById('shuffleBtn');
    shuffleBtn.addEventListener('click', function() {
        state.shuffleEnabled = !state.shuffleEnabled;
        this.classList.toggle('active', state.shuffleEnabled);
        showToast(state.shuffleEnabled ? '🔀 Shuffle on' : '🔀 Shuffle off');
    });

    const repeatBtn = document.getElementById('repeatBtn');
    const repeatModes = ['off', 'all', 'one'];
    repeatBtn.addEventListener('click', function() {
        const currentIndex = repeatModes.indexOf(state.repeatMode);
        const nextMode = repeatModes[(currentIndex + 1) % repeatModes.length];
        state.repeatMode = nextMode;

        this.classList.remove('active', 'repeat-one');

        if (nextMode === 'all') {
            this.classList.add('active');
            showToast('🔁 Repeat all');
        } else if (nextMode === 'one') {
            this.classList.add('active', 'repeat-one');
            showToast('� Repeat one');
        } else {
            showToast('Repeat off');
        }
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

    const libraryContainer = document.getElementById('libraryContainer');
    if (libraryContainer) {
        libraryContainer.addEventListener('click', (event) => {
            const actionBtn = event.target.closest('[data-action]');
            if (!actionBtn) return;

            const playlistId = actionBtn.dataset.id;
            const playlist = state.libraryPlaylists.find(item => item.id === playlistId);
            if (!playlist) return;

            if (actionBtn.dataset.action === 'play') {
                displayPlaylist(playlist.mood, playlist.tracks);
                showToast('Loaded saved playlist');
            } else if (actionBtn.dataset.action === 'delete') {
                state.libraryPlaylists = state.libraryPlaylists.filter(item => item.id !== playlistId);
                persistLibrary();
                renderLibrary();
                updateSaveButtonState();
                showToast('Removed from Your Library');
            }
        });
    }
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
