// Sets API base URL automatically:
// - when running locally (localhost/127.0.0.1) the UI talks to the local FastAPI backend
// - otherwise it falls back to the production Hugging Face Space
(() => {
	const hfSpace = "https://yashhugs-context-aware-music-recommendation.hf.space";
	let api = hfSpace;

	try {
		const host = (window && window.location && window.location.hostname) || '';
		if (host === 'localhost' || host === '127.0.0.1' || host === '') {
			api = 'http://127.0.0.1:8000';
		}
	} catch (e) {
		// if anything goes wrong, default to the HF Space
		api = hfSpace;
	}

	// Export normalized URL (no trailing slash)
	window.API_BASE_URL = api.replace(/\/$/, '');
})();
