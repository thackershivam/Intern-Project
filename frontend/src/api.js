export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

function toAbsoluteUrl(path) {
  if (!path) return "";
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${API_BASE_URL}${path}`;
}

export async function uploadNewspaper(file) {
  const form = new FormData();
  form.append("file", file);

  const response = await fetch(`${API_BASE_URL}/upload`, {
    method: "POST",
    body: form
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Upload failed");
  }

  const data = await response.json();
  return {
    ...data,
    image_url: toAbsoluteUrl(data.image_url)
  };
}

export async function ocrArticle({ sessionId, articleId, language }) {
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("article_id", articleId);
  form.append("language", language);

  const response = await fetch(`${API_BASE_URL}/ocr`, {
    method: "POST",
    body: form
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "OCR failed");
  }

  return response.json();
}

export async function generateAudio({ sessionId, articleId, language, text }) {
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("article_id", articleId);
  form.append("language", language);
  form.append("text", text);

  const response = await fetch(`${API_BASE_URL}/tts`, {
    method: "POST",
    body: form
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Audio generation failed");
  }

  const data = await response.json();
  return {
    ...data,
    audio_url: toAbsoluteUrl(data.audio_url)
  };
}

export async function deleteSession(sessionId) {
  if (!sessionId) return;
  await fetch(`${API_BASE_URL}/session/${sessionId}`, {
    method: "DELETE"
  }).catch(() => {});
}
