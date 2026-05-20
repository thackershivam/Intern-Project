import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Download,
  FileImage,
  Languages,
  Loader2,
  MousePointer2,
  Newspaper,
  Play,
  Sparkles,
  Upload
} from "lucide-react";
import { deleteSession, generateAudio, ocrArticle, uploadNewspaper } from "./api.js";

const LANGUAGES = ["Gujarati", "Hindi", "English"];

function pointInArticle(point, article) {
  if (!point || !article) return false;
  return (
    point.x >= article.x &&
    point.x <= article.x + article.w &&
    point.y >= article.y &&
    point.y <= article.y + article.h
  );
}

function getCanvasPoint(event, canvas, imageSize) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = imageSize.width / rect.width;
  const scaleY = imageSize.height / rect.height;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY
  };
}

function App() {
  const canvasRef = useRef(null);
  const previewCanvasRef = useRef(null);
  const imageRef = useRef(null);

  const [file, setFile] = useState(null);
  const [language, setLanguage] = useState("Gujarati");
  const [sessionId, setSessionId] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [articles, setArticles] = useState([]);
  const [hoveredId, setHoveredId] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [articleText, setArticleText] = useState("");
  const [audioUrl, setAudioUrl] = useState("");
  const [status, setStatus] = useState("Upload a newspaper to begin.");
  const [loading, setLoading] = useState(false);
  const [ocrLoading, setOcrLoading] = useState(false);
  const [audioLoading, setAudioLoading] = useState(false);
  const [error, setError] = useState("");

  const selectedArticle = useMemo(
    () => articles.find((article) => article.id === selectedId) || null,
    [articles, selectedId]
  );

  const hoveredArticle = useMemo(
    () => articles.find((article) => article.id === hoveredId) || null,
    [articles, hoveredId]
  );

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image || !imageSize.width || !imageSize.height) return;

    const ctx = canvas.getContext("2d");
    canvas.width = imageSize.width;
    canvas.height = imageSize.height;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, imageSize.width, imageSize.height);

    articles.forEach((article) => {
      const isHovered = article.id === hoveredId;
      const isSelected = article.id === selectedId;
      const color = isSelected ? "#22c55e" : isHovered ? "#38bdf8" : "#f97316";
      const fill = isSelected
        ? "rgba(34, 197, 94, 0.18)"
        : isHovered
          ? "rgba(56, 189, 248, 0.22)"
          : "rgba(249, 115, 22, 0.08)";

      ctx.save();
      ctx.fillStyle = fill;
      ctx.strokeStyle = color;
      ctx.lineWidth = isHovered || isSelected ? 7 : 4;
      ctx.shadowColor = isHovered || isSelected ? color : "transparent";
      ctx.shadowBlur = isHovered || isSelected ? 18 : 0;
      ctx.fillRect(article.x, article.y, article.w, article.h);
      ctx.strokeRect(article.x, article.y, article.w, article.h);

      const label = `Article ${article.id}`;
      ctx.font = "bold 26px Inter, system-ui, sans-serif";
      const labelWidth = ctx.measureText(label).width + 24;
      ctx.shadowBlur = 0;
      ctx.fillStyle = color;
      ctx.fillRect(article.x, article.y, labelWidth, 42);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, article.x + 12, article.y + 29);
      ctx.restore();
    });
  }, [articles, hoveredId, selectedId, imageSize]);

  const drawPreview = useCallback(() => {
    const preview = previewCanvasRef.current;
    const image = imageRef.current;
    if (!preview || !image || !selectedArticle) return;

    const maxWidth = 520;
    const scale = Math.min(1, maxWidth / selectedArticle.w);
    preview.width = Math.max(1, Math.round(selectedArticle.w * scale));
    preview.height = Math.max(1, Math.round(selectedArticle.h * scale));
    const ctx = preview.getContext("2d");
    ctx.clearRect(0, 0, preview.width, preview.height);
    ctx.drawImage(
      image,
      selectedArticle.x,
      selectedArticle.y,
      selectedArticle.w,
      selectedArticle.h,
      0,
      0,
      preview.width,
      preview.height
    );
  }, [selectedArticle]);

  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  useEffect(() => {
    drawPreview();
  }, [drawPreview]);

  useEffect(() => {
    return () => {
      if (sessionId) deleteSession(sessionId);
    };
  }, [sessionId]);

  async function handleUpload() {
    if (!file) {
      setError("Please choose a newspaper image or PDF.");
      return;
    }

    if (sessionId) {
      await deleteSession(sessionId);
    }

    setLoading(true);
    setError("");
    setArticleText("");
    setAudioUrl("");
    setSelectedId(null);
    setHoveredId(null);
    setStatus("Uploading and detecting article regions...");

    try {
      const data = await uploadNewspaper(file);
      const image = new Image();
      image.crossOrigin = "anonymous";
      image.onload = () => {
        imageRef.current = image;
        setSessionId(data.session_id);
        setImageUrl(data.image_url);
        setImageSize({ width: data.width, height: data.height });
        setArticles(data.articles || []);
        setStatus(`Detected ${data.articles?.length || 0} article regions. Hover and click an article.`);
        setLoading(false);
      };
      image.onerror = () => {
        setLoading(false);
        setError("Could not load processed newspaper image.");
      };
      image.src = data.image_url;
    } catch (err) {
      setLoading(false);
      setError(err.message);
      setStatus("Upload failed.");
    }
  }

  function handleMouseMove(event) {
    if (!canvasRef.current || !imageSize.width) return;
    const point = getCanvasPoint(event, canvasRef.current, imageSize);
    const match = articles.find((article) => pointInArticle(point, article));
    setHoveredId(match ? match.id : null);
  }

  function handleCanvasClick() {
    if (!hoveredArticle) return;
    setSelectedId(hoveredArticle.id);
    setArticleText("");
    setAudioUrl("");
    setStatus(`Selected Article ${hoveredArticle.id}. Click OCR to extract text.`);
  }

  async function handleOcr() {
    if (!sessionId || !selectedArticle) {
      setError("Select an article first.");
      return;
    }
    setOcrLoading(true);
    setError("");
    setArticleText("");
    setAudioUrl("");
    setStatus(`Running OCR on Article ${selectedArticle.id} only...`);
    try {
      const data = await ocrArticle({
        sessionId,
        articleId: selectedArticle.id,
        language
      });
      setArticleText(data.text);
      setStatus("OCR complete. You can generate audio now.");
    } catch (err) {
      setError(err.message);
      setStatus("OCR failed.");
    } finally {
      setOcrLoading(false);
    }
  }

  async function handleAudio() {
    if (!sessionId || !selectedArticle || !articleText.trim()) {
      setError("Extract text before generating audio.");
      return;
    }
    setAudioLoading(true);
    setError("");
    setStatus("Generating voice audio...");
    try {
      const data = await generateAudio({
        sessionId,
        articleId: selectedArticle.id,
        language,
        text: articleText
      });
      setAudioUrl(data.audio_url);
      setStatus("Audio ready.");
    } catch (err) {
      setError(err.message);
      setStatus("Audio generation failed.");
    } finally {
      setAudioLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-white/10 bg-slate-900/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          <div>
            <div className="flex items-center gap-3">
              <div className="rounded-2xl bg-sky-500/20 p-3 text-sky-300">
                <Newspaper size={30} />
              </div>
              <div>
                <h1 className="text-2xl font-bold tracking-tight">
                  Interactive Newspaper Article Highlighter
                </h1>
                <p className="text-sm text-slate-400">
                  Hover, highlight, click, OCR, and listen to articles.
                </p>
              </div>
            </div>
          </div>
          <div className="hidden items-center gap-2 rounded-full border border-emerald-400/30 bg-emerald-400/10 px-4 py-2 text-sm text-emerald-200 md:flex">
            <Sparkles size={16} />
            OCR only selected article
          </div>
        </div>
      </header>

      <main className="mx-auto grid max-w-7xl gap-6 px-6 py-6 lg:grid-cols-[360px_1fr]">
        <aside className="space-y-5">
          <section className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-2xl">
            <h2 className="mb-4 flex items-center gap-2 text-lg font-semibold">
              <Upload size={20} /> Upload
            </h2>
            <label className="flex cursor-pointer flex-col items-center justify-center rounded-2xl border border-dashed border-sky-400/50 bg-sky-400/10 p-6 text-center transition hover:bg-sky-400/15">
              <FileImage className="mb-3 text-sky-300" size={34} />
              <span className="font-medium">Choose JPEG, PNG, or PDF</span>
              <span className="mt-1 text-xs text-slate-400">{file ? file.name : "No file selected"}</span>
              <input
                type="file"
                accept=".jpg,.jpeg,.png,.pdf"
                className="hidden"
                onChange={(event) => setFile(event.target.files?.[0] || null)}
              />
            </label>

            <div className="mt-5">
              <label className="mb-2 flex items-center gap-2 text-sm font-medium text-slate-300">
                <Languages size={16} /> OCR / Voice Language
              </label>
              <select
                value={language}
                onChange={(event) => setLanguage(event.target.value)}
                className="w-full rounded-xl border border-white/10 bg-slate-900 px-4 py-3 text-slate-100 outline-none ring-sky-400 transition focus:ring-2"
              >
                {LANGUAGES.map((item) => (
                  <option key={item}>{item}</option>
                ))}
              </select>
            </div>

            <button
              onClick={handleUpload}
              disabled={loading || !file}
              className="mt-5 flex w-full items-center justify-center gap-2 rounded-xl bg-sky-500 px-4 py-3 font-semibold text-white transition hover:bg-sky-400 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? <Loader2 className="animate-spin" size={18} /> : <Upload size={18} />}
              {loading ? "Detecting..." : "Upload & Detect Articles"}
            </button>

            <p className="mt-4 rounded-xl bg-slate-900/80 p-3 text-sm text-slate-300">{status}</p>
            {error && <p className="mt-3 rounded-xl bg-red-500/15 p-3 text-sm text-red-200">{error}</p>}
          </section>

          <section className="rounded-3xl border border-white/10 bg-white/5 p-5">
            <h2 className="mb-3 flex items-center gap-2 text-lg font-semibold">
              <MousePointer2 size={20} /> Interaction
            </h2>
            <ul className="space-y-2 text-sm text-slate-300">
              <li>1. Move mouse over the newspaper canvas.</li>
              <li>2. The article under cursor glows blue.</li>
              <li>3. Click to select; selected article turns green.</li>
              <li>4. Run OCR and optional speech generation.</li>
            </ul>
          </section>

          <section className="rounded-3xl border border-white/10 bg-white/5 p-5">
            <h2 className="mb-3 text-lg font-semibold">Selected Article</h2>
            {selectedArticle ? (
              <div>
                <p className="mb-3 text-sm text-slate-300">
                  Article {selectedArticle.id} - {selectedArticle.w} x {selectedArticle.h}px
                </p>
                <canvas ref={previewCanvasRef} className="max-h-72 w-full rounded-xl bg-white object-contain" />
                <button
                  onClick={handleOcr}
                  disabled={ocrLoading}
                  className="mt-4 flex w-full items-center justify-center gap-2 rounded-xl bg-emerald-500 px-4 py-3 font-semibold text-white transition hover:bg-emerald-400 disabled:opacity-50"
                >
                  {ocrLoading && <Loader2 className="animate-spin" size={18} />}
                  Extract Article Text
                </button>
              </div>
            ) : (
              <p className="text-sm text-slate-400">No article selected yet.</p>
            )}
          </section>
        </aside>

        <section className="space-y-6">
          <div className="rounded-3xl border border-white/10 bg-white/5 p-4 shadow-2xl">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
              <div>
                <h2 className="text-xl font-semibold">Interactive Newspaper Viewer</h2>
                <p className="text-sm text-slate-400">
                  {hoveredArticle ? `Hovering Article ${hoveredArticle.id}` : "Move cursor over an article box"}
                </p>
              </div>
              <div className="rounded-full bg-slate-900 px-4 py-2 text-sm text-slate-300">
                {articles.length} article boxes
              </div>
            </div>
            <div className="overflow-auto rounded-2xl bg-slate-900 p-3">
              {imageUrl ? (
                <canvas
                  ref={canvasRef}
                  onMouseMove={handleMouseMove}
                  onMouseLeave={() => setHoveredId(null)}
                  onClick={handleCanvasClick}
                  className="max-h-[78vh] w-full cursor-crosshair rounded-xl bg-white shadow-glow"
                />
              ) : (
                <div className="flex min-h-[520px] items-center justify-center rounded-xl border border-dashed border-white/10 text-slate-500">
                  Upload a newspaper to display interactive boxes.
                </div>
              )}
            </div>
          </div>

          <div className="grid gap-6 xl:grid-cols-[1fr_360px]">
            <section className="rounded-3xl border border-white/10 bg-white/5 p-5">
              <div className="mb-3 flex items-center justify-between">
                <h2 className="text-xl font-semibold">Extracted Text</h2>
                {articleText && (
                  <span className="rounded-full bg-emerald-400/10 px-3 py-1 text-sm text-emerald-200">
                    {articleText.length} chars
                  </span>
                )}
              </div>
              <textarea
                value={articleText}
                onChange={(event) => setArticleText(event.target.value)}
                placeholder="OCR result will appear here after clicking an article and running OCR."
                className="h-80 w-full resize-y rounded-2xl border border-white/10 bg-slate-950 p-4 text-sm leading-6 text-slate-100 outline-none ring-sky-400 transition focus:ring-2"
              />
            </section>

            <section className="rounded-3xl border border-white/10 bg-white/5 p-5">
              <h2 className="mb-4 flex items-center gap-2 text-xl font-semibold">
                <Play size={20} /> Audio
              </h2>
              <button
                onClick={handleAudio}
                disabled={audioLoading || !articleText.trim()}
                className="flex w-full items-center justify-center gap-2 rounded-xl bg-violet-500 px-4 py-3 font-semibold text-white transition hover:bg-violet-400 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {audioLoading ? <Loader2 className="animate-spin" size={18} /> : <Play size={18} />}
                Generate Voice
              </button>
              {audioUrl ? (
                <div className="mt-5 space-y-4">
                  <audio controls src={audioUrl} className="w-full" />
                  <a
                    href={audioUrl}
                    download
                    className="flex w-full items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/10 px-4 py-3 font-semibold text-slate-100 transition hover:bg-white/15"
                  >
                    <Download size={18} />
                    Download MP3
                  </a>
                </div>
              ) : (
                <p className="mt-4 text-sm text-slate-400">
                  Generate audio after OCR. Uses gTTS with the selected language.
                </p>
              )}
            </section>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
