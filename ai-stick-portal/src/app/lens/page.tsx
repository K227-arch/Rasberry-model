"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import TopAppBar from "@/components/TopAppBar";
import BottomNavBar from "@/components/BottomNavBar";

type Direction = "English → Runyoro" | "Runyoro → English";

interface TextBlock {
  text: string;
  translation: string;
  bbox: number[][]; // [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
  confidence: number;
}

interface OCRResult {
  blocks: TextBlock[];
  image_width: number;
  image_height: number;
}

export default function LensPage() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const scanIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const scanningRef = useRef(false);

  const [cameraActive, setCameraActive] = useState(false);
  const [scanning, setScanning] = useState(false);
  const [direction, setDirection] = useState<Direction>("English → Runyoro");
  const [blocks, setBlocks] = useState<TextBlock[]>([]);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [error, setError] = useState<string>("");
  const [facingMode, setFacingMode] = useState<"environment" | "user">("environment");
  const [paused, setPaused] = useState(false);
  const [autoScan, setAutoScan] = useState(false);

  // Start camera
  const startCamera = useCallback(async () => {
    setError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode,
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current!.play();
        };
      }
      setCameraActive(true);
    } catch {
      setError(
        "Camera access denied. Please allow camera permissions in your browser settings."
      );
    }
  }, [facingMode]);

  // Stop camera
  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (scanIntervalRef.current) {
      clearInterval(scanIntervalRef.current);
      scanIntervalRef.current = null;
    }
    setCameraActive(false);
    setBlocks([]);
    setPaused(false);
    setAutoScan(false);
  }, []);

  // Flip camera
  const flipCamera = useCallback(() => {
    const wasActive = streamRef.current !== null;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    setFacingMode((prev) => (prev === "environment" ? "user" : "environment"));
    if (wasActive) {
      // Will restart via useEffect
    }
  }, []);

  // Restart camera when facingMode changes
  useEffect(() => {
    if (cameraActive && !streamRef.current) {
      startCamera();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [facingMode]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
      if (scanIntervalRef.current) {
        clearInterval(scanIntervalRef.current);
      }
    };
  }, []);

  // Capture a frame from video as base64
  const captureFrame = useCallback((): string | null => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return null;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;

    ctx.drawImage(video, 0, 0);
    return canvas.toDataURL("image/jpeg", 0.85);
  }, []);

  // Send frame to backend for OCR + translation
  const scanFrame = useCallback(async () => {
    if (paused || scanningRef.current) return;

    const imageData = captureFrame();
    if (!imageData) return;

    scanningRef.current = true;
    setScanning(true);
    try {
      const res = await fetch("/api/ocr", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageData, direction }),
      });

      if (!res.ok) {
        const err = await res.json();
        console.error("OCR error:", err);
        return;
      }

      const data: OCRResult = await res.json();
      setBlocks(data.blocks);
      setImageSize({ width: data.image_width, height: data.image_height });
    } catch (err) {
      console.error("OCR request failed:", err);
    } finally {
      scanningRef.current = false;
      setScanning(false);
    }
  }, [captureFrame, direction, paused]);

  // Single scan
  const doSingleScan = () => {
    scanFrame();
  };

  // Toggle auto scanning (every 3s)
  const toggleAutoScan = useCallback(() => {
    if (autoScan) {
      if (scanIntervalRef.current) {
        clearInterval(scanIntervalRef.current);
        scanIntervalRef.current = null;
      }
      setAutoScan(false);
    } else {
      scanFrame();
      scanIntervalRef.current = setInterval(scanFrame, 3500);
      setAutoScan(true);
    }
  }, [autoScan, scanFrame]);

  // Toggle pause
  const togglePause = () => {
    setPaused((prev) => !prev);
  };

  // Compute overlay style for a text block
  const getOverlayStyle = (bbox: number[][]) => {
    const container = overlayRef.current;
    if (!container || imageSize.width === 0) return { display: "none" as const };

    const displayedWidth = container.clientWidth;
    const displayedHeight = container.clientHeight;
    const scaleX = displayedWidth / imageSize.width;
    const scaleY = displayedHeight / imageSize.height;

    // bbox: [[x0,y0],[x1,y1],[x2,y2],[x3,y3]] — top-left, top-right, bottom-right, bottom-left
    const x0 = bbox[0][0];
    const y0 = bbox[0][1];
    const x1 = bbox[2][0];
    const y1 = bbox[2][1];

    return {
      left: `${x0 * scaleX}px`,
      top: `${y0 * scaleY}px`,
      width: `${(x1 - x0) * scaleX}px`,
      height: `${(y1 - y0) * scaleY}px`,
    };
  };

  return (
    <>
      <TopAppBar />
      <main className="mt-16 md:mt-0 flex-1 flex flex-col w-full px-margin-mobile pt-4 pb-36 md:pb-8 max-w-5xl md:mx-auto">
        {/* Header */}
        <div className="flex items-center gap-3 mb-4">
          <span className="material-symbols-outlined text-primary text-[28px]">
            photo_camera
          </span>
          <h1 className="text-headline-md text-on-background">Camera Lens</h1>
          <span className="ml-auto text-label-sm text-on-surface-variant bg-surface-container px-3 py-1 rounded-full">
            EasyOCR + NLLB
          </span>
        </div>

        {/* Direction selector */}
        <div className="flex items-center gap-3 mb-4 bg-surface-container-lowest rounded-xl p-3 premium-shadow border border-outline-variant/30">
          <span className="text-label-md text-on-surface-variant font-medium">
            Translate:
          </span>
          <button
            onClick={() =>
              setDirection((d) =>
                d === "English → Runyoro"
                  ? "Runyoro → English"
                  : "English → Runyoro"
              )
            }
            className="flex items-center gap-2 px-4 py-2 rounded-full bg-primary-fixed text-on-primary-fixed text-label-md font-semibold cursor-pointer active:scale-95 transition-transform"
          >
            {direction}
            <span className="material-symbols-outlined text-[18px]">swap_horiz</span>
          </button>
          {cameraActive && (
            <button
              onClick={toggleAutoScan}
              className={`ml-auto flex items-center gap-1 px-3 py-1.5 rounded-full text-label-sm font-medium cursor-pointer transition-colors ${
                autoScan
                  ? "bg-primary text-on-primary"
                  : "bg-surface-container text-on-surface-variant"
              }`}
            >
              <span className="material-symbols-outlined text-[16px]">
                {autoScan ? "stop_circle" : "play_circle"}
              </span>
              {autoScan ? "Auto ON" : "Auto OFF"}
            </button>
          )}
        </div>

        {/* Camera viewport */}
        <div className="relative w-full aspect-[4/3] md:aspect-video bg-black rounded-2xl overflow-hidden premium-shadow-lg">
          {!cameraActive ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-surface-container-highest/80">
              <span className="material-symbols-outlined text-primary text-[64px]">
                photo_camera
              </span>
              <p className="text-body-lg text-on-surface-variant text-center px-6">
                Point your camera at text to translate it in real-time
              </p>
              <button
                onClick={startCamera}
                className="bg-primary text-on-primary px-8 py-3 rounded-full text-label-lg font-semibold premium-shadow active:scale-95 transition-transform cursor-pointer"
              >
                Start Camera
              </button>
              {error && (
                <p className="text-error text-body-sm text-center px-6 mt-2">
                  {error}
                </p>
              )}
            </div>
          ) : (
            <>
              {/* Video feed with overlay */}
              <div ref={overlayRef} className="absolute inset-0">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="w-full h-full object-cover"
                />
                {/* Translation overlays positioned over detected text */}
                {blocks.map((block, i) => (
                  <div
                    key={`${i}-${block.text}`}
                    className="absolute pointer-events-none"
                    style={getOverlayStyle(block.bbox)}
                  >
                    <div className="bg-black/80 backdrop-blur-sm px-1.5 py-0.5 rounded w-full h-full flex items-center justify-center overflow-hidden">
                      <span className="text-white text-[clamp(9px,1.4vw,15px)] font-semibold leading-tight text-center">
                        {block.translation}
                      </span>
                    </div>
                  </div>
                ))}
              </div>

              {/* Scanning indicator */}
              {scanning && (
                <div className="absolute top-3 left-3 bg-primary/90 text-on-primary px-3 py-1 rounded-full text-label-sm flex items-center gap-1 z-10">
                  <span className="material-symbols-outlined text-[16px] animate-spin">
                    progress_activity
                  </span>
                  Processing...
                </div>
              )}

              {/* Paused indicator */}
              {paused && (
                <div className="absolute top-3 left-1/2 -translate-x-1/2 bg-surface-container/90 backdrop-blur-sm text-on-surface px-4 py-1.5 rounded-full text-label-sm font-medium z-10">
                  Paused
                </div>
              )}

              {/* Camera controls */}
              <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-3 z-10">
                <button
                  onClick={togglePause}
                  className="w-12 h-12 rounded-full bg-white/90 backdrop-blur-sm flex items-center justify-center premium-shadow cursor-pointer active:scale-90 transition-transform"
                  aria-label={paused ? "Resume" : "Pause"}
                >
                  <span className="material-symbols-outlined text-on-surface text-[24px]">
                    {paused ? "play_arrow" : "pause"}
                  </span>
                </button>
                <button
                  onClick={doSingleScan}
                  disabled={scanning}
                  className="w-16 h-16 rounded-full bg-primary text-on-primary flex items-center justify-center premium-shadow-lg cursor-pointer active:scale-90 transition-transform disabled:opacity-50"
                  aria-label="Scan text"
                >
                  <span className="material-symbols-outlined text-[32px]">
                    document_scanner
                  </span>
                </button>
                <button
                  onClick={flipCamera}
                  className="w-12 h-12 rounded-full bg-white/90 backdrop-blur-sm flex items-center justify-center premium-shadow cursor-pointer active:scale-90 transition-transform"
                  aria-label="Flip camera"
                >
                  <span className="material-symbols-outlined text-on-surface text-[24px]">
                    flip_camera_ios
                  </span>
                </button>
              </div>

              {/* Stop button */}
              <button
                onClick={stopCamera}
                className="absolute top-3 right-3 w-10 h-10 rounded-full bg-error/80 backdrop-blur-sm flex items-center justify-center cursor-pointer z-10"
                aria-label="Stop camera"
              >
                <span className="material-symbols-outlined text-white text-[20px]">
                  close
                </span>
              </button>
            </>
          )}

          {/* Hidden canvas for frame capture */}
          <canvas ref={canvasRef} className="hidden" />
        </div>

        {/* Detected text list */}
        {blocks.length > 0 && (
          <section className="mt-4">
            <h2 className="text-label-sm uppercase tracking-widest text-on-surface-variant mb-3">
              Detected & Translated ({blocks.length} blocks)
            </h2>
            <div className="flex flex-col gap-2">
              {blocks.map((block, i) => (
                <div
                  key={`list-${i}`}
                  className="bg-surface-container-lowest rounded-xl p-3 border border-outline-variant/30 flex flex-col gap-1"
                >
                  <div className="flex justify-between items-center">
                    <p className="text-body-sm text-outline">{block.text}</p>
                    <span className="text-label-sm text-outline bg-surface-container px-2 py-0.5 rounded-full">
                      {Math.round(block.confidence * 100)}%
                    </span>
                  </div>
                  <p className="text-body-md text-on-surface font-medium">
                    → {block.translation}
                  </p>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Instructions when camera is off */}
        {!cameraActive && (
          <section className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
            {[
              {
                icon: "photo_camera",
                title: "Point Camera",
                desc: "Aim at any English or Runyoro text — signs, documents, menus",
              },
              {
                icon: "text_fields",
                title: "EasyOCR Detect",
                desc: "Server-side OCR with OpenCV preprocessing for accuracy",
              },
              {
                icon: "translate",
                title: "Live Overlay",
                desc: "Translated text overlaid on the original using your NLLB model",
              },
            ].map((step) => (
              <div
                key={step.title}
                className="glass-card rounded-2xl p-4 border border-outline-variant flex items-start gap-3"
              >
                <div className="w-10 h-10 rounded-xl bg-primary-fixed/30 flex items-center justify-center flex-shrink-0">
                  <span className="material-symbols-outlined text-primary text-[22px]">
                    {step.icon}
                  </span>
                </div>
                <div>
                  <p className="text-label-md text-on-background font-semibold">
                    {step.title}
                  </p>
                  <p className="text-label-sm text-on-surface-variant">
                    {step.desc}
                  </p>
                </div>
              </div>
            ))}
          </section>
        )}
      </main>
      <BottomNavBar />
    </>
  );
}
