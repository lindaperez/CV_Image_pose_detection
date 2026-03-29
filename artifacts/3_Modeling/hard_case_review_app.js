(function () {
  "use strict";

  const PRIMARY_ISSUES = [
    {
      value: "visibility",
      title: "Occlusion or off-frame motion",
      description: "Important motion is blocked or leaves the frame.",
      examples: "Use for occlusion, off-frame limbs, cropped body parts, or portrait framing that hides the key phase.",
      tags: ["occlusion", "equipment_obstruction", "self_occlusion", "off_frame", "cropped_body", "portrait_framing"],
    },
    {
      value: "camera_viewpoint",
      title: "Camera motion or extreme viewpoint",
      description: "The camera or viewpoint makes the repetition hard to observe reliably.",
      examples: "Use for handheld camera drift, reframing, zoom changes, top view, oblique depth, or unstable side view.",
      tags: ["camera_motion", "reframe", "zoom_shift", "viewpoint_extreme", "depth_unclear", "side_view", "top_view"],
    },
    {
      value: "target_selection",
      title: "Multiple people or unclear target",
      description: "It is unclear which person should be counted.",
      examples: "Use when the main subject changes, another person is closer, or the background confuses target selection.",
      tags: ["multi_person", "target_switch", "background_person"],
    },
    {
      value: "rep_ambiguity",
      title: "Partial rep or unclear boundary",
      description: "The clip makes the repetition definition itself ambiguous.",
      examples: "Use for shallow depth, partial reps, holds, pauses, or unclear cycle boundaries.",
      tags: ["partial_rep", "shallow_depth", "not_full_extension", "pause_or_hold", "tempo_change", "boundary_unclear"],
    },
    {
      value: "execution_variation",
      title: "Execution variation or assistance",
      description: "The movement style is unusual enough to change the semantics.",
      examples: "Use for assisted motion, kipping pull-ups, momentum-driven reps, or modified form.",
      tags: ["assisted_motion", "kipping", "unusual_technique", "modified_form"],
    },
    {
      value: "label_mismatch",
      title: "Count label looks wrong or debatable",
      description: "The ground-truth count seems questionable, not just the model output.",
      examples: "Use when the annotation itself looks inconsistent with what a human would count.",
      tags: ["count_label_suspect", "annotation_disagreement"],
    },
    {
      value: "pose_failure",
      title: "Pose estimate breaks",
      description: "The pose frontend itself appears to fail on the motion.",
      examples: "Use when joints disappear, swap, jitter heavily, or fail to track the main body movement.",
      tags: ["pose_jitter", "missing_keypoints", "joint_swap", "low_confidence_pose"],
    },
    {
      value: "rgb_context_advantage",
      title: "RGB has extra context signal",
      description: "Raw video context clearly explains why RGB helps more than pose.",
      examples: "Use when appearance, equipment interaction, body orientation, or scene context is the key clue.",
      tags: ["scene_context", "equipment_context", "body_orientation", "appearance_cue"],
    },
    {
      value: "model_failure",
      title: "No obvious data issue, mostly model-side",
      description: "The clip looks countable and the branch still fails.",
      examples: "Use when there is no clear visibility or label problem and the miss looks like pure model error.",
      tags: ["no_clear_issue"],
    },
  ];

  const PRIMARY_ISSUE_BY_VALUE = Object.fromEntries(PRIMARY_ISSUES.map((item) => [item.value, item]));
  const REVIEW_STATUSES = ["pending", "reviewed", "confirmed"];
  const YES_NO_OPTIONS = ["", "yes", "no"];
  const FIELD_HELP = {
    manual_target_person_ok: "Mark yes when the video clearly shows the correct person to count. Mark no when another person, a target switch, or ambiguous framing makes the counted subject uncertain.",
  };
  const DEFAULT_MANIFEST_PATH = "./training_outputs/hard_case_review_manifest.csv";
  const DEFAULT_SERVER_SAVE_PATH = "artifacts/3_Modeling/training_outputs/hard_case_review_manifest.csv";
  const LOCAL_STORAGE_KEY = "hard_case_review_app_state_v1";
  const APP_PATH_MARKER = "/artifacts/3_Modeling/hard_case_review_app.html";
  const COCO_EDGES = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
    [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
    [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4], [3, 5], [4, 6],
  ];
  const POSE_CONF_THRESHOLD = 0.15;

  function inferRepoRootPath() {
    const pathname = window.location.pathname || "/";
    const markerIndex = pathname.indexOf(APP_PATH_MARKER);
    if (markerIndex >= 0) {
      return pathname.slice(0, markerIndex + 1);
    }
    return "/";
  }

  const REPO_ROOT_PATH = inferRepoRootPath();
  const DEFAULT_VIDEO_BASE_PATH = `${REPO_ROOT_PATH}Data/LLSP/video/`;
  const DEFAULT_POSE_BASE_PATH = `${REPO_ROOT_PATH}Data/LLSP/annotation_cleaned/pose_features/`;
  const DEFAULT_ANNOTATION_BASE_PATH = `${REPO_ROOT_PATH}Data/LLSP/annotation/`;
  const DEFAULT_BACKEND_API_BASE = inferDefaultBackendApiBase();

  function inferDefaultBackendApiBase() {
    const origin = window.location.origin && window.location.origin !== "null"
      ? window.location.origin
      : "http://127.0.0.1:8000";
    const cleanRoot = REPO_ROOT_PATH.endsWith("/") ? REPO_ROOT_PATH.slice(0, -1) : REPO_ROOT_PATH;
    return `${origin}${cleanRoot}/api`;
  }

  function normalizeBackendApiBase(value) {
    const raw = String(value || "").trim();
    if (!raw) {
      return DEFAULT_BACKEND_API_BASE;
    }
    let normalized = raw.replace(/\/+$/, "");
    if (!/\/api$/i.test(normalized)) {
      normalized = `${normalized}/api`;
    }
    return normalized;
  }

  function buildApiUrl(path) {
    const base = normalizeBackendApiBase(DOM.backendApiBase ? DOM.backendApiBase.value : DEFAULT_BACKEND_API_BASE);
    const cleanPath = String(path || "").replace(/^\/+/, "");
    return `${base}/${cleanPath}`;
  }

  const DOM = {
    manifestInput: document.getElementById("manifest-input"),
    loadDefaultBtn: document.getElementById("load-default-btn"),
    exportBtn: document.getElementById("export-btn"),
    downloadJsonBtn: document.getElementById("download-json-btn"),
    chooseAutosaveBtn: document.getElementById("choose-autosave-btn"),
    saveNowBtn: document.getElementById("save-now-btn"),
    serverSavePath: document.getElementById("server-save-path"),
    backendApiBase: document.getElementById("backend-api-base"),
    serverSaveLink: document.getElementById("server-save-link"),
    backendStatus: document.getElementById("backend-status"),
    autosaveStatus: document.getElementById("autosave-status"),
    videoBasePath: document.getElementById("video-base-path"),
    poseBasePath: document.getElementById("pose-base-path"),
    annotationBasePath: document.getElementById("annotation-base-path"),
    showPoseOverlay: document.getElementById("show-pose-overlay"),
    showAnnotationHud: document.getElementById("show-annotation-hud"),
    filterExercise: document.getElementById("filter-exercise"),
    filterStatus: document.getElementById("filter-status"),
    filterSearch: document.getElementById("filter-search"),
    caseList: document.getElementById("case-list"),
    mainInner: document.getElementById("main-inner"),
    statTotal: document.getElementById("stat-total"),
    statReviewed: document.getElementById("stat-reviewed"),
  };

  const state = {
    headers: [],
    rows: [],
    rowMap: new Map(),
    selectedKey: null,
    manifestName: "hard_case_review_manifest.csv",
    poseCache: new Map(),
    annotationCache: new Map(),
    poseAnimationFrameId: null,
    activePoseKey: null,
    autosaveSupported: typeof window.showSaveFilePicker === "function",
    autosaveHandle: null,
    backendAvailable: false,
  };

  window.__hardCaseReviewDebug = {
    REPO_ROOT_PATH,
    DEFAULT_VIDEO_BASE_PATH,
    DEFAULT_POSE_BASE_PATH,
    DEFAULT_ANNOTATION_BASE_PATH,
  };

  function makeRowKey(row, index) {
    return [row.source_run_name || "", row.name || "", String(index)].join("::");
  }

  function buildRepoHref(relativePath) {
    const clean = String(relativePath || "").trim().replace(/^\/+/, "");
    return clean ? `${REPO_ROOT_PATH}${clean}` : "#";
  }

  function buildManifestViewerHref(relativePath) {
    const clean = String(relativePath || "").trim().replace(/^\/+/, "");
    if (!clean) {
      return "#";
    }
    const params = new URLSearchParams({
      relative_path: clean,
      api_base: normalizeBackendApiBase(DOM.backendApiBase ? DOM.backendApiBase.value : DEFAULT_BACKEND_API_BASE),
    });
    return `${REPO_ROOT_PATH}artifacts/3_Modeling/hard_case_review_manifest_viewer.html?${params.toString()}`;
  }

  function updateServerSaveLink() {
    const relativePath = DOM.serverSavePath.value.trim();
    const href = buildManifestViewerHref(relativePath);
    DOM.serverSaveLink.href = href;
    DOM.serverSaveLink.textContent = "Check CSV";
    DOM.serverSaveLink.title = relativePath || "Saved review manifest";
  }

  function cleanRowsForExport() {
    return state.rows.map((row) => {
      const clone = { ...row };
      delete clone.__key;
      return clone;
    });
  }

  function buildCsvText() {
    return serializeCsv(state.headers, cleanRowsForExport());
  }

  function updateAutosaveUi(message) {
    if (message) {
      DOM.autosaveStatus.textContent = message;
      return;
    }
    const serverPath = DOM.serverSavePath.value.trim();
    if (state.backendAvailable && serverPath) {
      DOM.autosaveStatus.textContent = `Backend available at ${normalizeBackendApiBase(DOM.backendApiBase.value)}. Use Save now to write to ${serverPath}.`;
      return;
    }
    if (state.autosaveHandle) {
      DOM.autosaveStatus.textContent = `Save target ready: ${state.autosaveHandle.name}. Use Save now to write the current manifest.`;
      return;
    }
    if (!state.autosaveSupported) {
      DOM.autosaveStatus.textContent = state.backendAvailable
        ? "Backend save is available. Use Save now to write to the server path."
        : "Direct file save is unavailable in this browser and no backend was detected. Use Export updated CSV instead.";
      return;
    }
    DOM.autosaveStatus.textContent = "Changes stay in the browser draft while you navigate. Choose a CSV file or use the backend, then click Save now when you want to write them.";
  }

  function updateAutosaveControls() {
    const enabled = Boolean(state.rows.length);
    DOM.chooseAutosaveBtn.disabled = !enabled || !state.autosaveSupported;
    DOM.saveNowBtn.disabled = !enabled || (!state.backendAvailable && (!state.autosaveSupported || !state.autosaveHandle));
    updateServerSaveLink();
    updateAutosaveUi();
  }

  async function detectBackendAvailability() {
    try {
      const response = await fetch(buildApiUrl("health"), { cache: "no-store" });
      state.backendAvailable = response.ok;
    } catch (error) {
      state.backendAvailable = false;
    }
    DOM.backendStatus.textContent = state.backendAvailable
      ? `Backend status: connected (${normalizeBackendApiBase(DOM.backendApiBase.value)})`
      : `Backend status: not detected (${normalizeBackendApiBase(DOM.backendApiBase.value)})`;
    updateAutosaveControls();
  }

  function parseCsv(text) {
    const rows = [];
    let current = "";
    let row = [];
    let insideQuotes = false;

    for (let i = 0; i < text.length; i += 1) {
      const char = text[i];
      const next = text[i + 1];

      if (char === "\"") {
        if (insideQuotes && next === "\"") {
          current += "\"";
          i += 1;
        } else {
          insideQuotes = !insideQuotes;
        }
      } else if (char === "," && !insideQuotes) {
        row.push(current);
        current = "";
      } else if ((char === "\n" || char === "\r") && !insideQuotes) {
        if (char === "\r" && next === "\n") {
          i += 1;
        }
        row.push(current);
        current = "";
        if (row.length > 1 || row[0] !== "") {
          rows.push(row);
        }
        row = [];
      } else {
        current += char;
      }
    }

    if (current !== "" || row.length > 0) {
      row.push(current);
      rows.push(row);
    }

    if (!rows.length) {
      return { headers: [], records: [] };
    }

    const headers = rows[0];
    const records = rows.slice(1).map((values) => {
      const record = {};
      headers.forEach((header, index) => {
        record[header] = values[index] ?? "";
      });
      return record;
    });

    return { headers, records };
  }

  function serializeCsv(headers, rows) {
    const escapeValue = (value) => {
      const text = value == null ? "" : String(value);
      if (/[",\n\r]/.test(text)) {
        return `"${text.replace(/"/g, "\"\"")}"`;
      }
      return text;
    };

    const lines = [headers.map(escapeValue).join(",")];
    rows.forEach((row) => {
      lines.push(headers.map((header) => escapeValue(row[header] || "")).join(","));
    });
    return lines.join("\n");
  }

  function slugify(value) {
    return String(value || "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "");
  }

  function normalizeStatus(value) {
    const normalized = String(value || "").trim().toLowerCase();
    return normalized || "pending";
  }

  function reviewedCount() {
    return state.rows.filter((row) => ["reviewed", "confirmed"].includes(normalizeStatus(row.manual_review_status))).length;
  }

  function formatMetric(value) {
    if (value == null || value === "") {
      return "n/a";
    }
    const num = Number(value);
    return Number.isFinite(num) ? num.toFixed(2) : String(value);
  }

  function formatSeconds(value) {
    const seconds = Number(value);
    if (!Number.isFinite(seconds) || seconds < 0) {
      return "0:00";
    }
    const whole = Math.floor(seconds);
    const mins = Math.floor(whole / 60);
    const secs = String(whole % 60).padStart(2, "0");
    return `${mins}:${secs}`;
  }

  function buildVideoSrc(row) {
    const manifestVideoPath = String(row.video_path || "").trim();
    const normalizedManifestPath = manifestVideoPath.replace(/\\/g, "/");
    const repoVideoMarker = "/Data/LLSP/video/";
    const markerIndex = normalizedManifestPath.indexOf(repoVideoMarker);
    if (markerIndex >= 0) {
      return encodeURI(`${REPO_ROOT_PATH.replace(/\/$/, "")}${normalizedManifestPath.slice(markerIndex)}`);
    }
    if (/^https?:\/\//i.test(manifestVideoPath) || /^file:\/\//i.test(manifestVideoPath) || manifestVideoPath.startsWith("/Data/")) {
      if (manifestVideoPath.startsWith("/Data/")) {
        return encodeURI(`${REPO_ROOT_PATH.replace(/\/$/, "")}${manifestVideoPath}`);
      }
      return encodeURI(manifestVideoPath);
    }
    const base = DOM.videoBasePath.value.trim();
    const pathParts = manifestVideoPath.split(/[\\/]/).filter(Boolean);
    const fallbackName = pathParts.length ? pathParts[pathParts.length - 1] : "";
    const name = row.name || fallbackName;
    if (!base || !name) {
      return "";
    }
    const cleanBase = base.endsWith("/") ? base : `${base}/`;
    return encodeURI(`${cleanBase}${name}`);
  }

  function buildPoseSrc(row) {
    const base = DOM.poseBasePath.value.trim();
    const stem = String(row.name || "")
      .replace(/^.*[\\/]/, "")
      .replace(/\.[^.]+$/, "");
    if (!base || !stem) {
      return "";
    }
    const cleanBase = base.endsWith("/") ? base : `${base}/`;
    return encodeURI(`${cleanBase}${stem}.npy`);
  }

  function buildAnnotationSrc(row) {
    const base = DOM.annotationBasePath.value.trim();
    const split = String(row.split || "").trim().toLowerCase();
    if (!base || !split) {
      return "";
    }
    const cleanBase = base.endsWith("/") ? base : `${base}/`;
    return encodeURI(`${cleanBase}${split}.csv`);
  }

  function parseNpyHeader(buffer) {
    const view = new DataView(buffer);
    const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 6));
    if (magic !== "\u0093NUMPY") {
      throw new Error("Unsupported .npy file: invalid magic header");
    }
    const major = view.getUint8(6);
    const headerLength = major <= 1 ? view.getUint16(8, true) : view.getUint32(8, true);
    const headerOffset = major <= 1 ? 10 : 12;
    const headerText = new TextDecoder("latin1").decode(new Uint8Array(buffer, headerOffset, headerLength));
    const descrMatch = /'descr':\s*'([^']+)'/.exec(headerText);
    const shapeMatch = /'shape':\s*\(([^)]*)\)/.exec(headerText);
    if (!descrMatch || !shapeMatch) {
      throw new Error("Unsupported .npy header format");
    }
    const dtype = descrMatch[1];
    const shape = shapeMatch[1]
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean)
      .map((item) => Number(item));
    return {
      dtype,
      shape,
      dataOffset: headerOffset + headerLength,
    };
  }

  function parseFloat32Npy(buffer) {
    const meta = parseNpyHeader(buffer);
    if (!["<f4", "|f4", "f4"].includes(meta.dtype)) {
      throw new Error(`Unsupported pose dtype: ${meta.dtype}`);
    }
    if (meta.shape.length !== 2) {
      throw new Error(`Expected 2D pose array, got shape (${meta.shape.join(", ")})`);
    }
    const count = meta.shape[0] * meta.shape[1];
    const data = new Float32Array(buffer, meta.dataOffset, count);
    return {
      frames: meta.shape[0],
      featDim: meta.shape[1],
      data,
    };
  }

  async function getPoseData(row) {
    const poseSrc = buildPoseSrc(row);
    if (!poseSrc) {
      return { status: "missing", message: "No pose source path could be resolved.", poseSrc: "" };
    }
    if (state.poseCache.has(poseSrc)) {
      return state.poseCache.get(poseSrc);
    }
    const pending = { status: "loading", message: "Loading pose overlay...", poseSrc };
    state.poseCache.set(poseSrc, pending);
    try {
      const response = await fetch(poseSrc, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const buffer = await response.arrayBuffer();
      const npy = parseFloat32Npy(buffer);
      const result = {
        status: "ready",
        poseSrc,
        frames: npy.frames,
        featDim: npy.featDim,
        data: npy.data,
        message: `Pose overlay loaded (${npy.frames} detected frames, approximate time sync).`,
      };
      state.poseCache.set(poseSrc, result);
      return result;
    } catch (error) {
      const result = {
        status: "error",
        poseSrc,
        message: `Pose overlay unavailable: ${error.message}`,
      };
      state.poseCache.set(poseSrc, result);
      return result;
    }
  }

  async function getAnnotationLookup(split) {
    const splitKey = String(split || "").trim().toLowerCase();
    if (!splitKey) {
      return null;
    }
    const base = DOM.annotationBasePath.value.trim();
    const lookupKey = `${base}::${splitKey}`;
    if (state.annotationCache.has(lookupKey)) {
      return state.annotationCache.get(lookupKey);
    }
    const csvSrc = buildAnnotationSrc({ split: splitKey });
    const pending = fetch(csvSrc, { cache: "no-store" })
      .then((response) => {
        if (!response.ok) {
          throw new Error(`Could not load ${csvSrc} (${response.status})`);
        }
        return response.text();
      })
      .then((text) => {
        const parsed = parseCsv(text);
        const byName = new Map();
        parsed.records.forEach((record) => {
          if (record.name) {
            byName.set(record.name, record);
          }
        });
        return { csvSrc, byName };
      })
      .catch((error) => ({ csvSrc, error }));
    state.annotationCache.set(lookupKey, pending);
    return pending;
  }

  function parseAnnotationIntervals(annotationRow) {
    if (!annotationRow) {
      return [];
    }
    const intervals = [];
    for (let i = 1; i <= 302; i += 2) {
      const startRaw = annotationRow[`L${i}`];
      const endRaw = annotationRow[`L${i + 1}`];
      if (startRaw == null || endRaw == null || startRaw === "" || endRaw === "") {
        continue;
      }
      const startFrame = Number(startRaw);
      const endFrame = Number(endRaw);
      if (!Number.isFinite(startFrame) || !Number.isFinite(endFrame)) {
        continue;
      }
      intervals.push({
        repIndex: intervals.length + 1,
        startFrame,
        endFrame,
      });
    }
    return intervals;
  }

  async function getAnnotationData(row) {
    const splitLookup = await getAnnotationLookup(row.split);
    if (!splitLookup) {
      return { status: "missing", message: "No annotation split available.", intervals: [] };
    }
    if (splitLookup.error) {
      return {
        status: "error",
        message: `Annotation CSV unavailable: ${splitLookup.error.message}`,
        intervals: [],
      };
    }
    const annotationRow = splitLookup.byName.get(row.name);
    if (!annotationRow) {
      return {
        status: "missing",
        message: `No raw annotation row found for ${row.name}.`,
        intervals: [],
      };
    }
    const intervals = parseAnnotationIntervals(annotationRow);
    return {
      status: "ready",
      message: intervals.length
        ? `Loaded ${intervals.length} annotated repetition intervals from ${row.split}.csv.`
        : "No non-empty L* interval pairs found for this row.",
      intervals,
      annotationRow,
      annotationSrc: splitLookup.csvSrc,
    };
  }

  function findActiveInterval(frame, intervals) {
    return intervals.find((interval) => frame >= interval.startFrame && frame <= interval.endFrame) || null;
  }

  function intervalSeconds(interval, fps) {
    if (!Number.isFinite(fps) || fps <= 0) {
      return null;
    }
    return {
      startSec: interval.startFrame / fps,
      endSec: interval.endFrame / fps,
    };
  }

  function renderAnnotationIntervals(row, annotationData, activeInterval, currentFrame) {
    const container = document.getElementById("annotation-intervals");
    const sourceNode = document.getElementById("annotation-source");
    const currentFrameNode = document.getElementById("annotation-current-frame");
    const activeNode = document.getElementById("annotation-active-interval");
    const summaryNode = document.getElementById("annotation-interval-summary");
    if (!container || !sourceNode || !currentFrameNode || !activeNode || !summaryNode) {
      return;
    }

    currentFrameNode.textContent = Number.isFinite(currentFrame) ? String(Math.max(0, Math.round(currentFrame))) : "n/a";
    sourceNode.textContent = annotationData.message || "No annotation loaded.";

    const fps = Number(row.fps);
    if (!annotationData || annotationData.status !== "ready" || !annotationData.intervals.length) {
      activeNode.textContent = "none";
      summaryNode.textContent = "No annotation intervals available.";
      container.innerHTML = "";
      return;
    }

    if (activeInterval) {
      const secs = intervalSeconds(activeInterval, fps);
      activeNode.textContent = secs
        ? `rep ${activeInterval.repIndex} · frames ${activeInterval.startFrame}-${activeInterval.endFrame} · ${formatSeconds(secs.startSec)}-${formatSeconds(secs.endSec)}`
        : `rep ${activeInterval.repIndex} · frames ${activeInterval.startFrame}-${activeInterval.endFrame}`;
    } else {
      activeNode.textContent = "none";
    }

    summaryNode.textContent = `Annotated intervals: ${annotationData.intervals.length}`;
    container.innerHTML = annotationData.intervals
      .map((interval) => {
        const secs = intervalSeconds(interval, fps);
        const isActive = activeInterval && activeInterval.repIndex === interval.repIndex;
        return `
          <div class="annotation-interval-chip ${isActive ? "active" : ""}">
            <strong>Rep ${interval.repIndex}</strong>
            <span>frames ${interval.startFrame}-${interval.endFrame}</span>
            <span>${secs ? `${formatSeconds(secs.startSec)}-${formatSeconds(secs.endSec)}` : "time n/a"}</span>
          </div>
        `;
      })
      .join("");
  }

  function resizeOverlayCanvas(video, canvas) {
    const rect = video.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(rect.width * dpr));
    const height = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
    }
    return { width, height };
  }

  function clearOverlay(canvas) {
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
  }

  function drawPoseOverlay(video, canvas, poseData) {
    const context = canvas.getContext("2d");
    const { width, height } = resizeOverlayCanvas(video, canvas);
    context.clearRect(0, 0, width, height);

    if (!poseData || poseData.status !== "ready" || !DOM.showPoseOverlay.checked) {
      return;
    }
    if (!video.videoWidth || !video.videoHeight || !poseData.frames) {
      return;
    }

    const frameIndex = Math.max(
      0,
      Math.min(
        poseData.frames - 1,
        Math.round(((video.duration ? video.currentTime / video.duration : 0) || 0) * (poseData.frames - 1))
      )
    );
    const featDim = poseData.featDim;
    const frameOffset = frameIndex * featDim;
    const scaleX = width / video.videoWidth;
    const scaleY = height / video.videoHeight;

    context.lineWidth = Math.max(2, width / 320);
    context.strokeStyle = "rgba(249, 115, 22, 0.9)";
    context.fillStyle = "rgba(34, 197, 94, 0.92)";

    COCO_EDGES.forEach(([start, end]) => {
      const startConf = poseData.data[frameOffset + start * 3 + 2];
      const endConf = poseData.data[frameOffset + end * 3 + 2];
      if (startConf < POSE_CONF_THRESHOLD || endConf < POSE_CONF_THRESHOLD) {
        return;
      }
      const x1 = poseData.data[frameOffset + start * 3] * scaleX;
      const y1 = poseData.data[frameOffset + start * 3 + 1] * scaleY;
      const x2 = poseData.data[frameOffset + end * 3] * scaleX;
      const y2 = poseData.data[frameOffset + end * 3 + 1] * scaleY;
      context.beginPath();
      context.moveTo(x1, y1);
      context.lineTo(x2, y2);
      context.stroke();
    });

    for (let joint = 0; joint < featDim / 3; joint += 1) {
      const conf = poseData.data[frameOffset + joint * 3 + 2];
      if (conf < POSE_CONF_THRESHOLD) {
        continue;
      }
      const x = poseData.data[frameOffset + joint * 3] * scaleX;
      const y = poseData.data[frameOffset + joint * 3 + 1] * scaleY;
      context.beginPath();
      context.arc(x, y, Math.max(2.5, width / 180), 0, Math.PI * 2);
      context.fill();
    }
  }

  function stopPoseLoop() {
    if (state.poseAnimationFrameId != null) {
      cancelAnimationFrame(state.poseAnimationFrameId);
      state.poseAnimationFrameId = null;
    }
  }

  function updateAnnotationHud(video, row) {
    const hud = document.getElementById("annotation-hud");
    if (!hud) {
      return;
    }
    hud.classList.toggle("hidden", !DOM.showAnnotationHud.checked);
    const timeNode = document.getElementById("annotation-current-time");
    const durationNode = document.getElementById("annotation-duration");
    if (timeNode) {
      timeNode.textContent = formatSeconds(video.currentTime);
    }
    if (durationNode) {
      durationNode.textContent = formatSeconds(video.duration);
    }
    const manualIssueNode = document.getElementById("annotation-manual-issue");
    if (manualIssueNode) {
      manualIssueNode.textContent = row.manual_primary_issue || "unset";
    }
    const statusNode = document.getElementById("annotation-review-status");
    if (statusNode) {
      statusNode.textContent = normalizeStatus(row.manual_review_status);
    }
    const fpsNode = document.getElementById("annotation-fps");
    const frameNode = document.getElementById("annotation-frame");
    const fps = Number(row.fps);
    const currentFrame = Number.isFinite(fps) && fps > 0 ? video.currentTime * fps : NaN;
    if (fpsNode) {
      fpsNode.textContent = Number.isFinite(fps) ? fps.toFixed(2) : "n/a";
    }
    if (frameNode) {
      frameNode.textContent = Number.isFinite(currentFrame) ? String(Math.max(0, Math.round(currentFrame))) : "n/a";
    }
  }

  async function attachPoseOverlay(row) {
    stopPoseLoop();
    state.activePoseKey = row.__key;

    const video = document.getElementById("review-video");
    const canvas = document.getElementById("pose-overlay");
    const status = document.getElementById("pose-overlay-status");
    if (!video || !canvas || !status) {
      return;
    }

    const poseDataPromise = DOM.showPoseOverlay.checked ? getPoseData(row) : Promise.resolve({ status: "hidden", message: "Pose overlay hidden." });
    const annotationDataPromise = getAnnotationData(row);

    status.textContent = DOM.showPoseOverlay.checked ? "Loading pose overlay..." : "Pose overlay hidden.";
    const [poseData, annotationData] = await Promise.all([poseDataPromise, annotationDataPromise]);
    if (state.activePoseKey !== row.__key) {
      return;
    }
    status.textContent = poseData.message;
    const initialFrame = Number.isFinite(Number(row.fps)) ? 0 : NaN;
    renderAnnotationIntervals(row, annotationData, null, initialFrame);

    const renderFrame = () => {
      if (state.activePoseKey !== row.__key) {
        return;
      }
      drawPoseOverlay(video, canvas, poseData);
      updateAnnotationHud(video, row);
      const fps = Number(row.fps);
      const currentFrame = Number.isFinite(fps) && fps > 0 ? video.currentTime * fps : NaN;
      const activeInterval = annotationData && annotationData.status === "ready" ? findActiveInterval(currentFrame, annotationData.intervals) : null;
      renderAnnotationIntervals(row, annotationData, activeInterval, currentFrame);
      state.poseAnimationFrameId = requestAnimationFrame(renderFrame);
    };

    const triggerDraw = () => drawPoseOverlay(video, canvas, poseData);
    video.addEventListener("loadedmetadata", triggerDraw, { once: true });
    video.addEventListener("seeked", triggerDraw);
    triggerDraw();
    renderFrame();
  }

  function updateStats() {
    DOM.statTotal.textContent = String(state.rows.length);
    DOM.statReviewed.textContent = String(reviewedCount());
  }

  function updateExerciseOptions() {
    const current = DOM.filterExercise.value;
    const exercises = Array.from(new Set(state.rows.map((row) => row.type).filter(Boolean))).sort();
    DOM.filterExercise.innerHTML = `<option value="">All exercises</option>${exercises
      .map((exercise) => `<option value="${exercise}">${exercise}</option>`)
      .join("")}`;
    DOM.filterExercise.value = exercises.includes(current) ? current : "";
  }

  function filteredRows() {
    const exercise = DOM.filterExercise.value;
    const status = DOM.filterStatus.value;
    const search = DOM.filterSearch.value.trim().toLowerCase();

    return state.rows.filter((row) => {
      if (exercise && row.type !== exercise) {
        return false;
      }
      if (status && normalizeStatus(row.manual_review_status) !== status) {
        return false;
      }
      if (search && !String(row.name || "").toLowerCase().includes(search)) {
        return false;
      }
      return true;
    });
  }

  function ensureSelection() {
    const visible = filteredRows();
    if (!visible.length) {
      state.selectedKey = null;
      return;
    }
    if (!visible.some((row) => row.__key === state.selectedKey)) {
      state.selectedKey = visible[0].__key;
    }
  }

  function selectedRow() {
    return state.rowMap.get(state.selectedKey) || null;
  }

  function renderCaseList() {
    ensureSelection();
    const rows = filteredRows();

    if (!rows.length) {
      DOM.caseList.innerHTML = `
        <div class="empty-state">
          <h2>No matching rows</h2>
          <p>Adjust the filters or load a manifest with hard-case rows.</p>
        </div>
      `;
      renderMainPanel();
      return;
    }

    DOM.caseList.innerHTML = rows
      .map((row) => {
        const active = row.__key === state.selectedKey ? "active" : "";
        const status = normalizeStatus(row.manual_review_status);
        const statusClass = status === "pending" ? "warn" : "ok";
        return `
          <div class="case-item ${active}" data-row-key="${row.__key}">
            <div class="case-item-title">${row.name || "unnamed row"}</div>
            <div class="case-item-meta">${row.type || "unknown"} · priority ${row.review_priority || "?"} · ${row.model_outcome || "unknown"}</div>
            <div class="badge-row">
              <span class="badge ${statusClass}">${status}</span>
              <span class="badge">${row.audit_bucket || "no bucket"}</span>
              <span class="badge">${row.severity || "n/a"}</span>
            </div>
          </div>
        `;
      })
      .join("");

    DOM.caseList.querySelectorAll(".case-item").forEach((item) => {
      item.addEventListener("click", () => {
        state.selectedKey = item.dataset.rowKey;
        renderCaseList();
      });
    });

    renderMainPanel();
  }

  function optionMarkup(values, selectedValue) {
    return values
      .map((value) => {
        const label = value || "unset";
        const selected = value === selectedValue ? "selected" : "";
        return `<option value="${value}" ${selected}>${label}</option>`;
      })
      .join("");
  }

  function primaryIssueOptions(selectedValue) {
    const options = [`<option value="">unset</option>`];
    PRIMARY_ISSUES.forEach((issue) => {
      const selected = issue.value === selectedValue ? "selected" : "";
      options.push(`<option value="${issue.value}" ${selected}>${issue.value} — ${issue.title}</option>`);
    });
    return options.join("");
  }

  function currentTagSet(row) {
    return new Set(
      String(row.manual_issue_tags || "")
        .split(",")
        .map((tag) => tag.trim())
        .filter(Boolean)
    );
  }

  function issueGuideMarkup(primaryIssue) {
    const item = PRIMARY_ISSUE_BY_VALUE[primaryIssue];
    if (!item) {
      return `
        <div class="issue-guide">
          <h3>Primary issue guide</h3>
          <p>Select a primary issue to see the recommended interpretation and tags.</p>
        </div>
      `;
    }
    return `
      <div class="issue-guide">
        <h3>${item.value} — ${item.title}</h3>
        <p>${item.description}</p>
        <p><strong>When to use it:</strong> ${item.examples}</p>
      </div>
    `;
  }

  function tagGroupsMarkup(row) {
    const selectedPrimary = row.manual_primary_issue || "";
    const selectedTags = currentTagSet(row);

    return PRIMARY_ISSUES.map((issue) => {
      const primaryClass = issue.value === selectedPrimary ? "is-primary" : "";
      return `
        <section class="tag-group ${primaryClass}">
          <div class="tag-group-head">
            <div>
              <h3>${issue.title}</h3>
              <span>${issue.value}</span>
            </div>
            <span>${issue.description}</span>
          </div>
          <div class="tag-options">
            ${issue.tags
              .map((tag) => {
                const checked = selectedTags.has(tag) ? "checked" : "";
                const inputId = `${slugify(issue.value)}-${slugify(tag)}-${slugify(row.__key)}`;
                return `
                  <label class="tag-chip" for="${inputId}">
                    <input id="${inputId}" type="checkbox" data-tag="${tag}" ${checked}>
                    <span>${tag}</span>
                  </label>
                `;
              })
              .join("")}
          </div>
        </section>
      `;
    }).join("");
  }

  function renderMainPanel() {
    const row = selectedRow();
    if (!row) {
      DOM.mainInner.innerHTML = `
        <div class="empty-state">
          <h2>No row selected</h2>
          <p>Select a hard case on the left to start reviewing.</p>
        </div>
      `;
      return;
    }

    const videoSrc = buildVideoSrc(row);
    const poseSrc = buildPoseSrc(row);
    const selectedTags = Array.from(currentTagSet(row));
    DOM.mainInner.innerHTML = `
      <section class="hero">
        <div class="video-card">
          <h2>${row.name || "unnamed row"}</h2>
          <p>${row.type || "unknown"} · split ${row.split || "n/a"} · source run ${row.source_run_name || "n/a"}</p>
          <div class="badge-row">
            <span class="badge">${row.model_outcome || "unknown outcome"}</span>
            <span class="badge">${row.audit_bucket || "no heuristic bucket"}</span>
            <span class="badge">${row.severity || "n/a"}</span>
            <span class="badge ${normalizeStatus(row.manual_review_status) === "pending" ? "warn" : "ok"}">${normalizeStatus(row.manual_review_status)}</span>
          </div>
          <div class="video-stage">
            <video id="review-video" controls preload="metadata" playsinline src="${videoSrc}"></video>
            <canvas id="pose-overlay" class="pose-overlay"></canvas>
            <div id="annotation-hud" class="annotation-hud ${DOM.showAnnotationHud.checked ? "" : "hidden"}">
              <div class="annotation-hud-title">Clip annotation</div>
              <div class="annotation-hud-grid">
                <div><strong>Exercise</strong><span>${row.type || "n/a"}</span></div>
                <div><strong>Split</strong><span>${row.split || "n/a"}</span></div>
                <div><strong>True count</strong><span>${row.true_count || "n/a"}</span></div>
                <div><strong>Pose pred</strong><span>${row.pose_pred_count || "n/a"}</span></div>
                <div><strong>RGB pred</strong><span>${row.rgb_pred_count || "n/a"}</span></div>
                <div><strong>Outcome</strong><span>${row.model_outcome || "n/a"}</span></div>
                <div><strong>FPS</strong><span id="annotation-fps">${row.fps || "n/a"}</span></div>
                <div><strong>Frame</strong><span id="annotation-frame">n/a</span></div>
                <div><strong>Time</strong><span><span id="annotation-current-time">0:00</span> / <span id="annotation-duration">0:00</span></span></div>
                <div><strong>Review</strong><span id="annotation-review-status">${normalizeStatus(row.manual_review_status)}</span></div>
                <div style="grid-column: 1 / -1;"><strong>Primary issue</strong><span id="annotation-manual-issue">${row.manual_primary_issue || "unset"}</span></div>
              </div>
            </div>
          </div>
          <div id="pose-overlay-status" class="pose-status">Pose overlay pending.</div>
          <div class="video-fallback">
            Video path: <code>${videoSrc || "not resolved"}</code><br>
            <a href="${videoSrc || "#"}" target="_blank" rel="noopener noreferrer">Open video directly</a><br>
            Pose path: <code>${poseSrc || "not resolved"}</code><br>
            If the video does not load, adjust the base path at left or serve the repo root with <code>python3 artifacts/3_Modeling/hard_case_review_server.py --port 8000</code>.
          </div>
        </div>

        <div class="annotation-card">
          <h2>Original annotation intervals</h2>
          <div class="annotation-status">
            <div class="annotation-status-head">
              <strong>Interval review</strong>
              <span id="annotation-interval-summary" class="helper">Loading annotation intervals...</span>
            </div>
            <div class="annotation-status-grid">
              <div>
                <strong>Current frame</strong>
                <span id="annotation-current-frame">n/a</span>
              </div>
              <div>
                <strong>Active interval</strong>
                <span id="annotation-active-interval">none</span>
              </div>
              <div>
                <strong>Source</strong>
                <span id="annotation-source">Loading raw annotation row...</span>
              </div>
            </div>
            <div id="annotation-intervals" class="annotation-intervals"></div>
          </div>
        </div>
      </section>

      <section class="details-card">
          <h2>Audit context</h2>
          <div class="meta-grid">
            <div>
              <strong>True count</strong>
              <small>${row.true_count || "n/a"}</small>
            </div>
            <div>
              <strong>Pose abs error</strong>
              <small>${formatMetric(row.pose_abs_error)}</small>
            </div>
            <div>
              <strong>RGB abs error</strong>
              <small>${formatMetric(row.rgb_abs_error)}</small>
            </div>
            <div>
              <strong>Review priority</strong>
              <small>${row.review_priority || "n/a"}</small>
            </div>
            <div>
              <strong>Heuristic issue tags</strong>
              <small>${row.issue_tags || "n/a"}</small>
            </div>
            <div>
              <strong>Current manual tags</strong>
              <small>${selectedTags.length ? selectedTags.join(", ") : "none selected"}</small>
            </div>
          </div>
      </section>

      <section class="review-card">
        <h2>Manual review</h2>
        <p>Pick one main issue, add any secondary tags, and keep the booleans aligned with what you actually saw in the clip.</p>

        <div class="review-grid">
          <div>
            <label for="manual-review-status">manual_review_status</label>
            <select id="manual-review-status">
              ${optionMarkup(REVIEW_STATUSES, normalizeStatus(row.manual_review_status))}
            </select>
          </div>
          <div>
            <label for="manual-primary-issue">manual_primary_issue</label>
            <select id="manual-primary-issue">
              ${primaryIssueOptions(row.manual_primary_issue || "")}
            </select>
          </div>
        </div>

        <div id="issue-guide-wrap">
          ${issueGuideMarkup(row.manual_primary_issue || "")}
        </div>

        <div class="review-grid">
          <div>
            <label for="manual-target-person-ok" title="${FIELD_HELP.manual_target_person_ok}">manual_target_person_ok</label>
            <select id="manual-target-person-ok" title="${FIELD_HELP.manual_target_person_ok}">
              ${optionMarkup(YES_NO_OPTIONS, row.manual_target_person_ok || "")}
            </select>
          </div>
          <div>
            <label for="manual-count-label-ok">manual_count_label_ok</label>
            <select id="manual-count-label-ok">
              ${optionMarkup(YES_NO_OPTIONS, row.manual_count_label_ok || "")}
            </select>
          </div>
          <div>
            <label for="manual-rep-definition-ambiguous">manual_rep_definition_ambiguous</label>
            <select id="manual-rep-definition-ambiguous">
              ${optionMarkup(YES_NO_OPTIONS, row.manual_rep_definition_ambiguous || "")}
            </select>
          </div>
          <div>
            <label for="manual-visibility-issue-confirmed">manual_visibility_issue_confirmed</label>
            <select id="manual-visibility-issue-confirmed">
              ${optionMarkup(YES_NO_OPTIONS, row.manual_visibility_issue_confirmed || "")}
            </select>
          </div>
          <div>
            <label for="manual-pose-failure-confirmed">manual_pose_failure_confirmed</label>
            <select id="manual-pose-failure-confirmed">
              ${optionMarkup(YES_NO_OPTIONS, row.manual_pose_failure_confirmed || "")}
            </select>
          </div>
          <div>
            <label for="manual-rgb-context-advantage-confirmed">manual_rgb_context_advantage_confirmed</label>
            <select id="manual-rgb-context-advantage-confirmed">
              ${optionMarkup(YES_NO_OPTIONS, row.manual_rgb_context_advantage_confirmed || "")}
            </select>
          </div>
          <div>
            <label for="manual-keep-for-report">manual_keep_for_report</label>
            <select id="manual-keep-for-report">
              ${optionMarkup(YES_NO_OPTIONS, row.manual_keep_for_report || "")}
            </select>
          </div>
        </div>

        <div class="review-grid full">
          <div>
            <label>Secondary issue tags (multiple selection)</label>
            <div class="tag-groups">
              ${tagGroupsMarkup(row)}
            </div>
          </div>
          <div>
            <label for="manual-notes">manual_notes</label>
            <textarea id="manual-notes" placeholder="Short note about what you saw in the clip.">${row.manual_notes || ""}</textarea>
          </div>
        </div>

        <div class="status-bar">
          <div class="helper">Draft edits are kept in the browser while you navigate. Use Save now to write them through the backend or to your selected CSV.</div>
          <div class="row">
            <button id="prev-row-btn" class="secondary" type="button">Previous</button>
            <button id="save-row-btn" type="button">Save current review</button>
            <button id="next-row-btn" type="button">Next</button>
          </div>
        </div>
      </section>
    `;

    wireRowEvents(row);
    attachPoseOverlay(row);
  }

  function setRowField(row, field, value) {
    row[field] = value;
    persistDraft();
    updateStats();
  }

  function updateIssueTagsFromDom(row) {
    const checked = Array.from(document.querySelectorAll("[data-tag]:checked")).map((node) => node.dataset.tag);
    row.manual_issue_tags = checked.join(", ");
    persistDraft();
    renderMainPanel();
  }

  async function writeAutosaveFile() {
    if (!state.autosaveHandle) {
      throw new Error("No autosave file has been selected yet.");
    }
    const writable = await state.autosaveHandle.createWritable();
    await writable.write(buildCsvText());
    await writable.close();
  }

  async function chooseAutosaveFile() {
    if (!state.autosaveSupported) {
      window.alert("This browser does not support writable file handles. Use Export updated CSV instead.");
      updateAutosaveUi();
      return;
    }
    try {
      state.autosaveHandle = await window.showSaveFilePicker({
        suggestedName: state.manifestName || "hard_case_review_manifest.csv",
        types: [
          {
            description: "CSV files",
            accept: { "text/csv": [".csv"] },
          },
        ],
      });
      updateAutosaveUi(`Save target ready: ${state.autosaveHandle.name}`);
    } catch (error) {
      if (error && error.name === "AbortError") {
        updateAutosaveUi();
        return;
      }
      console.warn("Failed to choose autosave file", error);
      window.alert(`Could not select an autosave CSV.\n\n${error.message || error}`);
      updateAutosaveUi();
    }
  }

  async function saveManifestToServer() {
    const relativePath = DOM.serverSavePath.value.trim();
    if (!relativePath) {
      throw new Error("Server autosave path is empty.");
    }
    const response = await fetch(buildApiUrl("review-manifest/save"), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        relative_path: relativePath,
        csv_text: buildCsvText(),
      }),
    });
    let payload = null;
    try {
      payload = await response.json();
    } catch (error) {
      payload = null;
    }
    if (!response.ok || !payload || !payload.ok) {
      const message = payload && payload.error ? payload.error : `request failed with status ${response.status}`;
      throw new Error(message);
    }
    return payload;
  }

  async function performSave(trigger = "save") {
    const serverPath = DOM.serverSavePath.value.trim();
    if (state.backendAvailable && serverPath) {
      const payload = await saveManifestToServer();
      updateAutosaveUi(`Saved to ${payload.relative_path} at ${new Date().toLocaleTimeString()} (${trigger})`);
      return { ok: true, mode: "backend", payload };
    }
    if (!state.autosaveSupported || !state.autosaveHandle) {
      throw new Error("Neither backend save nor a writable CSV is available.");
    }
    await writeAutosaveFile();
    updateAutosaveUi(`Saved at ${new Date().toLocaleTimeString()} (${trigger})`);
    return { ok: true, mode: "file" };
  }

  async function saveNow() {
    try {
      if (!state.backendAvailable && !state.autosaveHandle && state.autosaveSupported) {
        await chooseAutosaveFile();
        if (!state.autosaveHandle) {
          return;
        }
      }
      if (!state.backendAvailable && !state.autosaveHandle && !state.autosaveSupported) {
        const message = "No backend save is available and this browser cannot write directly to a CSV. Use Export updated CSV or run the local review server.";
        updateAutosaveUi(message);
        window.alert(message);
        return;
      }
      await performSave("manual");
    } catch (error) {
      console.warn("Manual save failed", error);
      updateAutosaveUi("Manual save failed. Check the backend status or choose a writable CSV.");
      window.alert(`Save now failed.\n\n${error.message || error}`);
    }
  }

  function navigateRow(direction) {
    const rows = filteredRows();
    const index = rows.findIndex((row) => row.__key === state.selectedKey);
    if (index < 0) {
      return;
    }
    const nextIndex = Math.min(rows.length - 1, Math.max(0, index + direction));
    state.selectedKey = rows[nextIndex].__key;
    renderCaseList();
  }

  function wireRowEvents(row) {
    const bindSelect = (id, field) => {
      const node = document.getElementById(id);
      node.addEventListener("change", (event) => {
        setRowField(row, field, event.target.value);
        if (field === "manual_primary_issue") {
          renderMainPanel();
        } else {
          renderCaseList();
        }
      });
    };

    bindSelect("manual-review-status", "manual_review_status");
    bindSelect("manual-primary-issue", "manual_primary_issue");
    bindSelect("manual-target-person-ok", "manual_target_person_ok");
    bindSelect("manual-count-label-ok", "manual_count_label_ok");
    bindSelect("manual-rep-definition-ambiguous", "manual_rep_definition_ambiguous");
    bindSelect("manual-visibility-issue-confirmed", "manual_visibility_issue_confirmed");
    bindSelect("manual-pose-failure-confirmed", "manual_pose_failure_confirmed");
    bindSelect("manual-rgb-context-advantage-confirmed", "manual_rgb_context_advantage_confirmed");
    bindSelect("manual-keep-for-report", "manual_keep_for_report");

    document.getElementById("manual-notes").addEventListener("input", (event) => {
      setRowField(row, "manual_notes", event.target.value);
    });

    document.querySelectorAll("[data-tag]").forEach((node) => {
      node.addEventListener("change", () => updateIssueTagsFromDom(row));
    });

    const saveRowBtn = document.getElementById("save-row-btn");
    saveRowBtn.disabled = !state.backendAvailable && !state.autosaveSupported;
    saveRowBtn.title = saveRowBtn.disabled
      ? "No backend save and no browser file-save support are available."
      : "Save the current review row and manifest.";
    saveRowBtn.addEventListener("click", () => saveNow());
    document.getElementById("prev-row-btn").addEventListener("click", () => navigateRow(-1));
    document.getElementById("next-row-btn").addEventListener("click", () => navigateRow(1));
  }

  function setManifest(headers, rows, manifestName) {
    state.headers = headers;
    state.rows = rows.map((row, index) => {
      const next = { ...row };
      next.manual_review_status = normalizeStatus(next.manual_review_status);
      next.__key = makeRowKey(next, index);
      return next;
    });
    state.rowMap = new Map(state.rows.map((row) => [row.__key, row]));
    state.manifestName = manifestName || "hard_case_review_manifest.csv";
    updateExerciseOptions();
    updateStats();
    ensureSelection();
    DOM.exportBtn.disabled = !state.rows.length;
    DOM.downloadJsonBtn.disabled = !state.rows.length;
    updateAutosaveControls();
    renderCaseList();
    persistDraft();
  }

  function parseManifestText(text, manifestName) {
    const { headers, records } = parseCsv(text);
    if (!headers.length) {
      throw new Error("The CSV appears to be empty.");
    }
    const required = ["name", "manual_review_status", "manual_primary_issue", "manual_issue_tags"];
    const missing = required.filter((field) => !headers.includes(field));
    if (missing.length) {
      throw new Error(`Missing expected manifest columns: ${missing.join(", ")}`);
    }
    setManifest(headers, records, manifestName);
  }

  function downloadBlob(filename, content, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    URL.revokeObjectURL(url);
  }

  function exportCsv() {
    if (!state.rows.length) {
      return;
    }
    downloadBlob(state.manifestName, buildCsvText(), "text/csv;charset=utf-8");
  }

  function exportJsonSnapshot() {
    if (!state.rows.length) {
      return;
    }
    const payload = {
      manifest_name: state.manifestName,
      video_base_path: DOM.videoBasePath.value.trim(),
      exported_at: new Date().toISOString(),
      rows: state.rows.map((row) => {
        const clone = { ...row };
        delete clone.__key;
        return clone;
      }),
    };
    downloadBlob(
      state.manifestName.replace(/\.csv$/i, "") + ".json",
      JSON.stringify(payload, null, 2),
      "application/json;charset=utf-8"
    );
  }

  function persistDraft() {
    if (!state.rows.length) {
      return;
    }
    const payload = {
      manifestName: state.manifestName,
      videoBasePath: DOM.videoBasePath.value.trim(),
      poseBasePath: DOM.poseBasePath.value.trim(),
      annotationBasePath: DOM.annotationBasePath.value.trim(),
      serverSavePath: DOM.serverSavePath.value.trim(),
      backendApiBase: normalizeBackendApiBase(DOM.backendApiBase.value),
      showPoseOverlay: Boolean(DOM.showPoseOverlay.checked),
      showAnnotationHud: Boolean(DOM.showAnnotationHud.checked),
      headers: state.headers,
      rows: state.rows.map((row) => {
        const clone = { ...row };
        delete clone.__key;
        return clone;
      }),
    };
    window.localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(payload));
  }

  function restoreDraft() {
    const raw = window.localStorage.getItem(LOCAL_STORAGE_KEY);
    if (!raw) {
      return;
    }
    try {
      const payload = JSON.parse(raw);
      if (!payload || !Array.isArray(payload.rows) || !Array.isArray(payload.headers)) {
        return;
      }
      if (payload.videoBasePath) {
        DOM.videoBasePath.value =
          payload.videoBasePath === "../../Data/LLSP/video/" || payload.videoBasePath === "/Data/LLSP/video/"
            ? DEFAULT_VIDEO_BASE_PATH
            : payload.videoBasePath;
      }
      if (payload.poseBasePath) {
        DOM.poseBasePath.value =
          payload.poseBasePath === "/Data/LLSP/annotation_cleaned/pose_features/"
            ? DEFAULT_POSE_BASE_PATH
            : payload.poseBasePath;
      }
      if (payload.annotationBasePath) {
        DOM.annotationBasePath.value =
          payload.annotationBasePath === "/Data/LLSP/annotation/"
            ? DEFAULT_ANNOTATION_BASE_PATH
            : payload.annotationBasePath;
      }
      if (payload.serverSavePath) {
        DOM.serverSavePath.value = payload.serverSavePath;
      }
      if (payload.backendApiBase) {
        DOM.backendApiBase.value = normalizeBackendApiBase(payload.backendApiBase);
      }
      if (typeof payload.showPoseOverlay === "boolean") {
        DOM.showPoseOverlay.checked = payload.showPoseOverlay;
      }
      if (typeof payload.showAnnotationHud === "boolean") {
        DOM.showAnnotationHud.checked = payload.showAnnotationHud;
      }
      setManifest(payload.headers, payload.rows, payload.manifestName || "hard_case_review_manifest.csv");
    } catch (error) {
      console.warn("Failed to restore review draft", error);
    }
  }

  async function loadDefaultManifest() {
    try {
      if (state.backendAvailable) {
        const relativePath = DOM.serverSavePath.value.trim() || DEFAULT_SERVER_SAVE_PATH;
        const response = await fetch(`${buildApiUrl("review-manifest/load")}?relative_path=${encodeURIComponent(relativePath)}`, {
          cache: "no-store",
        });
        if (response.ok) {
          const payload = await response.json();
          if (payload && payload.ok && typeof payload.csv_text === "string") {
            parseManifestText(payload.csv_text, payload.relative_path.split("/").pop() || "hard_case_review_manifest.csv");
            return;
          }
        }
      }
      const response = await fetch(DEFAULT_MANIFEST_PATH, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`Could not fetch ${DEFAULT_MANIFEST_PATH} (${response.status})`);
      }
      const text = await response.text();
      parseManifestText(text, "hard_case_review_manifest.csv");
    } catch (error) {
      window.alert(`Failed to load default manifest.\n\n${error.message}\n\nStart the review server or load the CSV manually.`);
    }
  }

  function readManifestFile(file) {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        parseManifestText(String(reader.result || ""), file.name);
      } catch (error) {
        window.alert(`Failed to parse manifest.\n\n${error.message}`);
      }
    };
    reader.readAsText(file);
  }

  function bindGlobalEvents() {
    DOM.manifestInput.addEventListener("change", (event) => {
      const file = event.target.files && event.target.files[0];
      if (file) {
        readManifestFile(file);
      }
    });

    DOM.loadDefaultBtn.addEventListener("click", loadDefaultManifest);
    DOM.exportBtn.addEventListener("click", exportCsv);
    DOM.downloadJsonBtn.addEventListener("click", exportJsonSnapshot);
    DOM.chooseAutosaveBtn.addEventListener("click", chooseAutosaveFile);
    DOM.saveNowBtn.addEventListener("click", saveNow);
    DOM.serverSavePath.addEventListener("change", () => {
      persistDraft();
      updateServerSaveLink();
      updateAutosaveUi();
    });
    DOM.backendApiBase.addEventListener("change", () => {
      DOM.backendApiBase.value = normalizeBackendApiBase(DOM.backendApiBase.value);
      persistDraft();
      updateServerSaveLink();
      detectBackendAvailability();
    });
    DOM.filterExercise.addEventListener("change", renderCaseList);
    DOM.filterStatus.addEventListener("change", renderCaseList);
    DOM.filterSearch.addEventListener("input", renderCaseList);
    DOM.videoBasePath.addEventListener("change", () => {
      persistDraft();
      renderMainPanel();
    });
    DOM.poseBasePath.addEventListener("change", () => {
      state.poseCache.clear();
      persistDraft();
      renderMainPanel();
    });
    DOM.annotationBasePath.addEventListener("change", () => {
      state.annotationCache.clear();
      persistDraft();
      renderMainPanel();
    });
    DOM.showPoseOverlay.addEventListener("change", () => {
      persistDraft();
      renderMainPanel();
    });
    DOM.showAnnotationHud.addEventListener("change", () => {
      persistDraft();
      renderMainPanel();
    });
  }

  bindGlobalEvents();
  if (!DOM.videoBasePath.value || DOM.videoBasePath.value === "/Data/LLSP/video/") {
    DOM.videoBasePath.value = DEFAULT_VIDEO_BASE_PATH;
  }
  if (!DOM.poseBasePath.value || DOM.poseBasePath.value === "/Data/LLSP/annotation_cleaned/pose_features/") {
    DOM.poseBasePath.value = DEFAULT_POSE_BASE_PATH;
  }
  if (!DOM.annotationBasePath.value || DOM.annotationBasePath.value === "/Data/LLSP/annotation/") {
    DOM.annotationBasePath.value = DEFAULT_ANNOTATION_BASE_PATH;
  }
  if (!DOM.serverSavePath.value) {
    DOM.serverSavePath.value = DEFAULT_SERVER_SAVE_PATH;
  }
  if (!DOM.backendApiBase.value) {
    DOM.backendApiBase.value = DEFAULT_BACKEND_API_BASE;
  } else {
    DOM.backendApiBase.value = normalizeBackendApiBase(DOM.backendApiBase.value);
  }
  restoreDraft();
  updateServerSaveLink();
  updateAutosaveControls();
  detectBackendAvailability();
})();
