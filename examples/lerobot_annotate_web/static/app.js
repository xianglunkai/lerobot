const $ = (id) => document.getElementById(id);
const state = {
  dataset: null,
  episode: null,
  clips: {},
  playhead: 0,
  duration: 0,
  playing: false,
  parquetOffset: 0,
  parquetRows: [],
  job: null,
  logText: "",
  logChangedAt: 0,
  changeStamp: "",
};
const FEATURE_STYLES = {
  subtask: ["subtask"],
  plan: ["plan"],
  memory: ["memory"],
  task_aug: ["task_aug"],
  interjections: ["interjection", "say"],
  vqa: ["vqa"],
};

function escapeHtml(value) {
  return String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll('"', "&quot;");
}

async function api(path, body) {
  const response = await fetch(path, body ? {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body),
  } : undefined);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || response.statusText);
  return payload;
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function showPage(id) {
  document.querySelectorAll(".page").forEach((page) => { page.hidden = page.id !== id; });
  document.querySelectorAll(".nav-tab[data-page]").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.page === id);
  });
  document.querySelectorAll("#players video, #editPlayers video").forEach((video) => {
    if (!hostIsVisible(video.closest(".video-grid"))) releaseVideo(video);
  });
  const host = id === "playPage" ? $("players") : id === "editPage" ? $("editPlayers") : null;
  if (host && state.episode) engagePlayers(host);
}

document.querySelectorAll(".nav-tab[data-page]").forEach((tab) => {
  tab.onclick = () => showPage(tab.dataset.page);
});

function renderDataset(dataset) {
  state.dataset = dataset;
  const output = $("output");
  if (!output.value) output.value = `${dataset.root}_annotated`;
  $("datasetState").textContent = `${dataset.total_episodes} 条 · ${dataset.fps} fps`;
  $("loadMessage").textContent = dataset.has_language_columns ? "语言列已经在数据集里。" : "还没有语言列。自动标注完成并写回后会出现。";
  $("loadMessage").classList.remove("error");
  $("metrics").innerHTML = [
    ["版本", dataset.codebase_version],
    ["机器人", dataset.robot_type || "—"],
    ["轨迹 / 帧", `${dataset.total_episodes} / ${dataset.total_frames}`],
    ["任务", dataset.total_tasks],
  ].map(([name, value]) => `<div><dt>${name}</dt><dd>${escapeHtml(value)}</dd></div>`).join("");
  $("features").textContent = JSON.stringify(dataset.features, null, 2);
  const info = {
    codebase_version: dataset.codebase_version,
    fps: dataset.fps,
    robot_type: dataset.robot_type,
    total_episodes: dataset.total_episodes,
    total_frames: dataset.total_frames,
    total_tasks: dataset.total_tasks,
    splits: dataset.splits,
    data_path: dataset.data_path,
    video_path: dataset.video_path,
    tasks: dataset.tasks,
  };
  $("infoDict").textContent = JSON.stringify(info, null, 2);
  $("files").innerHTML = dataset.files.map((file) =>
    `<tr><td>${escapeHtml(file.path)}</td><td>${formatBytes(file.bytes)}</td></tr>`
  ).join("");
  const files = dataset.files.filter((file) => file.path.endsWith(".parquet") && file.path.startsWith("data/"));
  $("parquetFile").innerHTML = files.map((file) => `<option value="${escapeHtml(file.path)}">${escapeHtml(file.path)}</option>`).join("");
  $("parquetEpisode").innerHTML = `<option value="all">全部</option>` + dataset.episodes.map((episode) =>
    `<option value="${episode.index}">#${episode.index}</option>`
  ).join("");
  renderEpisodeButtons();
  state.parquetOffset = 0;
  if (files.length) loadParquet();
}

function renderEpisodeButtons() {
  const query = $("filter").value.trim().toLowerCase();
  const buttons = state.dataset.episodes.filter((episode) =>
    !query || String(episode.index).includes(query) || episode.task.toLowerCase().includes(query)
  ).map((episode) => {
    const mark = episode.annotated ? "已标" : "未标";
    return `<button type="button" class="episode" data-index="${episode.index}"><span>#${episode.index} ${mark}</span><small>${episode.length} 帧</small></button>`;
  }).join("");
  $("episodeList").innerHTML = buttons;
  $("editList").innerHTML = buttons;
  if (state.episode) {
    document.querySelectorAll(".episode").forEach((button) => {
      button.classList.toggle("selected", Number(button.dataset.index) === state.episode.episode_index);
    });
  }
  document.querySelectorAll(".episode").forEach((button) => {
    button.onclick = () => openEpisode(Number(button.dataset.index));
  });
}

async function openEpisode(index) {
  const episode = await api(`/api/episode?index=${index}`);
  state.episode = episode;
  state.clips = episode.video;
  const first = Object.values(episode.video)[0];
  state.duration = first ? Math.max(0, first.end - first.start) : 0;
  document.querySelectorAll(".episode").forEach((button) => {
    button.classList.toggle("selected", Number(button.dataset.index) === index);
  });
  $("playMeta").textContent = `episode ${index} · ${episode.task}`;
  $("editMeta").textContent = $("playMeta").textContent;
  mountPlayers($("players"), episode.video);
  mountPlayers($("editPlayers"), episode.video);
  renderTimeline(episode);
  renderEditors(episode);
  seek(0);
}

function hostIsVisible(host) {
  const page = host && host.closest(".page");
  return Boolean(page && !page.hidden);
}

function mountPlayers(host, videos) {
  host.innerHTML = Object.entries(videos).map(([key, video]) => `
    <div class="video-card">
      <strong>${escapeHtml(key)}</strong>
      <video data-start="${video.start}" data-end="${video.end}" data-src="/videos/${video.path}" preload="none"></video>
      <p class="muted video-note"></p>
    </div>
  `).join("");
  if (hostIsVisible(host)) engagePlayers(host);
}

function releaseVideo(video) {
  video.pause();
  if (!video.getAttribute("src")) return;
  video.removeAttribute("src");
  video.load();
}

function engagePlayers(host) {
  host.querySelectorAll("video").forEach((video) => {
    const note = video.parentElement.querySelector(".video-note");
    if (note) note.textContent = "";
    video.onerror = () => {
      if (video.dataset.retried === "1") {
        if (note) note.textContent = "这一路没有解出画面。";
        return;
      }
      video.dataset.retried = "1";
      if (note) note.textContent = "这一路跳转被中断，正在重新定位。";
      const src = video.dataset.src;
      releaseVideo(video);
      video.src = src;
      video.onloadedmetadata = () => placeVideo(video, state.playhead || 0);
    };
    if (video.getAttribute("src") !== video.dataset.src) video.src = video.dataset.src;
    if (video.readyState >= 1) placeVideo(video, state.playhead || 0);
    else video.onloadedmetadata = () => placeVideo(video, state.playhead || 0);
  });
}

function players() {
  return [...document.querySelectorAll("#players video, #editPlayers video")].filter((video) => hostIsVisible(video.closest(".video-grid")));
}

function placeVideo(video, local) {
  if (video.seeking) return;
  const start = Number(video.dataset.start);
  const end = Number(video.dataset.end);
  let target = start + Math.max(0, local);
  target = Math.min(target, Math.max(start, end - 0.05));
  if (Number.isFinite(video.duration) && video.duration > 0) {
    target = Math.min(target, Math.max(0, video.duration - 0.05));
  }
  if (Math.abs(video.currentTime - target) < 0.05) return;
  video.currentTime = target;
}

function localTime() {
  const video = players()[0];
  if (!video || video.readyState < 1 || video.seeking) return state.playhead || 0;
  return Math.max(0, video.currentTime - Number(video.dataset.start));
}

function seek(seconds) {
  const local = Math.min(Math.max(0, seconds), state.duration || 0);
  state.playhead = local;
  players().forEach((video) => {
    if (video.readyState >= 1) placeVideo(video, local);
    else video.onloadedmetadata = () => placeVideo(video, local);
  });
  $("seek").value = state.duration ? String(Math.round(local / state.duration * 1000)) : "0";
  $("clock").textContent = `${local.toFixed(1)} / ${state.duration.toFixed(1)} s`;
}

function setPlaying(playing) {
  state.playing = playing;
  const rate = Number($("rate").value);
  players().forEach((video) => {
    video.playbackRate = rate;
    if (playing) video.play();
    else video.pause();
  });
  $("play").textContent = playing ? "播放中" : "播放";
}

function tick() {
  if (!state.playing) return;
  const local = localTime();
  if (local >= state.duration - 0.05) {
    setPlaying(false);
    seek(0);
    return;
  }
  state.playhead = local;
  const master = players()[0];
  players().forEach((video) => {
    if (video === master || video.seeking) return;
    const expected = Number(video.dataset.start) + local;
    if (Math.abs(video.currentTime - expected) > 0.35) placeVideo(video, local);
  });
  $("seek").value = state.duration ? String(Math.round(local / state.duration * 1000)) : "0";
  $("clock").textContent = `${local.toFixed(1)} / ${state.duration.toFixed(1)} s`;
}
setInterval(tick, 200);

function renderTimeline(episode) {
  const rows = episode.persistent.filter((row) => row.style === "subtask");
  $("timeline").innerHTML = rows.map((row) =>
    `<button type="button" class="card" data-time="${row.timestamp}"><b>${escapeHtml(row.style)}</b> ${Number(row.timestamp).toFixed(1)}s ${escapeHtml(row.content)}</button>`
  ).join("") || `<p class="muted">这条还没有 subtask。标注写回之后可以点这里跳转。</p>`;
  $("timeline").querySelectorAll("button").forEach((button) => {
    button.onclick = () => seek(Number(button.dataset.time));
  });
}

function renderEditors(episode) {
  $("persistent").innerHTML = episode.persistent.map((row) => editorRow("persistent", row)).join("");
  $("events").innerHTML = episode.events.map((row) => editorRow("event", row)).join("");
  bindEditorRows();
}

function editorRow(kind, row) {
  const styles = kind === "persistent"
    ? ["subtask", "plan", "memory", "task_aug", "motion"]
    : ["interjection", "vqa", "trace", "say"];
  const options = styles.map((style) => `<option ${row.style === style ? "selected" : ""}>${style}</option>`).join("");
  const camera = kind === "event"
    ? `<input data-field="camera" value="${escapeHtml(row.camera || "")}" placeholder="vqa 需要相机名">`
    : "";
  return `<div class="edit-row ${kind === "event" ? "event" : ""}">
    <select data-field="style">${options}</select>
    <input data-field="role" value="${escapeHtml(row.role || "assistant")}">
    <input data-field="timestamp" value="${escapeHtml(row.timestamp ?? 0)}">
    ${camera}
    <textarea data-field="content">${escapeHtml(row.content || "")}</textarea>
    <button type="button" class="ghost drop">删除</button>
  </div>`;
}

function bindEditorRows() {
  document.querySelectorAll(".drop").forEach((button) => {
    button.onclick = () => button.closest(".edit-row").remove();
  });
}

function readEditors(kind) {
  const host = kind === "persistent" ? $("persistent") : $("events");
  return [...host.querySelectorAll(".edit-row")].map((row) => {
    const item = {};
    row.querySelectorAll("[data-field]").forEach((field) => { item[field.dataset.field] = field.value; });
    item.timestamp = Number(item.timestamp);
    if (kind === "persistent") item.camera = null;
    else item.camera = item.camera || null;
    return item;
  });
}

async function loadParquet() {
  const path = $("parquetFile").value;
  if (!path) return;
  const page = await api(`/api/parquet?path=${encodeURIComponent(path)}&offset=${state.parquetOffset}&limit=${$("pageSize").value}&episode=${$("parquetEpisode").value}`);
  state.parquetRows = page.rows;
  const columns = page.schema.map((field) => field.name);
  $("parquetHead").innerHTML = `<tr><th>#</th>${columns.map((name) => `<th>${escapeHtml(name)}</th>`).join("")}</tr>`;
  $("parquetBody").innerHTML = page.rows.map((row, position) =>
    `<tr data-position="${position}"><td>${row.index}</td>${columns.map((name) => `<td>${escapeHtml(row.cells[name] || "")}</td>`).join("")}</tr>`
  ).join("");
  $("parquetMeta").textContent = `${page.path} · ${page.rows_total} 行 · 本页从 ${page.offset} 起 · ${formatBytes(page.bytes)}`;
  $("parquetBody").querySelectorAll("tr").forEach((tr) => {
    tr.onclick = () => {
      $("parquetBody").querySelectorAll("tr").forEach((item) => item.classList.remove("picked"));
      tr.classList.add("picked");
      $("rowDict").textContent = JSON.stringify(state.parquetRows[Number(tr.dataset.position)].raw, null, 2);
    };
  });
  $("parquetPrev").disabled = page.offset <= 0;
  $("parquetNext").disabled = page.offset + page.rows.length >= page.rows_total;
  state.parquetNext = page.offset + page.limit < page.rows_total;
}

function targetEpisodes(spec, total) {
  const text = (spec || "all").trim();
  if (!text || text === "all") {
    return total ? Array.from({length: total}, (_, index) => index) : [];
  }
  const chosen = [];
  for (const part of text.split(",")) {
    const piece = part.trim();
    if (!piece) continue;
    if (piece.includes("-")) {
      const [startText, endText] = piece.split("-", 2);
      const start = Number(startText);
      const end = Number(endText);
      if (!Number.isInteger(start) || !Number.isInteger(end) || end < start) return null;
      for (let index = start; index <= end; index += 1) chosen.push(index);
    } else {
      const index = Number(piece);
      if (!Number.isInteger(index)) return null;
      chosen.push(index);
    }
  }
  return [...new Set(chosen)].sort((left, right) => left - right);
}

function selectionReport(staging, features, spec) {
  if (!staging) {
    return {ready: false, text: "加载数据集后，这里按当前勾选的特征和情节提示是否已经标完。", items: []};
  }
  const names = features.length ? features : [];
  if (!names.length) {
    return {ready: false, text: "先勾选至少一个特征。", items: []};
  }
  const targets = targetEpisodes(spec, staging.total_episodes || 0);
  if (targets === null) {
    return {ready: false, text: "情节范围无法识别。示例：all、0、0-2、0,3,5。", items: []};
  }
  if (!targets.length) {
    return {ready: false, text: "数据集里还没有情节。", items: []};
  }
  const styles = staging.styles || stylesFromEpisodes(staging);
  const items = names.map((name) => {
    const required = FEATURE_STYLES[name] || [name];
    const done = targets.filter((episode) => required.every((style) => (styles[style] || []).includes(episode))).length;
    return {name, done, total: targets.length, complete: done === targets.length};
  });
  const finished = items.every((item) => item.complete);
  const detail = items.map((item) => `${item.name} ${item.done}/${item.total}${item.complete ? " 已完成" : ""}`).join(" · ");
  const scope = targets.length === (staging.total_episodes || targets.length) ? "全部情节" : `${targets.length} 条情节`;
  let text = `${scope}：${detail}。`;
  if (finished) {
    text = staging.parquet_written
      ? `所选特征已标注完成（${scope}）。语言列已经在数据集里。`
      : `所选特征已标注完成（${scope}，暂存已齐）。语言列还没写入数据集，点「开始」会按当前勾选写入。`;
  }
  return {ready: finished, text, items};
}

function stylesFromEpisodes(staging) {
  const styles = {};
  (staging.episodes || []).forEach((episode) => {
    (episode.rows || []).forEach((row) => {
      const list = styles[row.style] || [];
      if (!list.includes(episode.episode)) list.push(episode.episode);
      styles[row.style] = list;
    });
  });
  return styles;
}

function renderSelection() {
  const features = selectedFeatures();
  const report = selectionReport(state.job && state.job.staging, features, $("spec").value);
  const status = $("featureStatus");
  status.textContent = report.text;
  status.classList.toggle("done", report.ready);
  status.classList.toggle("error", false);
  if (state.job && !state.job.alive) {
    const modules = state.job.staging && state.job.staging.modules;
    const total = state.job.staging && state.job.staging.total_episodes;
    $("jobState").textContent = report.ready
      ? "所选特征已标注完成"
      : (modules && modules.plan && modules.plan.done ? `暂存 plan ${modules.plan.done}/${total}` : "空闲");
  }
  $("bars").innerHTML = report.items.map((item) => {
    const width = item.total ? Math.round(item.done / item.total * 100) : 0;
    return `<p class="muted" style="margin-top:10px">${escapeHtml(item.name)} ${item.done}/${item.total}${item.complete ? " 已完成" : ""}</p><div class="progress${item.complete ? " done" : ""}"><span style="width:${width}%"></span></div>`;
  }).join("");
  return report;
}

function applyJobControls(job) {
  const alive = Boolean(job && job.alive);
  const running = Boolean(job && job.running);
  const paused = Boolean(job && job.paused);
  $("run").disabled = alive;
  $("jobPause").disabled = !running;
  $("jobResume").disabled = !paused;
  $("jobStop").disabled = !alive;
  $("jobPause").classList.toggle("ready", running);
  $("jobResume").classList.toggle("ready", paused);
  $("jobStop").classList.toggle("ready", alive);
  $("run").title = alive ? "任务进行中，开始已锁定" : "开始标注";
  $("jobPause").title = running ? "暂停当前这一条" : "只有正在跑的时候才能暂停";
  $("jobResume").title = paused ? "从暂停的地方继续" : "先点暂停，继续才会亮";
  $("jobStop").title = alive ? "结束进程，已经写好的暂存会留下" : "没有正在运行的任务";
}

function noteLog(job) {
  const text = job.log || "";
  if (text !== state.logText) {
    state.logText = text;
    state.logChangedAt = Date.now();
  } else if (!state.logChangedAt) {
    state.logChangedAt = Date.now();
  }
}

function renderLive(job) {
  const box = $("liveProgress");
  if (!box) return;
  const progress = job.progress || {};
  const last = progress.last;
  const phase = progress.phase || "plan";
  const total = progress.phase_total || (job.staging && job.staging.total_episodes) || 0;
  const done = last ? last.number : 0;
  const waited = Math.max(0, Math.round((Date.now() - (state.logChangedAt || Date.now())) / 1000));
  const features = job.features ? `特征 ${job.features}` : "";
  let line = "空闲。点「开始」才会跑。暂停、继续、结束要等任务起来。";
  if (job.paused) {
    line = `已暂停 pid ${job.pid}。${features}。${phase} 已完成 ${done}/${total || "?"}。点「继续」接着跑；「结束」会停掉进程。开始保持锁定。`;
  } else if (job.running) {
    if (last && total && done >= total) {
      line = `运行中 pid ${job.pid}。${features}。${phase} ${total}/${total} 已完成，进程还在，可能在切下一阶段或写 parquet。开始已锁定，暂停和结束可以点。`;
    } else {
      const current = Math.min(done + 1, total || done + 1);
      const prev = last ? `上一条 episode ${last.episode} 用时 ${last.seconds} 秒。` : "第一条还没有返回。";
      line = `运行中 pid ${job.pid}。${features}。${phase} 正在跑第 ${current}/${total || "?"} 条，本条已等待 ${waited} 秒。${prev}开始已锁定。暂停和结束可以点；继续要等暂停之后才亮。`;
    }
  } else if (progress.finished) {
    line = "标注进程已经结束。可以再点开始。";
  }
  box.textContent = line;
  const log = $("log");
  const lines = (job.log || "").trim().split("\n").filter(Boolean);
  const shown = lines.length ? lines.slice(-16).join("\n") : "还没有日志。";
  if (log.textContent !== shown) {
    log.textContent = shown;
    log.scrollTop = log.scrollHeight;
  }
}

function renderJob(job) {
  state.job = job;
  applyJobControls(job);
  noteLog(job);
  renderLive(job);
  const staging = job.staging;
  const total = staging && staging.total_episodes ? staging.total_episodes : 0;
  const modules = staging ? staging.modules : {};
  const stamp = staging
    ? `${(staging.episodes || []).length}:${JSON.stringify(staging.style_counts || {})}:${job.features || ""}`
    : "";
  if (stamp !== state.changeStamp) {
    state.changeStamp = stamp;
    renderChanges(staging, job.features);
  }
  const report = renderSelection();
  if (job.paused) {
    $("jobState").textContent = `已暂停 pid ${job.pid || ""}`;
    $("health").textContent = "已暂停";
    $("health").classList.remove("ok");
  } else if (job.running) {
    const where = job.external ? "命令行" : "本页";
    $("jobState").textContent = `${where}运行中 pid ${job.pid || ""}`;
    $("health").textContent = "运行中";
    $("health").classList.remove("ok");
  } else if (report.ready) {
    $("jobState").textContent = "所选特征已标注完成";
    $("health").textContent = "已连接";
    $("health").classList.add("ok");
  } else if (job.progress && job.progress.finished) {
    $("jobState").textContent = job.mode === "copy" ? `已写到 ${job.output}` : "已覆盖当前数据集";
    $("health").textContent = "已连接";
    $("health").classList.add("ok");
  } else {
    $("jobState").textContent = staging && modules.plan && modules.plan.done ? `暂存 plan ${modules.plan.done}/${total}` : "空闲";
    $("health").textContent = "已连接";
    $("health").classList.add("ok");
  }
}

function keptStyles(features) {
  if (!features || features === "all") return null;
  const styles = new Set();
  features.split(",").filter(Boolean).forEach((name) => {
    if (name === "interjections") {
      styles.add("interjection");
      styles.add("say");
    } else {
      styles.add(name);
    }
  });
  return styles;
}

function renderChanges(staging, features) {
  const keep = keptStyles(features);
  const chosen = keep ? `本次只写入 ${features}。未勾选的 style 留在暂存，不进 parquet。` : "";
  if (!staging || !staging.episodes || !staging.episodes.length) {
    $("changeSummary").textContent = chosen || "还没有暂存。点「开始」之后，这里按 episode、style 和将写入的语言列列出句子。";
    $("changeRows").innerHTML = "";
    return;
  }
  const counts = Object.entries(staging.style_counts || {}).map(([style, count]) => `${style} ${count}`).join(" · ");
  const disk = staging.parquet_written
    ? "语言列已经在 parquet 里。表中「已写入」表示该 episode 的首帧上能读到语言行。"
    : "action、state、图像和原来的任务句没有改。下面这些句子还在暂存里，parquet 要等本次标注结束才会写入语言列。";
  const rows = [];
  staging.episodes.forEach((episode) => {
    const stored = (staging.on_disk || {})[String(episode.episode)];
    episode.rows.forEach((row) => {
      const written = stored && ((row.column === "language_persistent" && stored.persistent) || (row.column === "language_events" && stored.events));
      const dropped = keep && !keep.has(row.style);
      const status = dropped ? "本次不写入" : (written ? "已写入 parquet" : "只在暂存");
      rows.push(`<tr>
        <td>#${episode.episode}</td>
        <td>${escapeHtml(row.module)}</td>
        <td>${escapeHtml(row.style)}</td>
        <td>${row.timestamp == null ? "" : Number(row.timestamp).toFixed(2)}</td>
        <td>${escapeHtml(row.column)}</td>
        <td>${status}</td>
        <td>${escapeHtml(row.content)}</td>
      </tr>`);
    });
  });
  const extra = rows.length > 80 ? ` 共 ${rows.length} 行，下面只显示最近 80 行。` : "";
  $("changeSummary").textContent = `${chosen}${counts}。${disk}${extra}`;
  $("changeRows").innerHTML = (rows.length > 80 ? rows.slice(-80) : rows).join("");
}

async function refreshJob() {
  try {
    renderJob(await api("/api/job"));
  } catch (error) {
    $("jobMessage").textContent = error.message;
  }
}

$("open").onclick = async () => {
  try {
    renderDataset(await api("/api/open", {dataset_path: $("path").value}));
    await refreshJob();
  } catch (error) {
    $("loadMessage").textContent = error.message;
    $("loadMessage").classList.add("error");
  }
};
$("filter").oninput = () => { if (state.dataset) renderEpisodeButtons(); };
$("play").onclick = () => setPlaying(true);
$("pause").onclick = () => setPlaying(false);
$("rate").onchange = () => {
  const rate = Number($("rate").value);
  players().forEach((video) => { video.playbackRate = rate; });
};
$("seek").oninput = () => seek(Number($("seek").value) / 1000 * state.duration);
$("parquetFile").onchange = () => { state.parquetOffset = 0; loadParquet(); };
$("parquetEpisode").onchange = () => { state.parquetOffset = 0; loadParquet(); };
$("pageSize").onchange = () => { state.parquetOffset = 0; loadParquet(); };
$("parquetPrev").onclick = () => { state.parquetOffset = Math.max(0, state.parquetOffset - Number($("pageSize").value)); loadParquet(); };
$("parquetNext").onclick = () => { state.parquetOffset += Number($("pageSize").value); loadParquet(); };
document.querySelectorAll("[data-spec]").forEach((button) => {
  button.onclick = () => { $("spec").value = button.dataset.spec; };
});
document.querySelectorAll("[data-features]").forEach((button) => {
  button.onclick = () => {
    const chosen = new Set(button.dataset.features === "all"
      ? ["subtask", "plan", "memory", "task_aug", "interjections", "vqa"]
      : button.dataset.features.split(","));
    document.querySelectorAll("#featureChoices input").forEach((input) => {
      input.checked = chosen.has(input.value);
    });
    renderSelection();
  };
});
function selectedFeatures() {
  return [...document.querySelectorAll("#featureChoices input:checked")].map((input) => input.value);
}
document.querySelectorAll("#featureChoices input").forEach((input) => {
  input.onchange = () => renderSelection();
});
$("spec").oninput = () => renderSelection();
$("mode").onchange = () => {
  $("output").disabled = $("mode").value !== "copy";
};
$("output").disabled = true;
$("run").onclick = async () => {
  if (state.job && state.job.alive) {
    $("jobMessage").textContent = `任务已在跑 pid ${state.job.pid}，开始已锁定。要停的话点结束。`;
    $("jobMessage").classList.add("error");
    applyJobControls(state.job);
    return;
  }
  const features = selectedFeatures();
  if (!features.length) {
    $("jobMessage").textContent = "至少勾选一个特征";
    $("jobMessage").classList.add("error");
    return;
  }
  $("run").disabled = true;
  try {
    const job = await api("/api/annotate", {
      episodes: $("spec").value,
      mode: $("mode").value,
      output: $("output").value,
      features,
    });
    $("jobMessage").textContent = `已启动 pid ${job.pid}，情节 ${job.episodes}，特征 ${job.features}，方式 ${job.mode}。`;
    $("jobMessage").classList.remove("error");
    $("jobPause").disabled = false;
    $("jobStop").disabled = false;
    $("jobPause").classList.add("ready");
    $("jobStop").classList.add("ready");
    await refreshJob();
  } catch (error) {
    $("jobMessage").textContent = error.message;
    $("jobMessage").classList.add("error");
    await refreshJob();
  }
};
for (const [id, action, label] of [["jobPause", "pause", "已暂停"], ["jobResume", "resume", "已继续"], ["jobStop", "stop", "已结束"]]) {
  $(id).onclick = async () => {
    try {
      await api("/api/job/control", {action});
      $("jobMessage").textContent = label;
      $("jobMessage").classList.remove("error");
      await refreshJob();
    } catch (error) {
      $("jobMessage").textContent = error.message;
      $("jobMessage").classList.add("error");
    }
  };
}
$("addPersistent").onclick = () => {
  $("persistent").insertAdjacentHTML("beforeend", editorRow("persistent", {style: "subtask", role: "assistant", timestamp: localTime().toFixed(3), content: ""}));
  bindEditorRows();
};
$("addEvent").onclick = () => {
  const camera = Object.keys(state.clips)[0] || "";
  $("events").insertAdjacentHTML("beforeend", editorRow("event", {style: "interjection", role: "user", timestamp: localTime().toFixed(3), camera, content: ""}));
  bindEditorRows();
};
$("save").onclick = async () => {
  if (!state.episode) {
    $("saveState").textContent = "先选择一条 episode";
    return;
  }
  try {
    await api("/api/episode", {
      episode_index: state.episode.episode_index,
      persistent: readEditors("persistent"),
      events: readEditors("event"),
    });
    $("saveState").textContent = `episode ${state.episode.episode_index} 已写回 parquet`;
    renderDataset(await api("/api/dataset"));
    await openEpisode(state.episode.episode_index);
  } catch (error) {
    $("saveState").textContent = error.message;
  }
};

api("/api/health").then(() => {
  $("health").textContent = "已连接";
  $("health").classList.add("ok");
}).catch(() => {
  $("health").textContent = "未连接";
});
refreshJob();
setInterval(refreshJob, 2000);
setInterval(() => {
  if (state.job && (state.job.running || state.job.paused)) renderLive(state.job);
}, 1000);
