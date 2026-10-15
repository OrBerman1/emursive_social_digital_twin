/* Shuk HaTikvah 360 - shared viewer engine (used by the design variants v1..v5).
   Owns: 360 video rendering (WebGL, same maths as pipeline/spherical.py), data loading, per-frame entity list
   (tracked, blended between 5 fps analysis frames, filtered), counts, camera position on the map, view controls.
   A design owns: layout, colours, what to draw and how. It gets a callback every frame:
     engine.onFrame = (state, ctx) => { ...draw on the overlay canvas... }
   state = { fi, t, duration, walk, data, entities[], counts, cam, view }   (see buildState)
   Coordinates of rects are in overlay-canvas pixels (devicePixelRatio applied); engine.k = canvas px per CSS px. */
(function () {
  const deg = Math.PI / 180;
  // Okabe-Ito colour-blind-safe palette, one hue per category
  const PALETTE = {
    person: "#0072B2", food: "#009E73", object: "#E69F00", level2: "#CC79A7", sign: "#56B4E9",
    pictured: "#F0E442", frontage: "#D55E00", waste: "#7a5c3d", ink: "#1c2430",
  };
  const KIND = { p: "person", o: "object", l2: "level2", s: "sign", dp: "pictured", fd: "food", fm: "frontage" };
  const EMOJI = {
    person: "🧍", bicycle: "🚲", motorcycle: "🏍️", car: "🚗", truck: "🚚", bus: "🚌", chair: "🪑", bench: "🪑", umbrella: "☂️",
    handbag: "👜", backpack: "🎒", suitcase: "🧳", "potted plant": "🪴", "dining table": "🪵", banana: "🍌", apple: "🍎", orange: "🍊",
    carrot: "🥕", broccoli: "🥦", bottle: "🧴", "plastic crate": "📦", "stack of crates": "📦", "cardboard box": "📦", "flattened cardboard": "📦",
    packaging: "📦", "delivery box": "📦", "plastic bag": "🛍️", "garbage bag": "🗑️", "trash bin": "🗑️", "organic waste": "🍂",
    "fruit and vegetable waste": "🍂", "hand truck": "🛒", "trolley cart": "🛒", "wheeled shopping trolley": "🛒", walker: "🦯",
    wheelchair: "🦽", "mobility scooter": "🦼", broom: "🧹", "e-bike": "🛵", pallet: "🪵", "display table": "🪵", "plastic chair": "🪑",
    stool: "🪑", awning: "⛱️", "price sign": "🏷️", "shop sign": "🪧", flag: "🚩", "hanging lamp": "💡", text: "🪧", sign: "🪧", pictured: "🖼️",
    frontage: "🧱",
    Tomatoes: "🍅", Cucumbers: "🥒", "Bell peppers": "🫑", "Chili peppers": "🌶️", Eggplants: "🍆", Potatoes: "🥔", Onions: "🧅", Garlic: "🧄",
    Carrots: "🥕", "Cabbage & lettuce": "🥬", Herbs: "🌿", Corn: "🌽", Mushrooms: "🍄", Watermelons: "🍉", Melons: "🍈", Bananas: "🍌",
    Apples: "🍎", "Oranges & citrus": "🍊", Grapes: "🍇", Strawberries: "🍓", Mangoes: "🥭", "Peaches & plums": "🍑", Pomegranates: "🔴",
    Pears: "🍐", "Dates & dried fruit": "🌴", "Nuts & seeds": "🥜", Spices: "🧂", "Olives & pickles": "🫒", "Sweets & halva": "🍬",
    "Bread & pita": "🥖", "Pastries & cakes": "🥐", "Cheese & dairy": "🧀", Meat: "🥩", Fish: "🐟", Eggs: "🥚", "Drinks & juice": "🧃",
    "Prepared food": "🍲", "Packaged groceries": "🛒", "Mixed fruit & vegetables": "🧺", "Produce (unsure)": "🥗", "Food picture (menu/sign)": "🖼️",
  };
  const AGE_EMOJI = { "child": "🧒", "young adult 18-35": "🧑", "middle adult 36-60": "🧔", "older adult 61+": "🧓", "uncertain": "👤" };
  const AGES = ["child", "young adult 18-35", "middle adult 36-60", "older adult 61+", "uncertain"];
  const emojiOf = c => EMOJI[c] || "❔";

  function create(opts) {
    const E = {
      PALETTE, KIND, EMOJI, AGE_EMOJI, AGES, emojiOf, deg,
      yaw: 0, pitch: -5, fov: 90, conf: 0.6, minLen: 5, speed: 1, playing: false,
      walk: null, data: null, motion: null, geo: { walks: {}, shops: [], frontage: [] },
      onFrame: null, onLoad: null, onWalk: null, k: 1, hits: [], showOperator: false,
    };
    const glc = opts.gl, ov = opts.overlay, host = opts.host || glc.parentElement;
    const gl = glc.getContext("webgl", { antialias: false, preserveDrawingBuffer: false });
    const ctx = ov.getContext("2d");
    E.ctx = ctx; E.overlay = ov;
    const video = document.createElement("video");
    video.crossOrigin = "anonymous"; video.playsInline = true; video.muted = true; video.loop = false; video.preload = "auto";
    video.style.display = "none"; document.body.appendChild(video);
    E.video = video;
    const still = new Image(); let stillFi = -1, stillReady = false, stillBusy = false, vt = 0, lastNow = performance.now();
    still.onload = () => { stillReady = true; stillBusy = false; }; still.onerror = () => { stillBusy = false; };
    const videoOk = () => video.readyState >= 2;
    E.time = () => (videoOk() ? video.currentTime : vt);
    // Looping is done here, not by the browser. Chrome sometimes takes a long video as ended early: after about a
    // minute of playing from the start its stream read stopped and it jumped to the end (with loop on: to 0). We keep
    // the last good playing position; a jump that no one asked for (back, or forward past it) or an early "ended"
    // resumes there, which makes Chrome fetch that part afresh. Only the real end loops to the start.
    let goodT = 0, ownSeek = false, goodAt = performance.now();
    const setT = t => { ownSeek = true; video.currentTime = t; };
    setInterval(() => {
      const now = performance.now();
      if (video.readyState < 3) { goodAt = now; return; }                  // loading / buffering: the clock is not running
      const c = video.currentTime, expect = video.paused ? 0 : (now - goodAt) / 1000 * video.playbackRate;
      if (ownSeek) { if (!video.seeking) { ownSeek = false; goodT = c; goodAt = now; } return; }
      const jump = Math.abs(c - goodT - expect) > 3;                      // compared with the real elapsed time (timers slow down in background tabs)
      if (E.playing && jump && goodT < E.duration() - 1) { setT(goodT); video.play().catch(() => {}); return; }
      if (!jump || !E.playing) { goodT = c; goodAt = now; }
    }, 200);
    video.addEventListener("ended", () => {
      if (!E.playing) return;
      if (goodT >= E.duration() - 1.5) { setT(0); goodT = 0; } else setT(goodT);        // real end: loop; early: resume
      video.play().catch(() => {});
    });
    E.duration = () => (video.duration && isFinite(video.duration)) ? video.duration : (E.data ? E.data.n_frames / E.data.fps : 0);

    // ---- WebGL equirect renderer ----
    const vs = `attribute vec2 p; varying vec2 v; void main(){ v = p; gl_Position = vec4(p, 0., 1.); }`;
    const fs = `precision highp float; varying vec2 v; uniform sampler2D tex; uniform float yaw, pitch, tx, ty;
      const float PI = 3.141592653589793;
      void main(){ vec3 d = vec3(v.x * tx, v.y * ty, 1.0);
        float cp = cos(pitch), sp = sin(pitch), cw = cos(yaw), sw = sin(yaw);
        float y1 = d.y * cp + d.z * sp, z1 = -d.y * sp + d.z * cp;
        float x2 = d.x * cw + z1 * sw, z2 = -d.x * sw + z1 * cw;
        float lon = atan(x2, z2), lat = atan(y1, length(vec2(x2, z2)));
        gl_FragColor = texture2D(tex, vec2(lon / (2.0 * PI) + 0.5, 0.5 - lat / PI)); }`;
    const sh = (t, s) => { const o = gl.createShader(t); gl.shaderSource(o, s); gl.compileShader(o); return o; };
    const prog = gl.createProgram(); gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs));
    gl.linkProgram(prog); gl.useProgram(prog);
    gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer()); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "p"); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
    gl.bindTexture(gl.TEXTURE_2D, gl.createTexture());
    for (const [k, v] of [[gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR], [gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE]])
      gl.texParameteri(gl.TEXTURE_2D, k, v);
    const U = n => gl.getUniformLocation(prog, n), uYaw = U("yaw"), uPitch = U("pitch"), uTx = U("tx"), uTy = U("ty");

    function resize() {
      const r = host.getBoundingClientRect(), dpr = Math.min(2, devicePixelRatio || 1);
      for (const c of [glc, ov]) { c.width = Math.max(2, Math.round(r.width * dpr)); c.height = Math.max(2, Math.round(r.height * dpr)); }
      gl.viewport(0, 0, glc.width, glc.height);
      E.k = ov.width / Math.max(1, host.clientWidth);
    }
    new ResizeObserver(resize).observe(host); resize();

    // ---- projection (yaw, pitch in degrees) -> overlay pixels; null behind the camera ----
    function project(yw, pt) {
      const W = ov.width, H = ov.height, tx = Math.tan(E.fov * deg / 2), ty = tx * H / W;
      const a = yw * deg, b = pt * deg;
      const x = Math.cos(b) * Math.sin(a), y = Math.sin(b), z = Math.cos(b) * Math.cos(a);
      const w = E.yaw * deg, p = E.pitch * deg;
      const x1 = x * Math.cos(w) - z * Math.sin(w), z1 = x * Math.sin(w) + z * Math.cos(w);
      const y0 = y * Math.cos(p) - z1 * Math.sin(p), z0 = y * Math.sin(p) + z1 * Math.cos(p);
      if (z0 <= 0.05) return null;
      return [(x1 / z0 / tx + 1) * W / 2, (1 - y0 / z0 / ty) * H / 2];
    }
    E.project = project;
    const inFov = (yw, margin = 0) => Math.abs(((yw - E.yaw + 540) % 360) - 180) < E.fov / 2 + margin;
    E.inFov = inFov;
    function boxRect(yw, yl, yr, pt, pb) {
      const span = ((yr - yl) % 360 + 360) % 360, pts = [];
      for (let k = 0; k <= 4; k++) { const yy = yl + span * k / 4; pts.push(project(yy, pt), project(yy, pb)); }
      const ok = pts.filter(Boolean); if (ok.length < 6) return null;
      const xs = ok.map(p => p[0]), ys = ok.map(p => p[1]);
      return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
    }
    E.boxRect = boxRect;

    // ---- project (?project=<id>; projects.html / pipeline/projects_api.py): data paths, walks and their labels ----
    const J = async u => { const r = await fetch(u); if (!r.ok) throw new Error(u); return r.json(); };
    const urlProject = new URLSearchParams(location.search).get("project");     // none -> the server's default (config/viewer.json)
    E.project = { id: urlProject || "shuk_hatikvah", name: "Shuk HaTikvah Legacy",
                  processed: "/data/processed", interim: "/data/interim",
                  walks: { walk1: { label: "Hatikvah St" }, walk2: { label: "Hamevasser St" } } };      // fallback: the original project
    async function loadProject() {
      try {
        const pj = await J("/api/projects"), ps = pj.projects || [];
        const pr = ps.find(x => x.id === (urlProject || pj.default || E.project.id)) || ps[0];
        if (pr) Object.assign(E.project, { id: pr.id, name: pr.name, processed: pr.urls.processed, interim: pr.urls.interim,
          walks: Object.fromEntries(Object.entries(pr.walks).filter(([, v]) => v.analysed).map(([w, v]) => [w, { label: v.label }])) });
      } catch (e) { /* old server without the projects API: original project */ }
    }
    E.walkIds = () => Object.keys(E.project.walks);
    async function loadStatic() {
      await loadProject();
      const P = E.project.processed;
      try { E.geo.shops = (await J(`${P}/shops.json`)).shops || []; } catch (e) {}
      for (const w of E.walkIds()) { try { E.geo.walks[w] = await J(`${P}/${w}/georef.json`); } catch (e) {} }
      try { E.geo.frontage = (await J(`${P}/frontage_materials.json`)).points_list || []; } catch (e) {}
    }
    E.ready = loadStatic();
    E.load = async function (walk) {
      await E.ready;
      const P = E.project.processed;
      const keepT = E.walk === walk ? E.time() : 0;
      E.walk = walk; E.data = null; E.motion = null; E.seen = new Set(); totKey = "";
      let d;
      try { d = await J(`${P}/${walk}/overlays_stab.json`); } catch (e) { d = await J(`${P}/${walk}/overlays.json`); }
      E.data = d;
      try { E.motion = await J(`${P}/${walk}/motion_stab.json`); } catch (e) { E.motion = null; }
      video.src = `${P}/${walk}/${d.video || "video/equirect.mp4"}`; video.load();
      vt = keepT; stillFi = -1; stillBusy = false;
      goodT = vt;
      video.addEventListener("loadedmetadata", () => { if (vt) setT(vt); video.playbackRate = E.speed; }, { once: true });
      if (E.onWalk) E.onWalk(walk);
      return d;
    };

    // ---- playback ----
    E.play = (on = !E.playing) => { E.playing = on; if (on) video.play().catch(() => {}); else video.pause(); return on; };
    E.seek = t => { vt = Math.max(0, Math.min(E.duration(), t)); goodT = vt; if (videoOk()) setT(vt); };
    E.seekFrame = f => E.seek(f / (E.data ? E.data.fps : 5));
    E.setSpeed = s => { E.speed = s; video.playbackRate = s; };
    E.look = (yaw, pitch) => { E.yaw = ((yaw + 540) % 360) - 180; if (pitch != null) E.pitch = Math.max(-85, Math.min(85, pitch)); };

    // ---- motion helpers ----
    const gidOf = id => (E.motion && E.motion.id2g && E.motion.id2g[id]) ?? id;
    E.gidOf = gidOf;
    E.motionTrack = id => E.motion && E.motion.tracks[gidOf(id)];
    E.motionAt = (id, fi) => {
      const m = E.motionTrack(id); if (!m) return null;
      const i = fi - m.f0; if (i < 0 || i >= m.lat.length || m.lat[i] == null) return null;
      return { lat: m.lat[i], lon: m.lon[i], spd: m.spd[i], brg: m.brg[i], ok: m.ok ? m.ok[i] === 1 : m.spd[i] >= 0.5 };
    };
    // screen-space ground arrow from the feet in the walking direction (null if unreliable / standing)
    E.groundArrow = (e, fi) => {
      const m = E.motionAt(e.id, fi), g = E.geo.walks[E.walk], h = (E.data && E.data.camera_height_m) || 1.3;
      if (!m || !m.ok || !g || e.pb > -1) return null;
      const D = h / Math.tan(-e.pb * deg), a = (m.brg - g.yaw0_bearing_deg[fi]) * deg, L = Math.max(0.8, Math.min(3, m.spd * 1.6));
      const Px = D * Math.sin(e.yaw * deg), Py = D * Math.cos(e.yaw * deg), Qx = Px + L * Math.sin(a), Qy = Py + L * Math.cos(a);
      const s0 = project(e.yaw, e.pb), s1 = project(Math.atan2(Qx, Qy) / deg, -Math.atan(h / Math.hypot(Qx, Qy)) / deg);
      if (!s0 || !s1) return null;
      return { dx: s1[0] - s0[0], dy: s1[1] - s0[1], spd: m.spd };
    };

    // person id of a track (link_persons.py, tracks_fields "person"): a person split into several tracks counts once
    E.personOf = (d, id) => (d.tracks[id] || [])[7] ?? id;
    // distance limit (Layers & filters "Show up to ... m"): ground distance of people (feet) and of objects standing on
    // the ground (bottom of the box + camera height). Signs, food displays and pictured people sit above the ground, so
    // one camera cannot measure their distance: they are not limited. Unknown (bottom at/above the horizon) = far.
    E.maxDist = Infinity;
    const groundD = (pb, d) => pb < -1 ? (d.camera_height_m || 1.4) / Math.tan(-pb * Math.PI / 180) : Infinity;
    E.distOf = (key, e, d) => key === "p" || key === "l2" ? (e[7] ?? groundD(e[6], d)) : key === "o" || key === "w" ? groundD(e[6], d) : null;
    E.near = (key, e, d) => { if (!isFinite(E.maxDist)) return true; const x = E.distOf(key, e, d); return x == null || x <= E.maxDist; };
    // ---- totals over the whole walk (unique persons whose tracks pass the same filters) ----
    let totKey = "", totVal = null;
    function totals(d) {
      const key = `${E.walk}|${E.conf}|${E.minLen}|${E.maxDist}`;
      if (key === totKey) return totVal;
      const best = { p: new Map(), o: new Map(), l2: new Map(), fd: new Map() };
      for (const F of d.frames) {
        for (const e of F.p) if (E.near("p", e, d)) best.p.set(e[0], Math.max(best.p.get(e[0]) || 0, e[1]));
        for (const e of F.o) if (E.near("o", e, d)) best.o.set(e[7], e[0]);
        for (const e of F.l2) if (E.near("l2", e, d)) best.l2.set(e[8], e[0]);
        for (const e of (F.fd || [])) best.fd.set(e[7], e[0]);
      }
      const ps = new Set(); for (const [id, c] of best.p) if (c >= E.conf && ((d.tracks[id] || [])[2] || 0) >= E.minLen) ps.add(E.personOf(d, id));
      const people = ps.size;
      const food = [...best.fd.values()].filter(c => c !== "Food picture (menu/sign)").length;
      totVal = { people, objects: best.o.size, level2: best.l2.size, food, signs: Object.values(d.signs || {}).filter(x => x[3]).length,
                 pictured: Object.keys(d.dp_tracks || {}).length };
      totKey = key; return totVal;
    }

    // ---- per-frame state ----
    function buildState() {
      const d = E.data; if (!d) return null;
      const fr = Math.max(0, E.time() * d.fps), f0 = Math.min(d.n_frames - 1, Math.floor(fr)), f1 = Math.min(d.n_frames - 1, f0 + 1), a = fr - f0;
      const fi = Math.min(d.n_frames - 1, Math.round(fr));
      const EMP = { p: [], o: [], l2: [], s: [], dp: [], fd: [] }, A = d.frames[f0] || EMP, B = d.frames[f1] || EMP;
      const lerpA = (x, y) => x + (((y - x + 540) % 360) - 180) * a;
      const lens = {
        p: id => (d.tracks[id] || [])[2] || 0, o: id => (d.obj_len || {})[id] || 0, l2: id => (d.obj_len || {})[id] || 0,
        s: id => (d.sign_len || {})[id] ?? 99, dp: id => ((d.dp_tracks || {})[id] || [])[2] || 99, fd: id => (d.food_len || {})[id] || 0,
      };
      const idIdx = { p: 0, o: 7, l2: 8, s: 0, dp: 0, fd: 7 };
      const out = [], counts = { people: { view: 0, frame: 0 }, byKind: {}, byClass: {}, age: {}, sex: {}, signsRead: [] };
      for (const key of ["p", "o", "l2", "s", "dp", "fd"]) {
        const la = A[key] || [], lb = B[key] || [], ii = idIdx[key], len = lens[key];
        const mb = new Map(lb.map(e => [e[ii], e])), ma = new Map(la.map(e => [e[ii], e])), pairs = [];
        for (const e of la) {
          if (len(e[ii]) < E.minLen) continue;
          const n = mb.get(e[ii]);
          if (n) { const m = e.slice(); for (const j of [2, 3, 4]) m[j] = lerpA(e[j], n[j]); for (const j of [5, 6]) m[j] = e[j] + (n[j] - e[j]) * a; pairs.push([m, 1]); }
          else pairs.push([e, 1 - a]);
        }
        for (const e of lb) if (!ma.has(e[ii]) && len(e[ii]) >= E.minLen) pairs.push([e, a]);
        for (const [e, al] of pairs) {
          const kind = KIND[key], conf = key === "s" ? 1 : e[1];
          const thr = key === "l2" ? Math.max(0.15, E.conf - 0.2) : key === "p" ? E.conf : key === "o" ? E.conf : 0;
          if (conf < thr || !E.near(key, e, d)) continue;
          let cls, id = e[ii], extra = {};
          if (key === "p") { cls = "person"; const t = d.tracks[id] || []; extra = { age: t[6] || "uncertain", sex: t[5] || "uncertain", dist: e[7] }; }
          else if (key === "s") { const info = (d.signs || {})[id]; if (!info) continue; cls = "sign"; extra = { text: info[0], lang: info[1], readable: !!info[3] }; }
          else if (key === "dp") cls = "pictured";
          else cls = e[0];
          if (key === "p" && al >= 0.5) { counts.people.frame++; E.seen.add(E.personOf(d, id)); }
          if (!inFov(e[2])) continue;
          const rect = boxRect(e[2], e[3], e[4], e[5], e[6]); if (!rect) continue;
          const ent = { key, kind, id, cls, conf, alpha: al, yaw: e[2], pt: e[5], pb: e[6], rect, ...extra };
          out.push(ent);
          if (al >= 0.5) {
            counts.byKind[kind] = (counts.byKind[kind] || 0) + 1;
            if (key !== "p" && key !== "s" && key !== "dp") { const ck = `${kind}:${cls}`; counts.byClass[ck] = (counts.byClass[ck] || 0) + 1; }
            if (key === "p") { counts.people.view++; counts.age[ent.age] = (counts.age[ent.age] || 0) + 1; counts.sex[ent.sex] = (counts.sex[ent.sex] || 0) + 1; }
            if (key === "s" && ent.readable) counts.signsRead.push(ent);
          }
        }
      }
      // frontage materials (location-based): visible within +-3 s of the spot
      const g = E.geo.walks[E.walk];
      if (g && g.lat[fi] != null) {
        const h = d.camera_height_m || 1.3;
        for (const p of E.geo.frontage) {
          if (p.walk !== E.walk || Math.abs(p.frame - fi) > 15) continue;
          const dx = (p.lon - g.lon[fi]) * 111320 * Math.cos(g.lat[fi] * deg), dy = (p.lat - g.lat[fi]) * 110574, dist = Math.hypot(dx, dy);
          const yw = ((Math.atan2(dx, dy) / deg - g.yaw0_bearing_deg[fi]) + 540) % 360 - 180, pt = Math.atan2(1.6 - h, Math.max(dist, 1)) / deg;
          if (!inFov(yw)) continue;
          const q = project(yw, pt); if (!q) continue;
          out.push({ key: "fm", kind: "frontage", id: p.id, cls: "frontage", conf: 1, alpha: 1, yaw: yw, pt, pb: pt, rect: [q[0] - 1, q[1] - 1, q[0] + 1, q[1] + 1], point: true, info: p });
          counts.byKind.frontage = (counts.byKind.frontage || 0) + 1;
        }
      }
      // camera on the map
      let cam = null;
      if (g && g.lat[fi] != null) cam = { lat: g.lat[fi], lon: g.lon[fi], bearing: (g.yaw0_bearing_deg[fi] + E.yaw + 360) % 360, yaw0: g.yaw0_bearing_deg[fi] };
      return { fi, t: E.time(), duration: E.duration(), walk: E.walk, data: d, entities: out, counts, totals: totals(d), uniqueSoFar: E.seen.size, cam };
    }
    E.state = null;

    // ---- render loop ----
    function frame() {
      requestAnimationFrame(frame);
      const now = performance.now(), dt = (now - lastNow) / 1000; lastNow = now;
      const d = E.data;
      if (videoOk()) { gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, video); vt = video.currentTime; }
      else if (d) {
        if (E.playing) { vt += dt * E.speed; if (vt >= E.duration()) vt = 0; }
        const want = Math.min(d.n_frames - 1, Math.round(vt * d.fps));
        if (want !== stillFi && !stillBusy) { stillFi = want; stillReady = false; stillBusy = true; still.src = `${E.project.interim}/${E.walk}/${d.frames_dir || "frames"}/${String(want).padStart(6, "0")}.jpg`; }
        if (stillReady) { gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, still); stillReady = false; }
      }
      const W = ov.width, H = ov.height, tx = Math.tan(E.fov * deg / 2);
      gl.uniform1f(uYaw, E.yaw * deg); gl.uniform1f(uPitch, E.pitch * deg); gl.uniform1f(uTx, tx); gl.uniform1f(uTy, tx * H / W);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      ctx.clearRect(0, 0, W, H); E.hits = [];
      const st = buildState(); E.state = st;
      if (st && E.onFrame) E.onFrame(st, ctx);
    }
    requestAnimationFrame(frame);

    // ---- interaction: drag to look, wheel to zoom, arrows/space; click -> engine.onClick(hit or null, ev) ----
    let drag = null;
    E.addHit = (x0, y0, x1, y1, ent) => E.hits.push({ r: [x0, y0, x1, y1], ent });
    E.hitAt = (cx, cy) => { const r = host.getBoundingClientRect(), x = (cx - r.left) * E.k, y = (cy - r.top) * E.k;
      return (E.hits.slice().reverse().find(h => x >= h.r[0] && x <= h.r[2] && y >= h.r[1] && y <= h.r[3]) || {}).ent || null; };
    host.addEventListener("pointerdown", e => {
      if (e.target.closest("[data-no-drag]")) return;
      drag = { x: e.clientX, y: e.clientY, yaw: E.yaw, pitch: E.pitch, moved: false }; host.setPointerCapture(e.pointerId); host.classList.add("dragging");
    });
    host.addEventListener("pointermove", e => {
      if (!drag) { if (E.onHover) E.onHover(E.hitAt(e.clientX, e.clientY), e); return; }
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 4) drag.moved = true;
      const s = E.fov / host.clientWidth; E.look(drag.yaw - dx * s, drag.pitch + dy * s);
    });
    host.addEventListener("pointerup", e => {
      host.classList.remove("dragging");
      if (drag && !drag.moved && E.onClick) E.onClick(E.hitAt(e.clientX, e.clientY), e);
      drag = null;
    });
    host.addEventListener("wheel", e => { if (e.target.closest("[data-no-drag]")) return; e.preventDefault(); E.fov = Math.max(40, Math.min(120, E.fov + e.deltaY * 0.05)); }, { passive: false });
    host.tabIndex = 0;
    host.addEventListener("keydown", e => {
      if (e.key === "ArrowLeft") E.look(E.yaw - 6); else if (e.key === "ArrowRight") E.look(E.yaw + 6);
      else if (e.key === "ArrowUp") E.look(E.yaw, E.pitch + 4); else if (e.key === "ArrowDown") E.look(E.yaw, E.pitch - 4);
      else if (e.key === " ") { E.play(); if (E.onPlay) E.onPlay(E.playing); } else return;
      e.preventDefault();
    });

    // ---- small drawing helpers designs may use ----
    E.draw = {
      roundRect(c, x, y, w, h, r) { c.beginPath(); c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r); c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath(); },
      emoji(c, e, x, y, size) { c.font = `${size}px "Segoe UI Emoji","Apple Color Emoji","Noto Color Emoji",sans-serif`; c.textBaseline = "middle"; c.textAlign = "center"; c.fillText(e, x, y); c.textAlign = "start"; c.textBaseline = "alphabetic"; },
      arrow(c, x, y, dx, dy, col, k) {
        const n = Math.hypot(dx, dy); if (n < 3) return; const ux = dx / n, uy = dy / n, hl = Math.min(12 * k, n * .45), ex = x + dx, ey = y + dy;
        for (const [w, s] of [[5 * k, "rgba(0,0,0,.35)"], [2.6 * k, col]]) { c.lineWidth = w; c.strokeStyle = s; c.lineCap = "round"; c.beginPath(); c.moveTo(x, y); c.lineTo(ex - ux * hl * .6, ey - uy * hl * .6); c.stroke(); }
        c.fillStyle = col; c.beginPath(); c.moveTo(ex, ey); c.lineTo(ex - ux * hl - uy * hl * .55, ey - uy * hl + ux * hl * .55); c.lineTo(ex - ux * hl + uy * hl * .55, ey - uy * hl - ux * hl * .55); c.closePath(); c.fill();
      },
    };
    E.fmtTime = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
    E.walkLabel = w => (E.project.walks[w] || {}).label || w;
    E.walkShort = w => `walk ${String(w).replace(/^walk/, "")}`;
    return E;
  }
  window.ShukEngine = { create, PALETTE, EMOJI, AGE_EMOJI, AGES, emojiOf: c => EMOJI[c] || "❔" };
})();
