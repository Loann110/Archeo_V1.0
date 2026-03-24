// app.js — WebSocket JPEG + contrôles + switch moteur (Open3D / MeshLab) + réglages APSS temps réel

class ArcheoViewer {
  constructor() {
    this.dom = {};
    this.state = {
      objColor: '#b0a9a2',
      bgColor: '#0b0028',
      engine: 'open3d', // 'open3d' | 'meshlab'
    };

    this.dragging = false;
    this.lastX = 0; this.lastY = 0; this.ctrlKey = false;

    this._serverObjToken = null;
    this._lastSend = 0; this._minDeltaMs = 12; // ~80 Hz max
    this._resizeTimer = null;

    // synchro inversion backend
    this.invertZoom = false;

    // timer pour press & hold sur les boutons zoom
    this._zoomHoldTimer = null;

    // debouncer partagé pour MeshLab
    this._debouncedSendMLParams = null;

    // progression APSS (MeshLab)
    this._mlProgressTimer = null;
    this._mlStatusTimer   = null;
    this._mlBaselineSize  = null;
    this._mlPollingBusy   = false;

    // WebSocket streaming
    this._ws               = null;
    this._wsUrl            = null;
    this._wsReconnectTimer = null;
    this._lastFrameUrl     = null;

    // ===== aspect réel du flux + stage "fit" =====
    this._streamAspect = 16 / 9; // mis à jour via naturalWidth/naturalHeight
    this._stageCssW = 0;
    this._stageCssH = 0;
  }

  async init() {
    // Gestion du splash une seule fois par session
    const splash = document.getElementById("splash");

    // Si le splash n'a jamais été montré dans cet onglet
    if (!sessionStorage.getItem("splashShown")) {
      splash.classList.remove("hidden"); // on l'affiche
      sessionStorage.setItem("splashShown", "yes");
    } else {
      splash.classList.add("hidden"); // on ne l'affiche plus
    }

    this.#queryDOM();
    this.#initMLProgressBar(); // barre de progression Apply

    // L'image de stream ne doit pas intercepter les events (wheel/drag)
    if (this.dom.stream) this.dom.stream.style.pointerEvents = 'none';

    await this.#syncBackendFlags(); // récupère invert_zoom + moteur actif
    this.#bindUI();
    this.#bindDnD();
    this.#createZoomButtons(); // boutons Zoom (icônes)

    // FIX: assure que le stage est déjà "fit" avant 1er resize backend
    this.#fitStageToLeft();
    this.#syncSize(); // envoie la taille initiale

    window.addEventListener('resize', ()=> this.#debouncedResize());

    // Si on démarre directement en MeshLab, pousse une 1ère fois les paramètres
    if ((this.state.engine || 'open3d') === 'meshlab') {
      this.#sendMLParams(this.#gatherMLPayload()).catch(()=>{});
    }

    // Splash screen auto-hide
    setTimeout(() => {
      const s = document.getElementById("splash");
      if (s) {
        s.style.display = "none"; // sécurité après fondu
      }
    }, 6200); // 5s + fade

    window.__archeo = this;
  }

  #queryDOM() {
    // zone viewer
    this.dom.left       = document.querySelector('#left');
    this.dom.container  = document.querySelector('#container3D');
    this.dom.stream     = document.querySelector('#stream');
    this.dom.drop       = document.querySelector('#drop');

    // toolbar
    this.dom.file       = document.querySelector('#fileObj');
    this.dom.reset      = document.querySelector('#reset');
    this.dom.remove     = document.querySelector('#removeObj');
    this.dom.capture    = document.querySelector('#capture');

    // apparence
    this.dom.objColor      = document.querySelector('#objColor');
    this.dom.resetObjColor = document.querySelector('#resetObjColor');
    this.dom.bgColor       = document.querySelector('#bgColor');
    this.dom.resetBgColor  = document.querySelector('#resetBgColor');

    // switch moteur
    this.dom.btnEngineO3D = document.querySelector('#btnEngineO3D');
    this.dom.btnEngineML  = document.querySelector('#btnEngineML');
    this.dom.mlPanel      = document.querySelector('#mlPanel');

    // réglages MeshLab (APSS)
    this.dom.mlFilter    = document.querySelector('#ml_filterscale');
    this.dom.mlSpherical = document.querySelector('#ml_spherical');
    this.dom.mlCType     = document.querySelector('#ml_ctype');
    this.dom.mlRotate    = document.querySelector('#ml_rotate');
    this.dom.mlApply     = document.querySelector('#ml_apply');
  }

  #initMLProgressBar() {
    const panel = this.dom.mlPanel;
    if (!panel) return;

    const wrap = document.createElement('div');
    wrap.id = 'mlProgressWrap';
    wrap.style.position = 'relative';
    wrap.style.marginTop = '6px';
    wrap.style.height = '3px';
    wrap.style.borderRadius = '999px';
    wrap.style.overflow = 'hidden';
    wrap.style.background = 'rgba(255,255,255,0.08)';
    wrap.style.opacity = '0';
    wrap.style.transition = 'opacity 0.18s ease';

    const bar = document.createElement('div');
    bar.id = 'mlProgressBar';
    bar.style.width = '0%';
    bar.style.height = '100%';
    bar.style.borderRadius = 'inherit';
    bar.style.background = 'linear-gradient(90deg, rgba(99,102,241,0.1), rgba(129,140,248,0.95), rgba(139,92,246,0.95))';
    bar.style.boxShadow = '0 0 12px rgba(129,140,248,0.8)';
    bar.style.transform = 'translateX(-40%)';

    wrap.appendChild(bar);
    panel.appendChild(wrap);

    this.dom.mlProgressWrap = wrap;
    this.dom.mlProgressBar  = bar;
  }

  async #syncBackendFlags() {
    try {
      const r = await fetch('/health', { cache: 'no-store' });
      if (!r.ok) return;
      const j = await r.json();
      this.invertZoom = !!j.invert_zoom;
      const active = j.active_engine || 'open3d';
      this.state.engine = active;
      this.#setEngineButtons(active);
      // rafraîchir le flux (au cas où le moteur ait changé côté serveur)
      this.#refreshStream();
    } catch (_) {}
  }

  #postJSON(url, data) {
    return fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data || {}) });
  }

  // ===== calcule la zone dispo + ajuste #container3D sur le ratio réel du flux =====
  #getLeftAvailableBox() {
    const left = this.dom.left;
    if (!left) return { w: 0, h: 0 };
    // #left est overflow hidden + grid center => on peut utiliser clientWidth/Height
    const w = Math.max(0, left.clientWidth);
    const h = Math.max(0, left.clientHeight);
    return { w, h };
  }

  #fitStageToLeft() {
    const c = this.dom.container;
    if (!c) return null;

    const { w: availW, h: availH } = this.#getLeftAvailableBox();
    if (availW < 50 || availH < 50) return null;

    const aspect = (this._streamAspect && isFinite(this._streamAspect) && this._streamAspect > 0)
      ? this._streamAspect
      : (16 / 9);

    // Fit "contain" dans #left en gardant le ratio du flux
    let w = availW;
    let h = w / aspect;
    if (h > availH) {
      h = availH;
      w = h * aspect;
    }

    // Clamps
    w = Math.max(320, Math.floor(w));
    h = Math.max(240, Math.floor(h));

    if (Math.abs(w - this._stageCssW) >= 1 || Math.abs(h - this._stageCssH) >= 1) {
      c.style.width  = `${w}px`;
      c.style.height = `${h}px`;
      this._stageCssW = w;
      this._stageCssH = h;
    }

    return { w, h, aspect };
  }

  // ---------- taille réellement affichée -> backend ----------
  async #syncSize() {
    // 1) on recale d'abord le stage CSS sur le ratio réel du flux
    const fitted = this.#fitStageToLeft();
    if (!fitted) return;

    // 2) puis on calcule la taille backend en pixels (DPR)
    const dpr = Math.min(2, window.devicePixelRatio || 1);

    let w = Math.max(320, Math.round(fitted.w * dpr));
    let h = Math.max(240, Math.round(fitted.h * dpr));

    // 3) force un aspect EXACT (évite les dérives d'arrondi)
    const aspect = fitted.aspect || (16 / 9);
    const hFromW = Math.max(240, Math.round(w / aspect));
    const wFromH = Math.max(320, Math.round(h * aspect));

    if (Math.abs(hFromW - h) <= Math.abs(wFromH - w)) {
      h = hFromW;
    } else {
      w = wFromH;
    }

    await this.#postJSON('/resize', { width:w, height:h });
    this.#refreshStream();
  }

  #debouncedResize() {
    clearTimeout(this._resizeTimer);
    this._resizeTimer = setTimeout(()=> this.#syncSize(), 180);
  }

  // --- Gestion du stream : WebSocket + fallback MJPEG ---
  #refreshStream() {
    this.#startWebSocketStream();
  }

  #startWebSocketStream() {
    const img = this.dom.stream;
    if (!img) return;

    // on vide l'éventuel src MJPEG pour ne pas faire 2 streams en parallèle
    if (img.tagName === 'IMG') {
      img.src = '';
    }

    const proto = (location.protocol === 'https:' ? 'wss://' : 'ws://');
    const url   = proto + location.host + '/ws_stream';

    // si déjà connecté au bon endpoint -> rien à faire
    if (this._ws && this._ws.readyState === WebSocket.OPEN && this._wsUrl === url) {
      return;
    }

    // ferme proprement l'ancien WS
    if (this._ws) {
      try {
        this._ws.onopen = this._ws.onclose = this._ws.onerror = this._ws.onmessage = null;
        this._ws.close();
      } catch (_) {}
      this._ws = null;
    }

    this._wsUrl = url;

    let ws;
    try {
      ws = new WebSocket(url);
    } catch (e) {
      console.error('[WS] création échouée, fallback MJPEG', e);
      if (img.tagName === 'IMG') {
        img.src = `/stream.mjpg?ts=${Date.now()}`;
      }
      return;
    }

    ws.binaryType = 'blob';
    this._ws = ws;

    ws.onopen = () => {
      // OK
    };

    ws.onmessage = (ev) => {
      const data = ev.data;
      if (data instanceof Blob) {
        this.#updateStreamImage(data);
      } else if (data instanceof ArrayBuffer) {
        const b = new Blob([data], { type:'image/jpeg' });
        this.#updateStreamImage(b);
      } else {
        // autre type inattendu -> ignore
      }
    };

    ws.onerror = (ev) => {
      console.error('[WS] erreur', ev);
    };

    ws.onclose = () => {
      this._ws = null;
      // tentative de reconnexion simple
      if (this._wsReconnectTimer) return;
      this._wsReconnectTimer = setTimeout(() => {
        this._wsReconnectTimer = null;
        this.#startWebSocketStream();
      }, 1000);
    };
  }

  #updateStreamImage(blob) {
    const img = this.dom.stream;
    if (!img) return;

    const oldUrl = this._lastFrameUrl;
    const url    = URL.createObjectURL(blob);
    this._lastFrameUrl = url;

    img.onload = () => {
      // ===== on met à jour l'aspect réel du flux dès qu'on a une frame =====
      const nw = img.naturalWidth || 0;
      const nh = img.naturalHeight || 0;
      if (nw > 10 && nh > 10) {
        const a = nw / nh;
        if (isFinite(a) && a > 0.1 && Math.abs(a - this._streamAspect) > 0.001) {
          this._streamAspect = a;
          // recale stage + resize backend 
          this.#debouncedResize();
        }
      }

      if (oldUrl) {
        try { URL.revokeObjectURL(oldUrl); } catch (_) {}
      }
    };
    img.src = url;
  }

  // ---------------- UI ----------------
  #bindUI() {
    const panel = this.dom.container;

    // Toggle inversion persistante (touche 'i') -> synchro backend
    window.addEventListener('keydown', (e) => {
      const key = (e.key || '').toLowerCase();

      if (key === 'i') {
        e.preventDefault();
        this.invertZoom = !this.invertZoom;
        this.#postJSON('/settings', { invert_zoom: this.invertZoom });
      }

      // Raccourcis clavier zoom (sémantique : ignore invert)
      if (e.key === '+' || e.key === '=' ) { e.preventDefault(); this.#sendZoomSem(-1); } // IN
      if (e.key === '-' ) { e.preventDefault(); this.#sendZoomSem(+1); } // OUT
    });

    // drag pour rotate/pan
    panel?.addEventListener('mousedown', (e) => {
      this.dragging = true;
      this.lastX = e.clientX; this.lastY = e.clientY;
      this.ctrlKey = e.ctrlKey;
      e.preventDefault();
    }, {passive:false});

    window.addEventListener('mouseup', () => this.dragging = false);

    panel?.addEventListener('mousemove', (e) => {
      if (!this.dragging) return;
      const now = performance.now();
      if (now - this._lastSend < this._minDeltaMs) return;
      const dx = e.clientX - this.lastX, dy = e.clientY - this.lastY;
      this.lastX = e.clientX; this.lastY = e.clientY;
      const action = this.ctrlKey ? 'pan' : 'rotate';
      this._lastSend = now;
      this.#postJSON('/control', { action, dx, dy });
    });

    // --------- Wheel ---------
    const getWheelDir = (e) => {
      let src = 0;
      if (typeof e.wheelDelta === 'number' && e.wheelDelta !== 0) {
        src = -e.wheelDelta;                  // DOWN => src>0, UP => src<0
      } else if (typeof e.deltaY === 'number' && e.deltaY !== 0) {
        src = e.deltaY;                       // DOWN => src>0, UP => src<0
      } else if (typeof e.detail === 'number' && e.detail !== 0) {
        src = e.detail;                       // DOWN => src>0, UP => src<0
      } else {
        src = 1;                              // défaut: OUT
      }
      if (e.shiftKey) src = -src;             // inversion temporaire
      return src > 0 ? +1 : -1;               // +1 = OUT, -1 = IN
    };

    const handleWheel = (e) => {
      e.preventDefault();
      e.stopPropagation();
      const dir = getWheelDir(e);
      this.#sendZoomDir(dir);
    };

    panel?.addEventListener('wheel', handleWheel, { passive:false });
    panel?.addEventListener('contextmenu', (e)=> e.preventDefault());

    // Couleur objet
    this.dom.objColor?.addEventListener('input', () => {
      this.state.objColor = this.dom.objColor.value;
      this.#postJSON('/settings', { obj_color: this.state.objColor });
    });
    this.dom.resetObjColor?.addEventListener('click', () => {
      this.state.objColor = '#b0a9a2';
      if (this.dom.objColor) this.dom.objColor.value = this.state.objColor;
      this.#postJSON('/settings', { obj_color: this.state.objColor });
    });

    // Couleur fond (UI + backend)
    this.dom.bgColor?.addEventListener('input', () => {
      this.state.bgColor = this.dom.bgColor.value;
      const left = document.querySelector('#left');
      if (left) left.style.background = this.state.bgColor;
      const t = document.querySelector('.torchglow'); if (t) t.style.display = 'none';
      this.#postJSON('/settings', { bg: this.state.bgColor });
    });
    this.dom.resetBgColor?.addEventListener('click', () => {
      this.state.bgColor = '#0b0028';
      if (this.dom.bgColor) this.dom.bgColor.value = this.state.bgColor;
      const left = document.querySelector('#left');
      if (left) left.style.background = this.state.bgColor;
      const t = document.querySelector('.torchglow'); if (t) t.style.display = '';
      this.#postJSON('/settings', { bg: this.state.bgColor });
    });

    // Réinit & supprimer
    this.dom.reset?.addEventListener('click', () => this.#postJSON('/reset', {}));
    this.dom.remove?.addEventListener('click', () => {
      this._serverObjToken = null;
      this.#postJSON('/remove', {});
    });

    // ===== Capture serveur (IMPORTANT: utilise la taille FIT réelle) =====
    this.dom.capture?.addEventListener('click', async () => {
      try {
        const fitted = this.#fitStageToLeft();
        if (!fitted) return alert("Taille viewer indisponible");

        const dpr = Math.min(2, window.devicePixelRatio || 1);
        const view_w = Math.max(320, Math.round(fitted.w * dpr));
        const view_h = Math.max(240, Math.round(fitted.h * dpr));

        const r = await this.#postJSON('/capture_png', {
          view_w, view_h,
          scale: 3.0,
          ssaa: 2,
          max_side: 2560
        });

        const j = await r.json().catch(()=>null);
        if (!r.ok || !j || j.ok === false) return alert('Capture échouée');

        const url = j.view_url || j.url;
        if (url) window.location.href = url;
        else alert('Capture réussie mais aucune URL de prévisualisation.');
      } catch (e) {
        console.error(e);
        alert('Capture échouée');
      }
    });

    // Upload .obj
    this.dom.file?.addEventListener('change', async (e) => {
      const f = e.target.files?.[0];
      if (!f) return;
      if (!f.name.toLowerCase().endsWith('.obj')) {
        alert('Choisis un .obj');
        this.dom.file.value = '';
        return;
      }
      const text = await f.text();
      await this.#uploadOBJToServer(f.name, text);
    });

    // ---- Switch moteur ----
    this.dom.btnEngineO3D?.addEventListener('click', ()=> this.#switchEngine('open3d'));
    this.dom.btnEngineML ?.addEventListener('click', ()=> this.#switchEngine('meshlab'));

    // ---- Réglages MeshLab : Apply avec barre de progression ----
    if (this.dom.mlApply) {
      this.dom.mlApply.addEventListener('click', () => {
        this.#startMLApplyWithProgress().catch(()=>{});
      });
    }

    // état initial boutons selon /health
    this.#setEngineButtons(this.state.engine || 'open3d');
  }

  // -- helpers MeshLab --
  #gatherMLPayload() {
    const fs = parseFloat(this.dom.mlFilter?.value || '2.0');
    const sp = parseFloat(this.dom.mlSpherical?.value || '1.0');
    const ct = (this.dom.mlCType?.value || 'mean');
    const ro = !!this.dom.mlRotate?.checked;
    return {
      filterscale: Number.isFinite(fs) ? fs : 2.0,
      sphericalparameter: Number.isFinite(sp) ? sp : 1.0,
      curvaturetype: ct,
      rotate: ro
    };
  }

  async #sendMLParams(payload) {
    try { await this.#postJSON('/meshlab_params', payload); }
    catch(e){ console.error(e); }
  }

  async #getFrameSize() {
    try {
      const r = await fetch(`/frame.jpg?ts=${Date.now()}`, { cache:'no-store' });
      if (!r.ok) return null;
      const len = r.headers.get('Content-Length');
      if (len != null) {
        const n = parseInt(len, 10);
        return Number.isFinite(n) ? n : null;
      }
      const blob = await r.blob();
      return blob.size || null;
    } catch (_) {
      return null;
    }
  }

  #showMLProgress() {
    const wrap = this.dom.mlProgressWrap;
    const bar  = this.dom.mlProgressBar;
    if (!wrap || !bar) return;

    if (this._mlProgressTimer) {
      clearInterval(this._mlProgressTimer);
      this._mlProgressTimer = null;
    }

    wrap.style.opacity = '1';
    bar.style.width = '0%';
    bar.style.transform = 'translateX(-40%)';

    const start = performance.now();
    const cycleMs = 1400;

    this._mlProgressTimer = setInterval(() => {
      const t = (performance.now() - start) % cycleMs;
      const phase = t / cycleMs;

      let w;
      let tx;
      if (phase < 0.5) {
        w  = 15 + phase * 150;     // 15% -> ~90%
        tx = -40 + phase * 60;     // -40% -> -10%
      } else {
        const p = (phase - 0.5) * 2;
        w  = 165 - p * 90;         // 90% -> ~75%
        tx = -10 + p * 40;         // -10% -> ~30%
      }
      bar.style.width = `${w}%`;
      bar.style.transform = `translateX(${tx}%)`;
    }, 70);
  }

  #hideMLProgress() {
    const wrap = this.dom.mlProgressWrap;
    const bar  = this.dom.mlProgressBar;
    if (!wrap || !bar) return;

    if (this._mlProgressTimer) {
      clearInterval(this._mlProgressTimer);
      this._mlProgressTimer = null;
    }

    bar.style.width = '100%';
    bar.style.transform = 'translateX(0%)';

    setTimeout(() => {
      wrap.style.opacity = '0';
    }, 180);
  }

  async #startMLApplyWithProgress() {
    // annule un éventuel polling précédent
    if (this._mlStatusTimer) {
      clearInterval(this._mlStatusTimer);
      this._mlStatusTimer = null;
    }

    // baseline sur le JPEG courant
    this._mlBaselineSize = await this.#getFrameSize();
    this.#showMLProgress();

    const payload = this.#gatherMLPayload();

    try {
      await this.#sendMLParams(payload);
    } catch (e) {
      console.error(e);
      this.#hideMLProgress();
      return;
    }

    const startedAt    = performance.now();
    const maxMs        = 20000; // timeout max
    const minVisibleMs = 450;   // durée mini pour éviter le flash

    this._mlStatusTimer = setInterval(async () => {
      if (this._mlPollingBusy) return;
      this._mlPollingBusy = true;
      try {
        const now = performance.now();
        if (now - startedAt > maxMs) {
          clearInterval(this._mlStatusTimer);
          this._mlStatusTimer = null;
          this.#hideMLProgress();
          return;
        }

        const size = await this.#getFrameSize();
        if (size && this._mlBaselineSize && Math.abs(size - this._mlBaselineSize) > 64) {
          // nouvelle frame significativement différente → APSS appliqué
          const elapsed = now - startedAt;
          const remaining = Math.max(0, minVisibleMs - elapsed);
          if (remaining > 0) {
            setTimeout(() => this.#hideMLProgress(), remaining);
          } else {
            this.#hideMLProgress();
          }
          clearInterval(this._mlStatusTimer);
          this._mlStatusTimer = null;
        }
      } catch (e) {
        console.error(e);
      } finally {
        this._mlPollingBusy = false;
      }
    }, 350);
  }

  // -- Boutons Zoom + / Zoom − (icônes SVG)
  #createZoomButtons() {
    if (!this.dom.container) return;

    const wrap = document.createElement('div');
    wrap.style.position = 'absolute';
    wrap.style.right = '12px';
    wrap.style.bottom = '12px';
    wrap.style.display = 'flex';
    wrap.style.gap = '8px';
    wrap.style.zIndex = '20';

    const mkIconBtn = (svg, aria) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.innerHTML = svg;
      b.setAttribute('aria-label', aria);
      b.title = aria;

      // style des boutons icônes
      b.style.width = '40px';
      b.style.height = '40px';
      b.style.display = 'inline-flex';
      b.style.alignItems = 'center';
      b.style.justifyContent = 'center';
      b.style.padding = '0';
      b.style.border = '1px solid rgba(255,255,255,0.2)';
      b.style.borderRadius = '12px';
      b.style.background = 'rgba(0,0,0,0.45)';
      b.style.color = '#fff';
      b.style.backdropFilter = 'blur(4px)';
      b.style.cursor = 'pointer';
      b.style.userSelect = 'none';
      b.style.boxShadow = '0 2px 8px rgba(0,0,0,0.35)';
      b.addEventListener('mouseenter', ()=> b.style.background = 'rgba(0,0,0,0.6)');
      b.addEventListener('mouseleave', ()=> b.style.background = 'rgba(0,0,0,0.45)');
      return b;
    };

    const ICON_ZOOM_IN = `
      <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"
           fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"
           style="pointer-events:none">
        <circle cx="11" cy="11" r="7"></circle>
        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
        <line x1="11" y1="8"  x2="11" y2="14"></line>
        <line x1="8"  y1="11" x2="14" y2="11"></line>
      </svg>`;

    const ICON_ZOOM_OUT = `
      <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true"
           fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"
           style="pointer-events:none">
        <circle cx="11" cy="11" r="7"></circle>
        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
        <line x1="8"  y1="11" x2="14" y2="11"></line>
      </svg>`;

    const btnIn  = mkIconBtn(ICON_ZOOM_IN,  'Zoomer (IN)');
    const btnOut = mkIconBtn(ICON_ZOOM_OUT, 'Dézoomer (OUT)');

    // Clic simple (sémantique)
    btnIn.addEventListener('click',  () => this.#sendZoomSem(-1));
    btnOut.addEventListener('click', () => this.#sendZoomSem(+1));

    // Press & hold (maintenir pour zoom continu)
    const startHold = (dir) => {
      if (this._zoomHoldTimer) clearInterval(this._zoomHoldTimer);
      this.#sendZoomSem(dir);
      this._zoomHoldTimer = setInterval(()=> this.#sendZoomSem(dir), 70);
    };
    const stopHold = () => {
      if (this._zoomHoldTimer) { clearInterval(this._zoomHoldTimer); this._zoomHoldTimer = null; }
    };
    btnIn.addEventListener('mousedown',  ()=> startHold(-1));
    btnOut.addEventListener('mousedown', ()=> startHold(+1));
    window.addEventListener('mouseup', stopHold);
    btnIn.addEventListener('mouseleave', stopHold);
    btnOut.addEventListener('mouseleave', stopHold);

    wrap.appendChild(btnIn);
    wrap.appendChild(btnOut);

    // le conteneur 3D doit être positionné pour l'absolu : on force si nécessaire
    const cs = getComputedStyle(this.dom.container);
    if (cs.position === 'static') this.dom.container.style.position = 'relative';

    this.dom.container.appendChild(wrap);
  }

  // --- Helpers zoom ---
  #clampStep(step) {
    const s = Number(step) || 1.12;
    return Math.min(1.5, Math.max(1.02, s));
  }

  // Boutons & clavier : zoom sémantique (ignore invert du backend)
  #sendZoomSem(dir, step = 1.12) {
    // dir : -1 => IN, +1 => OUT
    const sem = (dir < 0) ? 'in' : 'out';
    this.#postJSON('/control', { action:'zoom_sem', sem, step: this.#clampStep(step) });
  }

  // Molette : zoom directionnel (respecte invert côté backend)
  #sendZoomDir(dir, step = 1.12) {
    // dir : +1 => OUT, -1 => IN
    this.#postJSON('/control', { action:'zoom_dir', dir, step: this.#clampStep(step) });
  }

  // DnD
  #bindDnD() {
    const dz = this.dom.drop, left = this.dom.left;
    const dzOn  = (e)=>{ e.preventDefault(); dz?.classList.add('show'); };
    const dzOff = (e)=>{ e.preventDefault(); dz?.classList.remove('show'); };
    ['dragenter','dragover'].forEach(ev => left?.addEventListener(ev, dzOn));
    ['dragleave','drop'].forEach(ev => left?.addEventListener(ev, dzOff));
    left?.addEventListener('drop', async (e) => {
      e.preventDefault();
      const f = [...(e.dataTransfer?.files||[])].find(x=>x.name.toLowerCase().endsWith('.obj'));
      if (!f){ alert('Dépose un .obj'); return; }
      const text = await f.text();
      await this.#uploadOBJToServer(f.name || 'dropped.obj', text);
    });
  }

  async #uploadOBJToServer(name, text) {
    const obj_b64 = btoa(unescape(encodeURIComponent(text)));
    const res = await fetch('/upload_obj', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ obj_name:name, obj_b64 })
    });
    const json = await res.json().catch(()=>null);
    if (!res.ok || !json || !json.ok) { console.error(json); alert('Upload OBJ serveur échoué'); return; }
    this._serverObjToken = json.token;
    await this.#postJSON('/fit', {});
  }

  // ---- Switch moteur (UI + serveur) ----
  #setEngineButtons(name) {
    this.state.engine = name;
    if (this.dom.btnEngineO3D) this.dom.btnEngineO3D.classList.toggle('active', name==='open3d');
    if (this.dom.btnEngineML ) this.dom.btnEngineML .classList.toggle('active', name==='meshlab');
    if (this.dom.mlPanel) this.dom.mlPanel.style.display = (name==='meshlab') ? '' : 'none';
  }

  async #switchEngine(engineName) {
    try {
      const r = await this.#postJSON('/engine', { engine: engineName });
      const j = await r.json().catch(()=>null);
      if (!r.ok || !j || j.ok === false) { alert('Changement de moteur échoué'); return; }

      this.#setEngineButtons(engineName);

      // recale stage + pousse taille backend immédiatement
      this.#fitStageToLeft();
      await this.#syncSize();

      this.#refreshStream();

      if (engineName === 'meshlab') {
        this.#sendMLParams(this.#gatherMLPayload()).catch(()=>{});
      }
    } catch (e) {
      console.error(e);
      alert('Changement de moteur échoué');
    }
  }
}

const viewer = new ArcheoViewer();
viewer.init();
window.ArcheoViewerInstance = viewer;
