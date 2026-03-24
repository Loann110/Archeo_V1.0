// main.js — CommonJS
const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fetch = require('node-fetch');
let win = null;
let backend = null;

async function waitFor(url, timeoutMs = 25000) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeoutMs) {
    try {
      const r = await fetch(url, { timeout: 2000 });
      if (r.ok) return true;
    } catch (_) {}
    await new Promise(r => setTimeout(r, 400));
  }
  return false;
}

async function createWindow() {
  const backendPort = 5000;
  const isPackaged = app.isPackaged;
  let backendFolder, exePath;

  if (!isPackaged) {
    // MODE DEV → npm start
    backendFolder = path.join(__dirname, "backend_dist", "backend");
    exePath = path.join(backendFolder, "backend.exe");
  } else {
    // MODE BUILD → appli .exe installée
    backendFolder = path.join(process.resourcesPath, "backend");
    exePath = path.join(backendFolder, "backend.exe");
  }

  console.log("[PATH] Backend exe =", exePath);

  const internalDir = path.join(backendFolder, "_internal");

  backend = spawn(exePath, [], {
    cwd: backendFolder,
    env: {
      ...process.env,
      APP_PORT: String(backendPort),
      SEG_DEVICE: "cpu",
      O3D_FORCE_LEGACY: "1",
      PATH: `${backendFolder};${internalDir};${process.env.PATH}`
    },
    windowsHide: true
  });

  backend.stdout?.on('data', d => console.log('[backend]', d.toString()));
  backend.stderr?.on('data', d => console.error('[backend]', d.toString()));
  backend.on('exit', code => console.error('[backend exit]', code));

  const ok = await waitFor(`http://127.0.0.1:${backendPort}/health`, 25000);
  if (!ok) {
    console.error(' Backend indisponible (timeout /health)');
    return app.quit();
  }

  win = new BrowserWindow({
  width: 1800,
  height: 800,
  useContentSize: true,     // la taille correspond au contenu 
  resizable: false,         // interdit redimensionnement
  maximizable: false,       // interdit maximiser
  fullscreenable: false,    // interdit plein écran (UI)
  autoHideMenuBar: true,
  webPreferences: { nodeIntegration: false, contextIsolation: true }
});


  await win.loadURL(`http://127.0.0.1:${backendPort}/`);
  win.on('closed', () => win = null);
}

app.on('ready', createWindow);
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
app.on('before-quit', () => {
  if (backend && backend.pid) try { process.kill(backend.pid); } catch (_) {}
});
