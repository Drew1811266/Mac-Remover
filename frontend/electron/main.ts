import { app, BrowserWindow, Menu, shell } from 'electron';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { registerIpcHandlers, rendererHtmlPath } from './ipcHandlers.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND_ROOT = path.resolve(__dirname, '..', '..');
const PROJECT_ROOT = path.resolve(FRONTEND_ROOT, '..');
const USER_DATA_DIR_OVERRIDE = process.env.WMR_USER_DATA_DIR?.trim();

app.setName('Mac Watermark Remover');
if (USER_DATA_DIR_OVERRIDE) {
  app.setPath('userData', path.resolve(USER_DATA_DIR_OVERRIDE));
}

let mainWindow: BrowserWindow | null = null;

async function createWindow(): Promise<void> {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1200,
    minHeight: 700,
    title: 'Mac Watermark Remover',
    backgroundColor: '#f7f8fa',
    ...(process.platform === 'darwin'
      ? {
          titleBarStyle: 'hiddenInset' as const,
          trafficLightPosition: { x: 18, y: 18 },
        }
      : {}),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  registerIpcHandlers({
    window: mainWindow,
    userDataDir: app.getPath('userData'),
    appRoot: app.isPackaged ? process.resourcesPath : PROJECT_ROOT,
    manifestUrl: process.env.WMR_MODEL_MANIFEST_URL,
  });

  const renderer = rendererHtmlPath(app.isPackaged ? process.resourcesPath : PROJECT_ROOT, app.isPackaged);
  if (/^https?:\/\//.test(renderer)) {
    await mainWindow.loadURL(renderer);
  } else {
    await mainWindow.loadURL(pathToFileURL(renderer).toString());
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function installMenu(): void {
  const template: Electron.MenuItemConstructorOptions[] = [
    {
      label: app.name,
      submenu: [
        { role: 'about' },
        { type: 'separator' },
        { role: 'hide' },
        { role: 'hideOthers' },
        { type: 'separator' },
        { role: 'quit' },
      ],
    },
    {
      label: 'File',
      submenu: [{ role: 'close' }],
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' },
      ],
    },
    {
      label: 'Help',
      submenu: [
        {
          label: 'Project Repository',
          click: () => {
            void shell.openExternal('https://github.com/');
          },
        },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

app.whenReady().then(async () => {
  installMenu();
  await createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      void createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
