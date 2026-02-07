/**
 * kiQuant - Frontend Application
 */

// Constants
const Mode = {
    POSITIVE: 0,
    NEGATIVE: 1,
    SELECT: 2,
    ERASER: 3,
    PAN: 4,
    ROI: 5
};

const MarkerClass = {
    POSITIVE: 0,
    NEGATIVE: 1
};

const MARKER_RADIUS = 6;
const MARKER_THICKNESS = 3;

// Zoom constants
const MIN_ZOOM = 0.1;
const MAX_ZOOM = 10;
const ZOOM_STEP = 0.25;
const OVERVIEW_MAX_WIDTH = 200;
const OVERVIEW_MAX_HEIGHT = 150;

// State
let currentMode = Mode.POSITIVE;
let currentImage = null;
let markers = [];
let isProjectLoaded = false;
let isQuickMode = false;

// Zoom/Pan state
let zoom = 1;
let panX = 0;
let panY = 0;
let isPanning = false;
let panStartX = 0;
let panStartY = 0;
let lastPanX = 0;
let lastPanY = 0;

// Drag selection state
let isDragging = false;
let dragStart = { x: 0, y: 0 };
let dragEnd = { x: 0, y: 0 };

// Context menu state
let contextMenuJustClosed = false;

// Overview drag state
let isOverviewDragging = false;
let overviewAnimationFrame = null;

// Guide and selection state
let showGuide = true;
let selectionType = 'rect'; // 'rect' or 'lasso'
let lassoPoints = []; // For lasso selection
let selectionAdditive = false; // Shift held = add to selection
let selectionClickStart = null; // Track click position for single-click select

// Track unsaved changes
let hasUnsavedChanges = false;

// ROI state
let currentROI = null; // {x, y, width, height} or null
let isDrawingROI = false;
let roiStart = { x: 0, y: 0 };

// Eraser state
let eraserRadius = 15; // Default eraser radius in image pixels
let isErasing = false;
let eraserImagePos = null; // Current position in image coordinates {x, y}
let eraserHistorySaved = false; // Track if we've saved history for current stroke
let eraserPending = false; // Prevent overlapping erase calls
let eraserHeldMode = null; // Previous mode when holding X for temporary eraser

// Canvas and context
let canvas, ctx;
let overlayCanvas, overlayCtx;
let viewport;
let overviewCanvas, overviewCtx;
let overviewViewport;

// DOM Elements
let elements = {};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', init);

function init() {
    // Cache DOM elements
    elements = {
        canvas: document.getElementById('canvas'),
        overlayCanvas: document.getElementById('overlay-canvas'),
        canvasViewport: document.getElementById('canvas-viewport'),
        canvasContainer: document.getElementById('canvas-container'),
        welcomeMessage: document.getElementById('welcome-message'),
        progress: document.getElementById('progress'),
        countPositive: document.getElementById('count-positive'),
        countNegative: document.getElementById('count-negative'),
        countTotal: document.getElementById('count-total'),
        countPi: document.getElementById('count-pi'),
        statusFilename: document.getElementById('status-filename'),
        statusMode: document.getElementById('status-mode'),
        statusSummary: document.getElementById('status-summary'),
        btnFile: document.getElementById('btn-file'),
        fileMenu: document.getElementById('file-menu'),
        btnUndo: document.getElementById('btn-undo'),
        btnRedo: document.getElementById('btn-redo'),
        btnPositive: document.getElementById('btn-positive'),
        btnNegative: document.getElementById('btn-negative'),
        btnSelect: document.getElementById('btn-select'),
        btnEraser: document.getElementById('btn-eraser'),
        btnPrev: document.getElementById('btn-prev'),
        btnNext: document.getElementById('btn-next'),
        welcomeNew: document.getElementById('welcome-new'),
        welcomeLoad: document.getElementById('welcome-load'),
        welcomeQuick: document.getElementById('welcome-quick'),
        contextMenu: document.getElementById('context-menu'),
        overviewPanel: document.getElementById('overview-panel'),
        overviewCanvas: document.getElementById('overview-canvas'),
        overviewViewport: document.getElementById('overview-viewport'),
        zoomControls: document.getElementById('zoom-controls'),
        zoomLevel: document.getElementById('zoom-level'),
        btnZoomIn: document.getElementById('btn-zoom-in'),
        btnZoomOut: document.getElementById('btn-zoom-out'),
        btnZoomFit: document.getElementById('btn-zoom-fit'),
        btnZoom100: document.getElementById('btn-zoom-100'),
        btnToggleGuide: document.getElementById('btn-toggle-guide'),
        btnSelectDropdown: document.getElementById('btn-select-dropdown'),
        selectMenu: document.getElementById('select-menu'),
        btnEraserDropdown: document.getElementById('btn-eraser-dropdown'),
        eraserMenu: document.getElementById('eraser-menu'),
        aboutModal: document.getElementById('about-modal'),
        aboutVersion: document.getElementById('about-version'),
        aboutCopyright: document.querySelector('.about-copyright'),
        aboutLink: document.getElementById('about-link'),
        aboutClose: document.getElementById('about-close'),
        // Detection elements
        detectionGroup: document.getElementById('detection-group'),
        detectModel: document.getElementById('detect-model'),
        detectClassify: document.getElementById('detect-classify'),
        thresholdContainer: document.getElementById('threshold-container'),
        dabThreshold: document.getElementById('dab-threshold'),
        thresholdValue: document.getElementById('threshold-value'),
        btnDetect: document.getElementById('btn-detect'),
        detectionProgress: document.getElementById('detection-progress'),
        detectionProgressBar: document.getElementById('detection-progress-bar'),
        detectionProgressText: document.getElementById('detection-progress-text'),
        // Detection settings elements
        btnDetectSettings: document.getElementById('btn-detect-settings'),
        detectionSettingsModal: document.getElementById('detection-settings-modal'),
        settingDiameter: document.getElementById('setting-diameter'),
        settingCellprob: document.getElementById('setting-cellprob'),
        settingCellprobValue: document.getElementById('setting-cellprob-value'),
        settingFlow: document.getElementById('setting-flow'),
        settingFlowValue: document.getElementById('setting-flow-value'),
        settingProb: document.getElementById('setting-prob'),
        settingProbValue: document.getElementById('setting-prob-value'),
        settingNms: document.getElementById('setting-nms'),
        settingNmsValue: document.getElementById('setting-nms-value'),
        cellposeSettings: document.getElementById('cellpose-settings'),
        stardistSettings: document.getElementById('stardist-settings'),
        kinetSettings: document.getElementById('kinet-settings'),
        settingKinetThreshold: document.getElementById('setting-kinet-threshold'),
        settingKinetThresholdValue: document.getElementById('setting-kinet-threshold-value'),
        settingKinetDistance: document.getElementById('setting-kinet-distance'),
        settingKinetTilesize: document.getElementById('setting-kinet-tilesize'),
        settingsReset: document.getElementById('settings-reset'),
        settingsClose: document.getElementById('settings-close'),
        // ROI elements
        btnRoi: document.getElementById('btn-roi'),
        btnHotspot: document.getElementById('btn-hotspot'),
        btnClearRoi: document.getElementById('btn-clear-roi')
    };

    canvas = elements.canvas;
    ctx = canvas.getContext('2d');
    overlayCanvas = elements.overlayCanvas;
    overlayCtx = overlayCanvas.getContext('2d');
    viewport = elements.canvasViewport;
    overviewCanvas = elements.overviewCanvas;
    overviewCtx = overviewCanvas.getContext('2d');
    overviewViewport = elements.overviewViewport;

    // Bind event listeners
    bindEvents();

    // Initial UI state
    updateModeUI();
    elements.btnToggleGuide.classList.toggle('guide-active', showGuide);

    // Initialize detection controls
    initDetection();
}

function bindEvents() {
    // File dropdown menu
    elements.btnFile.addEventListener('click', toggleFileMenu);
    elements.fileMenu.addEventListener('click', handleFileMenuAction);

    // Undo/Redo
    elements.btnUndo.addEventListener('click', handleUndo);
    elements.btnRedo.addEventListener('click', handleRedo);

    // Welcome buttons
    elements.welcomeNew.addEventListener('click', handleNewProject);
    elements.welcomeLoad.addEventListener('click', handleLoadProject);
    elements.welcomeQuick.addEventListener('click', handleQuickMode);

    // Mode buttons
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            setMode(parseInt(btn.dataset.mode));
        });
    });

    // Navigation buttons
    elements.btnPrev.addEventListener('click', handlePrevImage);
    elements.btnNext.addEventListener('click', handleNextImage);

    // Canvas events
    canvas.addEventListener('mousedown', handleCanvasMouseDown);
    canvas.addEventListener('mousemove', handleCanvasMouseMove);
    canvas.addEventListener('mouseup', handleCanvasMouseUp);
    canvas.addEventListener('mouseleave', handleCanvasMouseLeave);
    canvas.addEventListener('contextmenu', handleContextMenu);
    canvas.addEventListener('wheel', handleCanvasWheel, { passive: false });

    // Overview panel drag to navigate
    elements.overviewPanel.addEventListener('mousedown', handleOverviewMouseDown);
    document.addEventListener('mousemove', handleOverviewMouseMove);
    document.addEventListener('mouseup', handleOverviewMouseUp);

    // Document-level handlers for selection/pan continuation outside canvas
    document.addEventListener('mousemove', handleDocumentMouseMove);
    document.addEventListener('mouseup', handleDocumentMouseUp);

    // Zoom controls
    elements.btnZoomIn.addEventListener('click', () => setZoom(zoom + ZOOM_STEP));
    elements.btnZoomOut.addEventListener('click', () => setZoom(zoom - ZOOM_STEP));
    elements.btnZoomFit.addEventListener('click', zoomToFit);
    elements.btnZoom100.addEventListener('click', () => setZoom(1));
    elements.btnToggleGuide.addEventListener('click', toggleGuide);

    // Selection type dropdown
    elements.btnSelectDropdown.addEventListener('click', toggleSelectMenu);
    elements.selectMenu.addEventListener('click', handleSelectMenuAction);

    // Eraser size dropdown
    elements.btnEraserDropdown.addEventListener('click', toggleEraserMenu);
    elements.eraserMenu.addEventListener('click', handleEraserMenuAction);

    // Context menu events
    document.querySelectorAll('.context-menu-item').forEach(item => {
        item.addEventListener('click', handleContextMenuAction);
    });

    // Hide menus on any click outside
    document.addEventListener('mousedown', (e) => {
        // Hide file menu
        if (!elements.fileMenu.contains(e.target) && !elements.btnFile.contains(e.target)) {
            hideFileMenu();
        }

        // Hide select menu
        if (!elements.selectMenu.contains(e.target) && !elements.btnSelectDropdown.contains(e.target)) {
            hideSelectMenu();
        }

        // Hide eraser menu
        if (!elements.eraserMenu.contains(e.target) && !elements.btnEraserDropdown.contains(e.target)) {
            hideEraserMenu();
        }

        // Hide context menu
        if (!elements.contextMenu.contains(e.target) && !elements.contextMenu.classList.contains('hidden')) {
            hideContextMenu();
            contextMenuJustClosed = true;
            setTimeout(() => { contextMenuJustClosed = false; }, 50);
        }
    }, true); // Use capture phase

    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('keyup', handleKeyUp);

    // Window resize
    window.addEventListener('resize', handleResize);

    // Warn before closing with unsaved changes
    window.addEventListener('beforeunload', handleBeforeUnload);

    // About modal
    elements.aboutClose.addEventListener('click', hideAboutModal);
    elements.aboutModal.addEventListener('click', (e) => {
        if (e.target === elements.aboutModal) hideAboutModal();
    });
    elements.aboutLink.addEventListener('click', (e) => {
        e.preventDefault();
        eel.open_url('https://github.com/cornish/kiquant')();
    });

    // Detection controls
    elements.btnDetect.addEventListener('click', handleDetect);
    elements.detectClassify.addEventListener('change', handleClassifyModeChange);
    elements.dabThreshold.addEventListener('input', handleThresholdChange);

    // Detection settings
    elements.btnDetectSettings.addEventListener('click', showDetectionSettings);
    elements.settingsClose.addEventListener('click', hideDetectionSettings);
    elements.settingsReset.addEventListener('click', resetDetectionSettings);
    elements.detectionSettingsModal.addEventListener('click', (e) => {
        if (e.target === elements.detectionSettingsModal) hideDetectionSettings();
    });
    elements.detectModel.addEventListener('change', updateSettingsVisibility);

    // Settings sliders
    elements.settingCellprob.addEventListener('input', () => {
        elements.settingCellprobValue.textContent = parseFloat(elements.settingCellprob.value).toFixed(1);
    });
    elements.settingFlow.addEventListener('input', () => {
        elements.settingFlowValue.textContent = parseFloat(elements.settingFlow.value).toFixed(2);
    });
    elements.settingProb.addEventListener('input', () => {
        elements.settingProbValue.textContent = parseFloat(elements.settingProb.value).toFixed(2);
    });
    elements.settingNms.addEventListener('input', () => {
        elements.settingNmsValue.textContent = parseFloat(elements.settingNms.value).toFixed(2);
    });
    elements.settingKinetThreshold.addEventListener('input', () => {
        elements.settingKinetThresholdValue.textContent = parseFloat(elements.settingKinetThreshold.value).toFixed(2);
    });

    // ROI controls
    elements.btnRoi.addEventListener('click', () => setMode(Mode.ROI));
    elements.btnHotspot.addEventListener('click', handleFindHotspot);
    elements.btnClearRoi.addEventListener('click', handleClearROI);
}

async function showAboutModal() {
    const version = await eel.get_version()();
    elements.aboutVersion.textContent = `Version ${version}`;
    elements.aboutCopyright.textContent = 'Copyright © 2025-2026 Toby Cornish';
    elements.aboutModal.classList.remove('hidden');
}

function hideAboutModal() {
    elements.aboutModal.classList.add('hidden');
}

function handleBeforeUnload(e) {
    if (hasUnsavedChanges && isProjectLoaded && !isQuickMode) {
        e.preventDefault();
        e.returnValue = 'You have unsaved changes. Are you sure you want to leave?';
        return e.returnValue;
    }
}

function markUnsavedChanges() {
    hasUnsavedChanges = true;
}

function clearUnsavedChanges() {
    hasUnsavedChanges = false;
}

// ============== Project Management ==============

async function handleNewProject() {
    const result = await eel.new_project()();
    if (result.success) {
        isProjectLoaded = true;
        isQuickMode = false;
        clearUnsavedChanges();
        hideWelcome();
        await loadCurrentImage();
        updateSummary();
    } else {
        if (result.message !== 'No directory selected' &&
            result.message !== 'No project file selected' &&
            result.message !== 'No output file selected') {
            alert(result.message);
        }
    }
}

async function handleLoadProject() {
    const result = await eel.load_project()();
    if (result.success) {
        isProjectLoaded = true;
        isQuickMode = false;
        clearUnsavedChanges();
        hideWelcome();
        await loadCurrentImage();
        updateSummary();
    } else {
        if (result.message !== 'No project file selected') {
            alert(result.message);
        }
    }
}

async function handleQuickMode() {
    const result = await eel.quick_mode()();
    if (result.success) {
        isProjectLoaded = true;
        isQuickMode = true;
        hideWelcome();
        await loadCurrentImage();
        updateSummary();
    } else {
        if (result.message !== 'No images selected') {
            alert(result.message);
        }
    }
}

async function handleSaveProject() {
    if (isQuickMode) {
        alert('Quick Mode does not support saving projects.\nUse "Export CSV" to save your results.');
        return;
    }
    const result = await eel.save_project()();
    if (result.success) {
        clearUnsavedChanges();
    } else {
        alert(result.message);
    }
}

async function handleExportCSV() {
    const result = await eel.export_csv()();
    if (result.success) {
        const summary = result.summary;
        elements.statusSummary.textContent =
            `PI: ${summary.proliferation_index}% (${summary.positive}/${summary.total})`;
        alert(`Results exported!\n${result.message}`);
    } else {
        alert(result.message);
    }
}

function hideWelcome() {
    elements.welcomeMessage.classList.add('hidden');
    viewport.classList.add('loaded');
    elements.overviewPanel.classList.remove('hidden');
    elements.zoomControls.classList.remove('hidden');
    // Update detection button state now that project is loaded
    updateDetectionButtonState();
}

// ============== File Menu ==============

function toggleFileMenu(e) {
    e.stopPropagation();
    elements.fileMenu.classList.toggle('hidden');
}

function hideFileMenu() {
    elements.fileMenu.classList.add('hidden');
}

async function handleFileMenuAction(e) {
    const item = e.target.closest('.dropdown-item');
    if (!item) return;

    const action = item.dataset.action;
    hideFileMenu();

    switch (action) {
        case 'new':
            handleNewProject();
            break;
        case 'load':
            handleLoadProject();
            break;
        case 'quick':
            handleQuickMode();
            break;
        case 'save':
            handleSaveProject();
            break;
        case 'export':
            handleExportCSV();
            break;
        case 'about':
            showAboutModal();
            break;
    }
}

// ============== Guide Toggle ==============

function toggleGuide() {
    showGuide = !showGuide;
    elements.btnToggleGuide.classList.toggle('guide-active', showGuide);
    render();
    renderOverview();
}

// ============== Selection Type Menu ==============

function toggleSelectMenu(e) {
    e.stopPropagation();
    elements.selectMenu.classList.toggle('hidden');
    updateSelectMenuUI();
}

function hideSelectMenu() {
    elements.selectMenu.classList.add('hidden');
}

function updateSelectMenuUI() {
    document.querySelectorAll('#select-menu .dropdown-item').forEach(item => {
        item.classList.toggle('active', item.dataset.selectType === selectionType);
    });
}

function handleSelectMenuAction(e) {
    const item = e.target.closest('.dropdown-item');
    if (!item) return;

    selectionType = item.dataset.selectType;
    hideSelectMenu();
    updateSelectMenuUI();

    // Switch to select mode when changing selection type
    setMode(Mode.SELECT);
}

// ============== Eraser Size Menu ==============

function toggleEraserMenu(e) {
    e.stopPropagation();
    elements.eraserMenu.classList.toggle('hidden');
    updateEraserMenuUI();
}

function hideEraserMenu() {
    elements.eraserMenu.classList.add('hidden');
}

function updateEraserMenuUI() {
    document.querySelectorAll('#eraser-menu .dropdown-item').forEach(item => {
        item.classList.toggle('active', parseInt(item.dataset.eraserSize) === eraserRadius);
    });
}

function handleEraserMenuAction(e) {
    const item = e.target.closest('.dropdown-item');
    if (!item) return;

    eraserRadius = parseInt(item.dataset.eraserSize);
    hideEraserMenu();
    updateEraserMenuUI();

    // Switch to eraser mode when changing eraser size
    setMode(Mode.ERASER);
}

// ============== Undo/Redo ==============

async function handleUndo() {
    if (!isProjectLoaded) return;
    const result = await eel.undo()();
    if (result) {
        markers = result.markers;
        updateCounts(result.positive_count, result.negative_count);
        updateUndoRedoButtons(result.can_undo, result.can_redo);
        markUnsavedChanges();
        render();
        renderOverview();
    }
}

async function handleRedo() {
    if (!isProjectLoaded) return;
    const result = await eel.redo()();
    if (result) {
        markers = result.markers;
        updateCounts(result.positive_count, result.negative_count);
        updateUndoRedoButtons(result.can_undo, result.can_redo);
        markUnsavedChanges();
        render();
        renderOverview();
    }
}

function updateUndoRedoButtons(canUndo, canRedo) {
    elements.btnUndo.disabled = !canUndo;
    elements.btnRedo.disabled = !canRedo;
}

// ============== Image Loading ==============

async function loadCurrentImage() {
    const data = await eel.get_image_data()();
    if (!data || data.error) {
        console.error('Failed to load image:', data?.error);
        return;
    }

    const isFirstLoad = !currentImage;
    currentImage = new Image();
    currentImage.onload = async () => {
        // Set canvas size to match image
        canvas.width = data.width;
        canvas.height = data.height;

        // Store markers
        markers = data.markers;
        currentROI = data.roi || null;

        // Fit to window on first load, preserve viewport on navigation
        if (isFirstLoad) {
            zoomToFit();
        } else {
            constrainPan();
        }

        // Render
        render();
        renderOverview();

        // Update UI
        updateProgress(data.index + 1, data.total);
        updateCounts(data.positive_count, data.negative_count);
        elements.statusFilename.textContent = data.filename;

        // Update undo/redo state
        const undoRedoState = await eel.get_undo_redo_state()();
        updateUndoRedoButtons(undoRedoState.can_undo, undoRedoState.can_redo);
    };
    currentImage.src = data.image;
}

// ============== Navigation ==============

async function handlePrevImage() {
    if (!isProjectLoaded) return;
    const data = await eel.previous_image()();
    if (data && !data.error) {
        loadImageData(data);
    }
}

async function handleNextImage() {
    if (!isProjectLoaded) return;
    const data = await eel.next_image()();
    if (data && !data.error) {
        loadImageData(data);
    }
}

async function loadImageData(data) {
    currentImage = new Image();
    currentImage.onload = async () => {
        canvas.width = data.width;
        canvas.height = data.height;
        markers = data.markers;
        currentROI = data.roi || null;

        // Constrain pan to new image bounds, preserve zoom level
        constrainPan();

        render();
        renderOverview();
        updateProgress(data.index + 1, data.total);
        updateCounts(data.positive_count, data.negative_count);
        elements.statusFilename.textContent = data.filename;

        const undoRedoState = await eel.get_undo_redo_state()();
        updateUndoRedoButtons(undoRedoState.can_undo, undoRedoState.can_redo);
    };
    currentImage.src = data.image;
}

// ============== Zoom/Pan ==============

function resetView() {
    zoom = 1;
    panX = 0;
    panY = 0;
    zoomToFit();
}

function setZoom(newZoom) {
    zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, newZoom));
    constrainPan();
    updateZoomUI();
    render();
    updateOverviewViewport();
}

function zoomToFit() {
    if (!currentImage) return;

    const container = elements.canvasViewport;
    // Fallback to canvasContainer if viewport has no dimensions yet
    const containerWidth = (container.clientWidth || elements.canvasContainer.clientWidth) - 40;
    const containerHeight = (container.clientHeight || elements.canvasContainer.clientHeight) - 40;

    const scaleX = containerWidth / currentImage.width;
    const scaleY = containerHeight / currentImage.height;
    zoom = Math.min(scaleX, scaleY, 1); // Don't zoom beyond 100%

    // Center the image
    panX = 0;
    panY = 0;

    updateZoomUI();
    render();
    updateOverviewViewport();
}

function constrainPan() {
    if (!currentImage) return;

    const container = elements.canvasViewport;
    const viewWidth = container.clientWidth || elements.canvasContainer.clientWidth;
    const viewHeight = container.clientHeight || elements.canvasContainer.clientHeight;
    const imgWidth = currentImage.width * zoom;
    const imgHeight = currentImage.height * zoom;

    // Allow panning only if image is larger than viewport
    if (imgWidth <= viewWidth) {
        panX = 0;
    } else {
        const maxPan = (imgWidth - viewWidth) / 2;
        panX = Math.max(-maxPan, Math.min(maxPan, panX));
    }

    if (imgHeight <= viewHeight) {
        panY = 0;
    } else {
        const maxPan = (imgHeight - viewHeight) / 2;
        panY = Math.max(-maxPan, Math.min(maxPan, panY));
    }
}

function updateZoomUI() {
    elements.zoomLevel.textContent = Math.round(zoom * 100) + '%';
}

function handleCanvasWheel(e) {
    if (!isProjectLoaded || !currentImage) return;
    e.preventDefault();

    // Get mouse position relative to viewport
    const container = elements.canvasViewport;
    const rect = container.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Calculate image coordinates under cursor before zoom
    const viewWidth = container.clientWidth || elements.canvasContainer.clientWidth;
    const viewHeight = container.clientHeight || elements.canvasContainer.clientHeight;

    // Image point under cursor
    const imgX = (mouseX - viewWidth / 2 - panX) / zoom + currentImage.width / 2;
    const imgY = (mouseY - viewHeight / 2 - panY) / zoom + currentImage.height / 2;

    // Apply zoom
    const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP;
    const oldZoom = zoom;
    zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom + delta));

    if (zoom !== oldZoom) {
        // Adjust pan so the same image point stays under cursor
        panX = (currentImage.width / 2 - imgX) * zoom + mouseX - viewWidth / 2;
        panY = (currentImage.height / 2 - imgY) * zoom + mouseY - viewHeight / 2;

        constrainPan();
        updateZoomUI();
        render();
        updateOverviewViewport();
    }
}

function handleResize() {
    if (isProjectLoaded && currentImage) {
        constrainPan();
        render();
        renderOverview();
    }
}

// ============== Overview Panel ==============

function renderOverview() {
    if (!currentImage) return;

    // Calculate overview size maintaining aspect ratio
    const aspectRatio = currentImage.width / currentImage.height;
    let overviewWidth, overviewHeight;

    if (aspectRatio > OVERVIEW_MAX_WIDTH / OVERVIEW_MAX_HEIGHT) {
        overviewWidth = OVERVIEW_MAX_WIDTH;
        overviewHeight = OVERVIEW_MAX_WIDTH / aspectRatio;
    } else {
        overviewHeight = OVERVIEW_MAX_HEIGHT;
        overviewWidth = OVERVIEW_MAX_HEIGHT * aspectRatio;
    }

    overviewCanvas.width = overviewWidth;
    overviewCanvas.height = overviewHeight;

    // Draw scaled image
    overviewCtx.drawImage(currentImage, 0, 0, overviewWidth, overviewHeight);

    // Draw markers on overview
    const scale = overviewWidth / currentImage.width;
    markers.forEach(marker => {
        const x = marker.x * scale;
        const y = marker.y * scale;
        overviewCtx.fillStyle = marker.marker_class === MarkerClass.POSITIVE ? '#4caf50' : '#f44336';
        overviewCtx.beginPath();
        overviewCtx.arc(x, y, 2, 0, Math.PI * 2);
        overviewCtx.fill();
    });

    // Draw bounding box on overview
    if (showGuide && markers.length >= 2) {
        const padding = 8;
        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;

        markers.forEach(m => {
            minX = Math.min(minX, m.x);
            minY = Math.min(minY, m.y);
            maxX = Math.max(maxX, m.x);
            maxY = Math.max(maxY, m.y);
        });

        minX = Math.max(0, minX - padding) * scale;
        minY = Math.max(0, minY - padding) * scale;
        maxX = Math.min(currentImage.width, maxX + padding) * scale;
        maxY = Math.min(currentImage.height, maxY + padding) * scale;

        overviewCtx.strokeStyle = '#ffd700';
        overviewCtx.lineWidth = 1;
        overviewCtx.setLineDash([4, 2]);
        overviewCtx.strokeRect(minX, minY, maxX - minX, maxY - minY);
        overviewCtx.setLineDash([]);
    }

    updateOverviewViewport();
}

function updateOverviewViewport() {
    if (!currentImage) return;

    const container = elements.canvasViewport;
    const viewWidth = container.clientWidth || elements.canvasContainer.clientWidth;
    const viewHeight = container.clientHeight || elements.canvasContainer.clientHeight;

    const imgWidth = currentImage.width * zoom;
    const imgHeight = currentImage.height * zoom;

    // Calculate visible portion in image coordinates
    const visibleLeft = Math.max(0, (imgWidth / 2 - viewWidth / 2 - panX) / zoom);
    const visibleTop = Math.max(0, (imgHeight / 2 - viewHeight / 2 - panY) / zoom);
    const visibleWidth = Math.min(currentImage.width, viewWidth / zoom);
    const visibleHeight = Math.min(currentImage.height, viewHeight / zoom);

    // Convert to overview coordinates
    const scale = overviewCanvas.width / currentImage.width;
    const vpLeft = visibleLeft * scale + 4; // +4 for panel padding
    const vpTop = visibleTop * scale + 4;
    const vpWidth = visibleWidth * scale;
    const vpHeight = visibleHeight * scale;

    overviewViewport.style.left = vpLeft + 'px';
    overviewViewport.style.top = vpTop + 'px';
    overviewViewport.style.width = vpWidth + 'px';
    overviewViewport.style.height = vpHeight + 'px';

    // Hide viewport indicator if showing entire image
    const showingAll = vpWidth >= overviewCanvas.width - 1 && vpHeight >= overviewCanvas.height - 1;
    overviewViewport.style.display = showingAll ? 'none' : 'block';
}

function handleOverviewMouseDown(e) {
    if (!currentImage) return;
    e.preventDefault();

    isOverviewDragging = true;
    panToOverviewPoint(e);
}

function handleOverviewMouseMove(e) {
    if (!isOverviewDragging) return;

    // Use requestAnimationFrame for smooth updates
    if (overviewAnimationFrame) {
        cancelAnimationFrame(overviewAnimationFrame);
    }

    overviewAnimationFrame = requestAnimationFrame(() => {
        panToOverviewPoint(e);
    });
}

function handleOverviewMouseUp() {
    isOverviewDragging = false;
    if (overviewAnimationFrame) {
        cancelAnimationFrame(overviewAnimationFrame);
        overviewAnimationFrame = null;
    }
}

function panToOverviewPoint(e) {
    if (!currentImage) return;

    const rect = overviewCanvas.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    // Clamp to overview bounds
    const clampedX = Math.max(0, Math.min(overviewCanvas.width, clickX));
    const clampedY = Math.max(0, Math.min(overviewCanvas.height, clickY));

    // Convert to image coordinates
    const scale = currentImage.width / overviewCanvas.width;
    const imgX = clampedX * scale;
    const imgY = clampedY * scale;

    // Pan to center on clicked point
    const container = elements.canvasViewport;
    const viewWidth = container.clientWidth || elements.canvasContainer.clientWidth;
    const viewHeight = container.clientHeight || elements.canvasContainer.clientHeight;

    panX = (currentImage.width / 2 - imgX) * zoom;
    panY = (currentImage.height / 2 - imgY) * zoom;

    constrainPan();
    render();
    updateOverviewViewport();
}

// ============== Mode Handling ==============

function setMode(mode) {
    // Clear eraser state when leaving eraser mode
    if (currentMode === Mode.ERASER && mode !== Mode.ERASER) {
        eraserImagePos = null;
        isErasing = false;
    }
    currentMode = mode;
    eel.set_mode(mode);
    updateModeUI();
}

function updateModeUI() {
    // Update button states
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', parseInt(btn.dataset.mode) === currentMode);
    });

    // Update canvas cursor class (preserve other classes)
    canvas.classList.remove('mode-positive', 'mode-negative', 'mode-select', 'mode-eraser', 'mode-pan', 'mode-roi');
    switch (currentMode) {
        case Mode.POSITIVE:
            canvas.classList.add('mode-positive');
            elements.statusMode.textContent = 'Mode: Positive';
            break;
        case Mode.NEGATIVE:
            canvas.classList.add('mode-negative');
            elements.statusMode.textContent = 'Mode: Negative';
            break;
        case Mode.SELECT:
            canvas.classList.add('mode-select');
            elements.statusMode.textContent = 'Mode: Select';
            break;
        case Mode.ERASER:
            canvas.classList.add('mode-eraser');
            elements.statusMode.textContent = 'Mode: Eraser';
            break;
        case Mode.PAN:
            canvas.classList.add('mode-pan');
            elements.statusMode.textContent = 'Mode: Pan';
            break;
        case Mode.ROI:
            canvas.classList.add('mode-roi');
            elements.statusMode.textContent = 'Mode: ROI';
            break;
    }
}

// ============== Context Menu ==============

function handleContextMenu(e) {
    e.preventDefault();

    if (!isProjectLoaded || !currentImage) return;

    const menu = elements.contextMenu;
    menu.style.left = e.clientX + 'px';
    menu.style.top = e.clientY + 'px';
    menu.classList.remove('hidden');

    // Highlight current mode
    document.querySelector('[data-action="mode-positive"]').classList.toggle('active', currentMode === Mode.POSITIVE);
    document.querySelector('[data-action="mode-negative"]').classList.toggle('active', currentMode === Mode.NEGATIVE);
    document.querySelector('[data-action="mode-pan"]').classList.toggle('active', currentMode === Mode.PAN);

    // Enable/disable selection-dependent actions
    const hasSelected = markers.some(m => m.selected);
    document.querySelector('[data-action="delete-selected"]').classList.toggle('disabled', !hasSelected);
    document.querySelector('[data-action="change-positive"]').classList.toggle('disabled', !hasSelected);
    document.querySelector('[data-action="change-negative"]').classList.toggle('disabled', !hasSelected);
}

function hideContextMenu() {
    elements.contextMenu.classList.add('hidden');
}

async function handleContextMenuAction(e) {
    // Find the menu item (might click on child element like icon)
    const menuItem = e.target.closest('.context-menu-item');
    if (!menuItem) return;

    const action = menuItem.dataset.action;
    if (!action || menuItem.classList.contains('disabled')) return;

    hideContextMenu();

    switch (action) {
        case 'mode-positive':
            setMode(Mode.POSITIVE);
            break;

        case 'mode-negative':
            setMode(Mode.NEGATIVE);
            break;

        case 'mode-pan':
            setMode(Mode.PAN);
            break;

        case 'delete-selected':
            const deleteResult = await eel.delete_selected_markers()();
            if (deleteResult) {
                markers = deleteResult.markers;
                updateCounts(deleteResult.positive_count, deleteResult.negative_count);
                updateUndoRedoButtons(deleteResult.can_undo, deleteResult.can_redo);
                if (deleteResult.removed > 0) markUnsavedChanges();
                render();
                renderOverview();
            }
            break;

        case 'select-all':
            const selectResult = await eel.select_all_markers()();
            if (selectResult) {
                markers = selectResult.markers;
                render();
            }
            break;

        case 'deselect-all':
            const deselectResult = await eel.deselect_all()();
            if (deselectResult) {
                markers = deselectResult.markers;
                render();
            }
            break;

        case 'invert-selection':
            const invertResult = await eel.invert_selection()();
            if (invertResult) {
                markers = invertResult.markers;
                render();
            }
            break;

        case 'change-positive':
            const posResult = await eel.convert_selected_markers(MarkerClass.POSITIVE)();
            if (posResult) {
                markers = posResult.markers;
                updateCounts(posResult.positive_count, posResult.negative_count);
                updateUndoRedoButtons(posResult.can_undo, posResult.can_redo);
                markUnsavedChanges();
                render();
                renderOverview();
            }
            break;

        case 'change-negative':
            const negResult = await eel.convert_selected_markers(MarkerClass.NEGATIVE)();
            if (negResult) {
                markers = negResult.markers;
                updateCounts(negResult.positive_count, negResult.negative_count);
                updateUndoRedoButtons(negResult.can_undo, negResult.can_redo);
                markUnsavedChanges();
                render();
                renderOverview();
            }
            break;
    }
}

// ============== Canvas Event Handlers ==============

function getImageCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;

    // Convert screen coordinates to image coordinates
    // Account for zoom and pan
    const container = elements.canvasViewport;
    const viewWidth = container.clientWidth;
    const viewHeight = container.clientHeight;
    const imgWidth = currentImage.width * zoom;
    const imgHeight = currentImage.height * zoom;

    // Canvas position within container
    const canvasLeft = (viewWidth - imgWidth) / 2 + panX;
    const canvasTop = (viewHeight - imgHeight) / 2 + panY;

    // Image coordinates
    const imgX = (e.clientX - rect.left) / zoom;
    const imgY = (e.clientY - rect.top) / zoom;

    return {
        x: Math.round(imgX),
        y: Math.round(imgY)
    };
}

function getClampedImageCoords(e) {
    // Get image coords, clamped to image bounds (for selection that continues outside viewport)
    const rect = canvas.getBoundingClientRect();
    const imgX = (e.clientX - rect.left) / zoom;
    const imgY = (e.clientY - rect.top) / zoom;
    const maxX = currentImage ? currentImage.width : 0;
    const maxY = currentImage ? currentImage.height : 0;
    return {
        x: Math.round(Math.max(0, Math.min(imgX, maxX))),
        y: Math.round(Math.max(0, Math.min(imgY, maxY)))
    };
}

function findMarkerAtPosition(x, y, hitRadius = MARKER_RADIUS + 2) {
    // Find marker within hit radius of position
    for (let i = markers.length - 1; i >= 0; i--) {
        const m = markers[i];
        const dist = Math.sqrt(Math.pow(m.x - x, 2) + Math.pow(m.y - y, 2));
        if (dist <= hitRadius) {
            return i;
        }
    }
    return -1;
}

async function handleCanvasMouseDown(e) {
    if (!isProjectLoaded || !currentImage) return;

    // If context menu was just closed by this click, don't process further
    if (contextMenuJustClosed) return;

    hideContextMenu();

    const coords = getImageCoords(e);
    const isLeftClick = e.button === 0;
    const isRightClick = e.button === 2;
    const isMiddleClick = e.button === 1;

    // Middle-click, pan mode, or Space+click for panning
    if (isMiddleClick || (isLeftClick && currentMode === Mode.PAN)) {
        isPanning = true;
        panStartX = e.clientX;
        panStartY = e.clientY;
        lastPanX = panX;
        lastPanY = panY;
        canvas.classList.add('panning');
        return;
    }

    if (currentMode === Mode.POSITIVE || currentMode === Mode.NEGATIVE) {
        // Only left-click adds markers; right-click shows context menu
        if (isLeftClick) {
            const markerClass = currentMode === Mode.POSITIVE ? MarkerClass.POSITIVE : MarkerClass.NEGATIVE;
            const result = await eel.add_marker(coords.x, coords.y, markerClass)();
            if (result) {
                markers = result.markers;
                updateCounts(result.positive_count, result.negative_count);
                updateUndoRedoButtons(result.can_undo, result.can_redo);
                markUnsavedChanges();
                render();
                renderOverview();
            }
        }
        // Right-click is handled by handleContextMenu
    } else if (currentMode === Mode.SELECT) {
        if (isLeftClick) {
            selectionAdditive = e.shiftKey;
            selectionClickStart = { x: coords.x, y: coords.y };
            isDragging = true;
            dragStart = coords;
            dragEnd = coords;
            if (selectionType === 'lasso') {
                lassoPoints = [coords];
            }
        }
    } else if (currentMode === Mode.ERASER) {
        if (isLeftClick) {
            isErasing = true;
            eraserHistorySaved = false;
            eraserImagePos = { x: coords.x, y: coords.y };
            // Delete markers at initial click position (non-blocking)
            eraseAtPosition(coords.x, coords.y);
        }
    } else if (currentMode === Mode.ROI) {
        if (isLeftClick) {
            isDrawingROI = true;
            roiStart = coords;
            dragEnd = coords;
        }
    }
}

function handleCanvasMouseMove(e) {
    if (isPanning) {
        const dx = e.clientX - panStartX;
        const dy = e.clientY - panStartY;
        panX = lastPanX + dx;
        panY = lastPanY + dy;
        constrainPan();
        render();
        updateOverviewViewport();
        return;
    }

    // Handle ROI drawing
    if (isDrawingROI) {
        const coords = getImageCoords(e);
        dragEnd = coords;
        render();
        drawROIPreviewOnOverlay(coords);
        return;
    }

    // Handle eraser cursor and continuous erasing
    if (currentMode === Mode.ERASER) {
        const coords = getImageCoords(e);
        eraserImagePos = { x: coords.x, y: coords.y };

        if (isErasing && !eraserPending) {
            eraseAtPosition(coords.x, coords.y);
        }
        // Always render to update cursor
        render();
        return;
    }

    if (!isDragging) return;

    const coords = getImageCoords(e);
    dragEnd = coords;

    if (selectionType === 'lasso' && currentMode === Mode.SELECT) {
        // Add point to lasso path (throttle to avoid too many points)
        const lastPoint = lassoPoints[lassoPoints.length - 1];
        const dx = coords.x - lastPoint.x;
        const dy = coords.y - lastPoint.y;
        if (dx * dx + dy * dy > 16) { // Min 4px distance between points
            lassoPoints.push(coords);
        }
    }

    render();

    // Draw selection on overlay (crisp at any zoom)
    if (currentMode === Mode.SELECT) {
        drawSelectionOnOverlay(coords);
    }
}

function drawROIPreviewOnOverlay(currentCoords) {
    if (!currentImage) return;

    const vp = elements.canvasViewport;
    const viewWidth = vp.clientWidth || elements.canvasContainer.clientWidth;
    const viewHeight = vp.clientHeight || elements.canvasContainer.clientHeight;
    const imgWidth = currentImage.width * zoom;
    const imgHeight = currentImage.height * zoom;
    const imgLeft = (viewWidth - imgWidth) / 2 + panX;
    const imgTop = (viewHeight - imgHeight) / 2 + panY;

    // Convert to screen coords
    const x1 = imgLeft + roiStart.x * zoom;
    const y1 = imgTop + roiStart.y * zoom;
    const x2 = imgLeft + currentCoords.x * zoom;
    const y2 = imgTop + currentCoords.y * zoom;

    const x = Math.min(x1, x2);
    const y = Math.min(y1, y2);
    const w = Math.abs(x2 - x1);
    const h = Math.abs(y2 - y1);

    overlayCtx.strokeStyle = '#00bfff';
    overlayCtx.lineWidth = 2;
    overlayCtx.setLineDash([5, 5]);
    overlayCtx.strokeRect(x, y, w, h);
    overlayCtx.setLineDash([]);

    overlayCtx.fillStyle = 'rgba(0, 191, 255, 0.1)';
    overlayCtx.fillRect(x, y, w, h);
}

function drawSelectionOnOverlay(currentCoords) {
    if (!currentImage) return;

    // Get screen transform values
    const vp = elements.canvasViewport;
    const viewWidth = vp.clientWidth || elements.canvasContainer.clientWidth;
    const viewHeight = vp.clientHeight || elements.canvasContainer.clientHeight;
    const imgWidth = currentImage.width * zoom;
    const imgHeight = currentImage.height * zoom;
    const imgLeft = (viewWidth - imgWidth) / 2 + panX;
    const imgTop = (viewHeight - imgHeight) / 2 + panY;

    // Convert image coords to screen coords
    const toScreen = (imgX, imgY) => ({
        x: imgLeft + imgX * zoom,
        y: imgTop + imgY * zoom
    });

    if (selectionType === 'rect') {
        const start = toScreen(dragStart.x, dragStart.y);
        const end = toScreen(dragEnd.x, dragEnd.y);
        const x = Math.min(start.x, end.x);
        const y = Math.min(start.y, end.y);
        const w = Math.abs(end.x - start.x);
        const h = Math.abs(end.y - start.y);

        overlayCtx.strokeStyle = '#0078d4';
        overlayCtx.lineWidth = 2;
        overlayCtx.setLineDash([5, 5]);
        overlayCtx.strokeRect(x, y, w, h);
        overlayCtx.setLineDash([]);
    } else if (selectionType === 'lasso' && lassoPoints.length > 1) {
        overlayCtx.strokeStyle = '#0078d4';
        overlayCtx.lineWidth = 2;
        overlayCtx.setLineDash([5, 5]);
        overlayCtx.beginPath();

        const first = toScreen(lassoPoints[0].x, lassoPoints[0].y);
        overlayCtx.moveTo(first.x, first.y);

        for (let i = 1; i < lassoPoints.length; i++) {
            const pt = toScreen(lassoPoints[i].x, lassoPoints[i].y);
            overlayCtx.lineTo(pt.x, pt.y);
        }

        const current = toScreen(currentCoords.x, currentCoords.y);
        overlayCtx.lineTo(current.x, current.y);
        overlayCtx.closePath();
        overlayCtx.stroke();
        overlayCtx.setLineDash([]);

        overlayCtx.fillStyle = 'rgba(0, 120, 212, 0.1)';
        overlayCtx.fill();
    }
}

// ============== Eraser Functions ==============

function eraseAtPosition(imgX, imgY) {
    // Prevent overlapping calls
    if (eraserPending) return;
    eraserPending = true;

    // Save history only once per stroke
    const saveHistory = !eraserHistorySaved;
    if (saveHistory) {
        eraserHistorySaved = true;
    }

    // Fire-and-forget for smooth dragging
    eel.delete_markers_in_radius(imgX, imgY, eraserRadius, saveHistory)().then(result => {
        eraserPending = false;
        if (result && result.deleted_count > 0) {
            markers = result.markers;
            updateCounts(result.positive_count, result.negative_count);
            updateUndoRedoButtons(result.can_undo, result.can_redo);
            markUnsavedChanges();
            render();
        }
    }).catch(() => {
        eraserPending = false;
    });
}

function drawEraserCursorOnOverlay(imgLeft, imgTop) {
    // Called from renderOverlayMarkers, draws on existing overlay without clearing
    if (!eraserImagePos || currentMode !== Mode.ERASER) return;

    // Convert image coordinates to screen coordinates (same as markers)
    const screenRadius = eraserRadius * zoom;
    const x = imgLeft + eraserImagePos.x * zoom;
    const y = imgTop + eraserImagePos.y * zoom;

    // Draw eraser circle
    overlayCtx.beginPath();
    overlayCtx.arc(x, y, screenRadius, 0, Math.PI * 2);
    overlayCtx.strokeStyle = '#ff4444';
    overlayCtx.lineWidth = 2;
    overlayCtx.stroke();

    // Draw subtle fill
    overlayCtx.fillStyle = 'rgba(255, 68, 68, 0.15)';
    overlayCtx.fill();

    // Draw crosshair at center
    overlayCtx.beginPath();
    overlayCtx.moveTo(x - 5, y);
    overlayCtx.lineTo(x + 5, y);
    overlayCtx.moveTo(x, y - 5);
    overlayCtx.lineTo(x, y + 5);
    overlayCtx.strokeStyle = '#ff4444';
    overlayCtx.lineWidth = 1;
    overlayCtx.stroke();
}

async function handleCanvasMouseUp(e) {
    if (isPanning) {
        isPanning = false;
        canvas.classList.remove('panning');
        return;
    }

    if (!isProjectLoaded) return;

    // Handle ROI drawing completion
    if (currentMode === Mode.ROI && isDrawingROI) {
        const coords = getImageCoords(e);
        const x = Math.min(roiStart.x, coords.x);
        const y = Math.min(roiStart.y, coords.y);
        const w = Math.abs(coords.x - roiStart.x);
        const h = Math.abs(coords.y - roiStart.y);

        isDrawingROI = false;

        if (w > 20 && h > 20) {
            handleSetROI(x, y, w, h);
        } else {
            render(); // Clear preview
        }
        return;
    }

    // Handle eraser stroke end
    if (currentMode === Mode.ERASER && isErasing) {
        isErasing = false;
        eraserHistorySaved = false;
        renderOverview();
        return;
    }

    if (currentMode === Mode.SELECT && isDragging) {
        isDragging = false;
        const coords = getImageCoords(e);
        dragEnd = coords;

        // Check if this was a single click (minimal movement)
        const dx = Math.abs(coords.x - selectionClickStart.x);
        const dy = Math.abs(coords.y - selectionClickStart.y);
        const isSingleClick = dx < 3 && dy < 3;

        if (isSingleClick) {
            // Single click - select marker under cursor
            const markerIdx = findMarkerAtPosition(coords.x, coords.y);
            if (markerIdx >= 0) {
                await selectMarkerAtIndex(markerIdx, selectionAdditive);
            } else if (!selectionAdditive) {
                // Click on empty space without shift - deselect all
                await deselectAllMarkers();
            }
        } else if (selectionType === 'lasso' && lassoPoints.length > 2) {
            // Close the lasso and select markers inside polygon
            lassoPoints.push(coords);
            await selectMarkersInPolygon(lassoPoints, selectionAdditive);
            lassoPoints = [];
        } else if (selectionType === 'rect') {
            const x = Math.min(dragStart.x, dragEnd.x);
            const y = Math.min(dragStart.y, dragEnd.y);
            const w = Math.abs(dragEnd.x - dragStart.x);
            const h = Math.abs(dragEnd.y - dragStart.y);
            if (w > 2 || h > 2) {
                await selectMarkersInRect(x, y, w, h, selectionAdditive);
            }
        }

        selectionClickStart = null;
        render();
    }
}

// Point-in-polygon test using ray casting algorithm
function isPointInPolygon(x, y, polygon) {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const xi = polygon[i].x, yi = polygon[i].y;
        const xj = polygon[j].x, yj = polygon[j].y;

        if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
            inside = !inside;
        }
    }
    return inside;
}

async function selectMarkersInRect(x, y, width, height, additive = false) {
    const result = await eel.select_markers_in_rect(x, y, width, height, additive)();
    if (result) {
        markers = result.markers;
        render();
    }
}

async function selectMarkersInPolygon(polygon, additive = false) {
    const result = await eel.select_markers_in_polygon(polygon, additive)();
    if (result) {
        markers = result.markers;
        render();
    }
}

async function selectMarkerAtIndex(index, additive = false) {
    const result = await eel.select_marker_at_index(index, additive)();
    if (result) {
        markers = result.markers;
        render();
    }
}

async function deselectAllMarkers() {
    const result = await eel.deselect_all()();
    if (result) {
        markers = result.markers;
        render();
    }
}

function handleCanvasMouseLeave(e) {
    // Don't cancel selection or panning - let document handlers continue tracking
    // Only cancel eraser (we don't want eraser to work outside canvas)
    if (isErasing) {
        isErasing = false;
        eraserHistorySaved = false;
        renderOverview();
    }
    // Clear eraser cursor
    if (currentMode === Mode.ERASER) {
        eraserImagePos = null;
        render();
    }
}

function handleDocumentMouseMove(e) {
    // Continue selection/pan even when outside canvas (only if not already on canvas)
    if (e.target === canvas) return;

    if (isDragging && currentMode === Mode.SELECT) {
        const coords = getClampedImageCoords(e);
        dragEnd = coords;

        if (selectionType === 'lasso') {
            const lastPoint = lassoPoints[lassoPoints.length - 1];
            const dist = Math.sqrt(Math.pow(coords.x - lastPoint.x, 2) +
                                   Math.pow(coords.y - lastPoint.y, 2));
            if (dist > 3) {
                lassoPoints.push(coords);
            }
        }
        render();
    }

    if (isPanning) {
        const dx = e.clientX - panStartX;
        const dy = e.clientY - panStartY;
        panX = lastPanX + dx;
        panY = lastPanY + dy;
        constrainPan();
        render();
        updateOverviewViewport();
    }
}

async function handleDocumentMouseUp(e) {
    // Finish selection/pan even when outside canvas (only if not already on canvas)
    if (e.target === canvas) return;

    if (isPanning) {
        isPanning = false;
        canvas.classList.remove('panning');
    }

    if (isDragging && currentMode === Mode.SELECT) {
        isDragging = false;
        const coords = getClampedImageCoords(e);

        if (selectionType === 'lasso' && lassoPoints.length > 2) {
            lassoPoints.push(coords);
            await selectMarkersInPolygon(lassoPoints, selectionAdditive);
            lassoPoints = [];
        } else if (selectionType === 'rect') {
            const x = Math.min(dragStart.x, dragEnd.x);
            const y = Math.min(dragStart.y, dragEnd.y);
            const w = Math.abs(dragEnd.x - dragStart.x);
            const h = Math.abs(dragEnd.y - dragStart.y);
            if (w > 2 || h > 2) {
                await selectMarkersInRect(x, y, w, h, selectionAdditive);
            }
        }
        selectionClickStart = null;
        render();
    }
}

// ============== Keyboard Handlers ==============

async function handleKeyDown(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    // Ctrl/Cmd shortcuts
    if (e.ctrlKey || e.metaKey) {
        switch (e.key.toLowerCase()) {
            case 'z':
                e.preventDefault();
                if (e.shiftKey) {
                    handleRedo();
                } else {
                    handleUndo();
                }
                return;
            case 'y':
                e.preventDefault();
                handleRedo();
                return;
            case 'a':
                e.preventDefault();
                if (isProjectLoaded) {
                    const result = await eel.select_all_markers()();
                    if (result) {
                        markers = result.markers;
                        render();
                    }
                }
                return;
            case 'i':
                e.preventDefault();
                if (isProjectLoaded) {
                    const result = await eel.invert_selection()();
                    if (result) {
                        markers = result.markers;
                        render();
                    }
                }
                return;
        }
    }

    switch (e.key.toLowerCase()) {
        case 'p':
            setMode(Mode.POSITIVE);
            break;
        case 'n':
            setMode(Mode.NEGATIVE);
            break;
        case 's':
            setMode(Mode.SELECT);
            break;
        case 'e':
            setMode(Mode.ERASER);
            break;
        case 'h':
            setMode(Mode.PAN);
            break;
        case 'r':
            setMode(Mode.ROI);
            break;
        case 'g':
            toggleGuide();
            break;
        case 'd':
            if (isProjectLoaded && detectionAvailable) {
                handleDetect();
            }
            break;
        case 'f':
            if (isProjectLoaded) zoomToFit();
            break;
        case '1':
            if (isProjectLoaded) setZoom(1);
            break;
        case '=':
        case '+':
            if (isProjectLoaded) setZoom(zoom + ZOOM_STEP);
            break;
        case '-':
            if (isProjectLoaded) setZoom(zoom - ZOOM_STEP);
            break;
        case 'delete':
        case 'backspace':
            if (isProjectLoaded) {
                e.preventDefault();
                const result = await eel.delete_selected_markers()();
                if (result) {
                    markers = result.markers;
                    updateCounts(result.positive_count, result.negative_count);
                    updateUndoRedoButtons(result.can_undo, result.can_redo);
                    if (result.removed > 0) markUnsavedChanges();
                    render();
                    renderOverview();
                }
            }
            break;
        case 'arrowleft':
            e.preventDefault();
            handlePrevImage();
            break;
        case 'arrowright':
            e.preventDefault();
            handleNextImage();
            break;
        case 'escape':
            hideContextMenu();
            if (isProjectLoaded) {
                const result = await eel.deselect_all()();
                if (result) {
                    markers = result.markers;
                    render();
                }
            }
            break;
        case 'x':
            // Hold X for temporary eraser
            if (!eraserHeldMode && currentMode !== Mode.ERASER) {
                eraserHeldMode = currentMode;
                setMode(Mode.ERASER);
            }
            break;
    }
}

function handleKeyUp(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    switch (e.key.toLowerCase()) {
        case 'x':
            // Release X to restore previous mode
            if (eraserHeldMode !== null) {
                setMode(eraserHeldMode);
                eraserHeldMode = null;
            }
            break;
    }
}

// ============== Rendering ==============

function render() {
    if (!currentImage) return;

    // Clear main canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw image
    ctx.drawImage(currentImage, 0, 0);

    // Update canvas transform for zoom/pan
    const vp = elements.canvasViewport;
    const viewWidth = vp.clientWidth || elements.canvasContainer.clientWidth;
    const viewHeight = vp.clientHeight || elements.canvasContainer.clientHeight;
    const imgWidth = currentImage.width * zoom;
    const imgHeight = currentImage.height * zoom;

    const left = (viewWidth - imgWidth) / 2 + panX;
    const top = (viewHeight - imgHeight) / 2 + panY;

    canvas.style.transform = `scale(${zoom})`;
    canvas.style.left = left + 'px';
    canvas.style.top = top + 'px';

    // Render markers on overlay canvas (at screen resolution)
    renderOverlayMarkers(viewWidth, viewHeight, left, top);
}

function renderOverlayMarkers(viewWidth, viewHeight, imgLeft, imgTop) {
    // Resize overlay to match viewport (account for device pixel ratio for crisp rendering)
    const dpr = window.devicePixelRatio || 1;
    overlayCanvas.width = viewWidth * dpr;
    overlayCanvas.height = viewHeight * dpr;
    overlayCanvas.style.width = viewWidth + 'px';
    overlayCanvas.style.height = viewHeight + 'px';
    overlayCtx.scale(dpr, dpr);

    // Clear overlay
    overlayCtx.clearRect(0, 0, viewWidth, viewHeight);

    // Draw ROI if present
    drawROIOnOverlay(imgLeft, imgTop);

    // Draw marker bounding box guide on overlay
    drawGuideOnOverlay(imgLeft, imgTop);

    // Draw each marker at screen coordinates
    markers.forEach(marker => {
        // Convert image coordinates to screen coordinates
        const screenX = imgLeft + marker.x * zoom;
        const screenY = imgTop + marker.y * zoom;

        // Dim markers outside ROI if ROI is active
        const inROI = !currentROI || (
            marker.x >= currentROI.x &&
            marker.x <= currentROI.x + currentROI.width &&
            marker.y >= currentROI.y &&
            marker.y <= currentROI.y + currentROI.height
        );

        drawMarkerOnOverlay(marker, screenX, screenY, inROI ? 1.0 : 0.3);
    });

    // Draw eraser cursor on top of everything else
    drawEraserCursorOnOverlay(imgLeft, imgTop);
}

function drawGuideOnOverlay(imgLeft, imgTop) {
    if (!showGuide || markers.length < 2) return;

    // Calculate bounding box with tight padding
    const padding = 8;
    let minX = Infinity, minY = Infinity;
    let maxX = -Infinity, maxY = -Infinity;

    markers.forEach(m => {
        minX = Math.min(minX, m.x);
        minY = Math.min(minY, m.y);
        maxX = Math.max(maxX, m.x);
        maxY = Math.max(maxY, m.y);
    });

    // Add padding and clamp to image bounds
    minX = Math.max(0, minX - padding);
    minY = Math.max(0, minY - padding);
    maxX = Math.min(currentImage.width, maxX + padding);
    maxY = Math.min(currentImage.height, maxY + padding);

    // Convert to screen coordinates
    const screenMinX = imgLeft + minX * zoom;
    const screenMinY = imgTop + minY * zoom;
    const screenMaxX = imgLeft + maxX * zoom;
    const screenMaxY = imgTop + maxY * zoom;
    const width = screenMaxX - screenMinX;
    const height = screenMaxY - screenMinY;

    // Draw dashed rectangle
    overlayCtx.strokeStyle = '#ffd700';
    overlayCtx.lineWidth = 2;
    overlayCtx.setLineDash([6, 3]);
    overlayCtx.strokeRect(screenMinX, screenMinY, width, height);
    overlayCtx.setLineDash([]);

    // Draw corner brackets for emphasis
    const bracketSize = Math.min(15, width / 5, height / 5);
    overlayCtx.strokeStyle = '#ffd700';
    overlayCtx.lineWidth = 2;

    // Top-left bracket
    overlayCtx.beginPath();
    overlayCtx.moveTo(screenMinX, screenMinY + bracketSize);
    overlayCtx.lineTo(screenMinX, screenMinY);
    overlayCtx.lineTo(screenMinX + bracketSize, screenMinY);
    overlayCtx.stroke();

    // Top-right bracket
    overlayCtx.beginPath();
    overlayCtx.moveTo(screenMaxX - bracketSize, screenMinY);
    overlayCtx.lineTo(screenMaxX, screenMinY);
    overlayCtx.lineTo(screenMaxX, screenMinY + bracketSize);
    overlayCtx.stroke();

    // Bottom-left bracket
    overlayCtx.beginPath();
    overlayCtx.moveTo(screenMinX, screenMaxY - bracketSize);
    overlayCtx.lineTo(screenMinX, screenMaxY);
    overlayCtx.lineTo(screenMinX + bracketSize, screenMaxY);
    overlayCtx.stroke();

    // Bottom-right bracket
    overlayCtx.beginPath();
    overlayCtx.moveTo(screenMaxX - bracketSize, screenMaxY);
    overlayCtx.lineTo(screenMaxX, screenMaxY);
    overlayCtx.lineTo(screenMaxX, screenMaxY - bracketSize);
    overlayCtx.stroke();
}

function drawMarkerOnOverlay(marker, screenX, screenY, opacity = 1.0) {
    const r = MARKER_RADIUS;
    const t = MARKER_THICKNESS;

    overlayCtx.globalAlpha = opacity;

    if (marker.selected) {
        overlayCtx.fillStyle = '#ffd700';
        drawCrossOnOverlay(screenX, screenY, r + 3, t + 3);
    }

    overlayCtx.fillStyle = marker.marker_class === MarkerClass.POSITIVE ? '#4caf50' : '#f44336';
    drawCrossOnOverlay(screenX, screenY, r, t);

    overlayCtx.globalAlpha = 1.0;
}

function drawCrossOnOverlay(x, y, l, t) {
    const halfT = t / 2;
    overlayCtx.beginPath();
    overlayCtx.rect(x - halfT, y - l, t, l * 2);
    overlayCtx.rect(x - l, y - halfT, l * 2, t);
    overlayCtx.fill();
}

// ============== UI Updates ==============

function updateProgress(current, total) {
    elements.progress.textContent = `${current} / ${total}`;
}

function updateCounts(positive, negative) {
    elements.countPositive.textContent = positive;
    elements.countNegative.textContent = negative;
    const total = positive + negative;
    elements.countTotal.textContent = total;
    // Calculate proliferation index to 1 decimal place
    if (total > 0) {
        const pi = (positive / total * 100).toFixed(1);
        elements.countPi.textContent = pi + '%';
    } else {
        elements.countPi.textContent = '-';
    }
}

async function updateSummary() {
    const summary = await eel.get_summary()();
    if (summary.total > 0) {
        elements.statusSummary.textContent =
            `PI: ${summary.proliferation_index}% (${summary.positive}/${summary.total})`;
    } else {
        elements.statusSummary.textContent = '';
    }
}

// ============== ROI Functions ==============

async function handleFindHotspot() {
    if (!isProjectLoaded) return;

    const result = await eel.find_hotspot(500)();
    if (result.success) {
        currentROI = result.roi;
        updateCounts(result.positive_count, result.negative_count);
        markUnsavedChanges();
        render();
        renderOverview();
    } else {
        alert(result.message);
    }
}

async function handleClearROI() {
    if (!isProjectLoaded) return;

    const result = await eel.clear_roi()();
    if (result) {
        currentROI = null;
        updateCounts(result.positive_count, result.negative_count);
        render();
        renderOverview();
    }
}

async function handleSetROI(x, y, width, height) {
    if (!isProjectLoaded) return;

    // Ensure positive width/height
    if (width < 0) {
        x += width;
        width = -width;
    }
    if (height < 0) {
        y += height;
        height = -height;
    }

    // Minimum size
    if (width < 20 || height < 20) return;

    const result = await eel.set_roi(x, y, width, height)();
    if (result) {
        currentROI = result.roi;
        updateCounts(result.positive_count, result.negative_count);
        markUnsavedChanges();
        render();
        renderOverview();
    }
}

function drawROIOnOverlay(imgLeft, imgTop) {
    if (!currentROI) return;

    // Convert ROI coordinates to screen coordinates
    const screenX = imgLeft + currentROI.x * zoom;
    const screenY = imgTop + currentROI.y * zoom;
    const screenW = currentROI.width * zoom;
    const screenH = currentROI.height * zoom;

    // Draw ROI rectangle
    overlayCtx.strokeStyle = '#00bfff';
    overlayCtx.lineWidth = 2;
    overlayCtx.setLineDash([]);
    overlayCtx.strokeRect(screenX, screenY, screenW, screenH);

    // Draw semi-transparent fill
    overlayCtx.fillStyle = 'rgba(0, 191, 255, 0.1)';
    overlayCtx.fillRect(screenX, screenY, screenW, screenH);

    // Draw corner handles for resizing
    const handleSize = 8;
    overlayCtx.fillStyle = '#00bfff';

    // Corners
    overlayCtx.fillRect(screenX - handleSize/2, screenY - handleSize/2, handleSize, handleSize);
    overlayCtx.fillRect(screenX + screenW - handleSize/2, screenY - handleSize/2, handleSize, handleSize);
    overlayCtx.fillRect(screenX - handleSize/2, screenY + screenH - handleSize/2, handleSize, handleSize);
    overlayCtx.fillRect(screenX + screenW - handleSize/2, screenY + screenH - handleSize/2, handleSize, handleSize);

    // Label
    overlayCtx.fillStyle = '#00bfff';
    overlayCtx.font = '12px sans-serif';
    overlayCtx.fillText('ROI', screenX + 4, screenY - 4);
}

// ============== Detection Functions ==============

let detectionAvailable = false;

async function initDetection() {
    try {
        const availability = await eel.get_detection_availability()();

        detectionAvailable = availability.available;

        // Populate model dropdown
        const modelSelect = elements.detectModel;
        modelSelect.innerHTML = '';

        if (availability.models.length === 0) {
            const option = document.createElement('option');
            option.value = '';
            option.disabled = true;
            option.selected = true;
            option.textContent = 'No AI models';
            modelSelect.appendChild(option);
            modelSelect.disabled = true;
        } else {
            availability.models.forEach((model, index) => {
                const option = document.createElement('option');
                option.value = model.id;
                option.textContent = model.name;
                if (index === 0) option.selected = true;
                modelSelect.appendChild(option);
            });
            modelSelect.disabled = false;
        }

        // Update button state
        updateDetectionButtonState();

        // Show/hide threshold based on classify mode
        handleClassifyModeChange();

    } catch (error) {
        console.error('Failed to initialize detection:', error);
        detectionAvailable = false;
        elements.detectModel.disabled = true;
        elements.btnDetect.disabled = true;
    }
}

function updateDetectionButtonState() {
    const modelSelected = elements.detectModel.value && elements.detectModel.value !== '';
    elements.btnDetect.disabled = !detectionAvailable || !modelSelected || !isProjectLoaded;
    elements.btnDetectSettings.disabled = !detectionAvailable || !modelSelected;
}

function showDetectionSettings() {
    updateSettingsVisibility();
    elements.detectionSettingsModal.classList.remove('hidden');
}

function hideDetectionSettings() {
    elements.detectionSettingsModal.classList.add('hidden');
}

function updateSettingsVisibility() {
    const model = elements.detectModel.value;
    elements.cellposeSettings.style.display = (model === 'cellpose') ? 'block' : 'none';
    elements.stardistSettings.style.display = (model === 'stardist') ? 'block' : 'none';
    elements.kinetSettings.style.display = (model === 'kinet') ? 'block' : 'none';
}

function resetDetectionSettings() {
    // CellPose
    elements.settingDiameter.value = 0;
    elements.settingCellprob.value = 0;
    elements.settingCellprobValue.textContent = '0.0';
    elements.settingFlow.value = 0.4;
    elements.settingFlowValue.textContent = '0.40';

    // StarDist
    elements.settingProb.value = 0.5;
    elements.settingProbValue.textContent = '0.50';
    elements.settingNms.value = 0.3;
    elements.settingNmsValue.textContent = '0.30';

    // KiNet
    elements.settingKinetThreshold.value = 0.3;
    elements.settingKinetThresholdValue.textContent = '0.30';
    elements.settingKinetDistance.value = 5;
    elements.settingKinetTilesize.value = 1024;
}

function getDetectionSettings() {
    return {
        // CellPose
        diameter: parseInt(elements.settingDiameter.value) || 0,
        cellprob_threshold: parseFloat(elements.settingCellprob.value),
        flow_threshold: parseFloat(elements.settingFlow.value),
        // StarDist
        prob_thresh: parseFloat(elements.settingProb.value),
        nms_thresh: parseFloat(elements.settingNms.value),
        // KiNet
        threshold: parseFloat(elements.settingKinetThreshold.value),
        min_distance: parseInt(elements.settingKinetDistance.value) || 5,
        tile_size: parseInt(elements.settingKinetTilesize.value) || 1024
    };
}

function handleClassifyModeChange() {
    const classifyMode = elements.detectClassify.value;
    // Show threshold slider only in auto mode
    if (classifyMode === 'auto') {
        elements.thresholdContainer.style.display = 'flex';
    } else {
        elements.thresholdContainer.style.display = 'none';
    }
}

function handleThresholdChange() {
    const value = elements.dabThreshold.value;
    elements.thresholdValue.textContent = value + '%';
}

async function handleDetect() {
    if (!isProjectLoaded || !detectionAvailable) return;

    const modelName = elements.detectModel.value;
    const classifyMode = elements.detectClassify.value;
    const threshold = parseInt(elements.dabThreshold.value) / 100;
    const settings = getDetectionSettings();

    if (!modelName) {
        alert('Please select a detection model.');
        return;
    }

    // Show progress modal
    showDetectionProgress();

    try {
        const result = await eel.detect_nuclei(modelName, classifyMode, threshold, settings)();

        hideDetectionProgress();

        if (result.success) {
            markers = result.markers;
            updateCounts(result.positive_count, result.negative_count);
            updateUndoRedoButtons(result.can_undo, result.can_redo);
            markUnsavedChanges();
            render();
            renderOverview();
            updateSummary();
        } else {
            alert('Detection failed: ' + result.message);
        }
    } catch (error) {
        hideDetectionProgress();
        alert('Detection error: ' + error.message);
    }
}

function showDetectionProgress() {
    elements.detectionProgress.classList.remove('hidden');
    elements.detectionProgressBar.style.width = '0%';
    elements.detectionProgressText.textContent = 'Initializing...';
}

function hideDetectionProgress() {
    elements.detectionProgress.classList.add('hidden');
}

// Expose progress callback for Python to call
eel.expose(onDetectionProgress);
function onDetectionProgress(message, progress) {
    elements.detectionProgressText.textContent = message;
    elements.detectionProgressBar.style.width = (progress * 100) + '%';
}
