/**
 * KiNet Trainer - Frontend Application
 * 3-class annotation for Ki-67 IHC images.
 */

// Constants
const Mode = {
    POSITIVE: 0,
    NEGATIVE: 1,
    OTHER: 2,
    SELECT: 3,
    ERASER: 4,
    PAN: 5,
    BRUSH: 6
};

const MarkerClass = {
    POSITIVE: 0,
    NEGATIVE: 1,
    OTHER: 2
};

const MARKER_RADIUS = 6;
const MARKER_THICKNESS = 3;

const MIN_ZOOM = 0.1;
const MAX_ZOOM = 10;
const ZOOM_STEP = 0.25;
const OVERVIEW_MAX_WIDTH = 200;
const OVERVIEW_MAX_HEIGHT = 150;

const ReviewStatus = {
    NOT_STARTED: 0,
    NEEDS_REVIEW: 1,
    REVIEWED: 2
};

const POSITIVE_COLOR = '#4caf50';
const NEGATIVE_COLOR = '#f44336';
const OTHER_COLOR = '#2196f3';

// State
let currentMode = Mode.POSITIVE;
let currentImage = null;
let markers = [];
let isProjectLoaded = false;
let currentReviewStatus = ReviewStatus.NOT_STARTED;
let sidebarFilter = 'all';  // 'all', 'needs-review', 'reviewed'
let detectMode = 'current'; // 'current' or 'all' — set before showing detect modal

// Zoom/Pan state
let zoom = 1;
let panX = 0;
let panY = 0;
let isPanning = false;
let panStartX = 0;
let panStartY = 0;
let lastPanX = 0;
let lastPanY = 0;

// Eraser state
let eraserRadius = 15;
let isErasing = false;
let eraserImagePos = null;
let eraserHistorySaved = false;
let eraserPending = false;

// Brush state (like eraser but changes class instead of deleting)
let brushRadius = 15;
let brushClass = MarkerClass.POSITIVE; // Class to paint markers to
let isBrushing = false;
let brushImagePos = null;
let brushHistorySaved = false;
let brushPending = false;

// Selection state
let isSelecting = false;
let selectionStartX = 0;
let selectionStartY = 0;
let selectionEndX = 0;
let selectionEndY = 0;
let selectionType = 'rect'; // 'rect' or 'lasso'
let lassoPoints = []; // For lasso selection
let selectionAdditive = false; // Shift held = add to selection
let selectionClickStart = null; // Track click position for single-click select

// Overview drag state
let isOverviewDragging = false;
let overviewAnimationFrame = null;

// Track unsaved changes
let hasUnsavedChanges = false;

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
    elements = {
        canvas: document.getElementById('canvas'),
        overlayCanvas: document.getElementById('overlay-canvas'),
        canvasViewport: document.getElementById('canvas-viewport'),
        canvasContainer: document.getElementById('canvas-container'),
        welcomeMessage: document.getElementById('welcome-message'),
        progress: document.getElementById('progress'),
        contextMenu: document.getElementById('context-menu'),
        statusFilename: document.getElementById('status-filename'),
        statusMode: document.getElementById('status-mode'),
        statusSummary: document.getElementById('status-summary'),
        btnFile: document.getElementById('btn-file'),
        fileMenu: document.getElementById('file-menu'),
        btnUndo: document.getElementById('btn-undo'),
        btnRedo: document.getElementById('btn-redo'),
        btnPrev: document.getElementById('btn-prev'),
        btnNext: document.getElementById('btn-next'),
        welcomeNew: document.getElementById('welcome-new'),
        welcomeLoad: document.getElementById('welcome-load'),
        welcomeImport: document.getElementById('welcome-import'),
        overviewPanel: document.getElementById('overview-panel'),
        overviewCanvas: document.getElementById('overview-canvas'),
        overviewViewport: document.getElementById('overview-viewport'),
        zoomControls: document.getElementById('zoom-controls'),
        zoomLevel: document.getElementById('zoom-level'),
        btnZoomIn: document.getElementById('btn-zoom-in'),
        btnZoomOut: document.getElementById('btn-zoom-out'),
        btnZoomFit: document.getElementById('btn-zoom-fit'),
        btnZoom100: document.getElementById('btn-zoom-100'),
        sidebar: document.getElementById('sidebar'),
        sidebarStats: document.getElementById('sidebar-stats'),
        imageList: document.getElementById('image-list'),
        // Review
        btnReviewed: document.getElementById('btn-reviewed'),
        reviewSummary: document.getElementById('review-summary'),
        // About
        aboutModal: document.getElementById('about-modal'),
        aboutVersion: document.getElementById('about-version'),
        aboutClose: document.getElementById('about-close'),
        // Export
        exportModal: document.getElementById('export-modal'),
        exportReviewInfo: document.getElementById('export-review-info'),
        exportValSplit: document.getElementById('export-val-split'),
        exportValSplitValue: document.getElementById('export-val-split-value'),
        exportSigma: document.getElementById('export-sigma'),
        exportReviewedOnly: document.getElementById('export-reviewed-only'),
        exportCancel: document.getElementById('export-cancel'),
        exportConfirm: document.getElementById('export-confirm'),
        // Tile
        tileModal: document.getElementById('tile-modal'),
        tileSize: document.getElementById('tile-size'),
        tileSkipBlank: document.getElementById('tile-skip-blank'),
        tileBlankThreshold: document.getElementById('tile-blank-threshold'),
        tileBlankThresholdValue: document.getElementById('tile-blank-threshold-value'),
        tileCancel: document.getElementById('tile-cancel'),
        tileConfirm: document.getElementById('tile-confirm'),
        // Detect
        detectModal: document.getElementById('detect-modal'),
        detectModalTitle: document.getElementById('detect-modal-title'),
        detectModel: document.getElementById('detect-model'),
        detectThreshold: document.getElementById('detect-threshold'),
        detectThresholdValue: document.getElementById('detect-threshold-value'),
        detectMinDistance: document.getElementById('detect-min-distance'),
        detectCancel: document.getElementById('detect-cancel'),
        detectConfirm: document.getElementById('detect-confirm'),
        detectWarning: document.querySelector('.detect-warning'),
        // Model browser
        modelBrowserModal: document.getElementById('model-browser-modal'),
        modelList: document.getElementById('model-list'),
        modelBrowserClose: document.getElementById('model-browser-close'),
        // Progress
        progressModal: document.getElementById('progress-modal'),
        progressBar: document.getElementById('progress-bar'),
        progressText: document.getElementById('progress-text'),
        // Selection dropdown
        btnSelectDropdown: document.getElementById('btn-select-dropdown'),
        selectMenu: document.getElementById('select-menu'),
        // Eraser dropdown
        btnEraserDropdown: document.getElementById('btn-eraser-dropdown'),
        eraserMenu: document.getElementById('eraser-menu'),
        // Brush dropdown
        btnBrushDropdown: document.getElementById('btn-brush-dropdown'),
        brushMenu: document.getElementById('brush-menu')
    };

    canvas = elements.canvas;
    ctx = canvas.getContext('2d');
    overlayCanvas = elements.overlayCanvas;
    overlayCtx = overlayCanvas.getContext('2d');
    viewport = elements.canvasViewport;
    overviewCanvas = elements.overviewCanvas;
    overviewCtx = overviewCanvas.getContext('2d');
    overviewViewport = elements.overviewViewport;

    bindEvents();
    updateModeUI();
}

function bindEvents() {
    // File dropdown
    elements.btnFile.addEventListener('click', toggleFileMenu);
    elements.fileMenu.addEventListener('click', handleFileMenuAction);

    // Undo/Redo
    elements.btnUndo.addEventListener('click', handleUndo);
    elements.btnRedo.addEventListener('click', handleRedo);

    // Welcome buttons
    elements.welcomeNew.addEventListener('click', handleNewProject);
    elements.welcomeLoad.addEventListener('click', handleLoadProject);
    elements.welcomeImport.addEventListener('click', handleImportProject);

    // Mode buttons
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            setMode(parseInt(btn.dataset.mode));
        });
    });

    // Navigation
    elements.btnPrev.addEventListener('click', handlePrevImage);
    elements.btnNext.addEventListener('click', handleNextImage);

    // Canvas events
    canvas.addEventListener('mousedown', handleCanvasMouseDown);
    canvas.addEventListener('mousemove', handleCanvasMouseMove);
    canvas.addEventListener('mouseup', handleCanvasMouseUp);
    canvas.addEventListener('mouseleave', handleCanvasMouseLeave);
    canvas.addEventListener('contextmenu', handleContextMenu);
    canvas.addEventListener('wheel', handleCanvasWheel, { passive: false });

    // Context menu
    elements.contextMenu.addEventListener('click', handleContextMenuAction);
    document.addEventListener('mousedown', hideContextMenu);

    // Overview panel
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

    // Hide menus on outside click
    document.addEventListener('mousedown', (e) => {
        if (!elements.fileMenu.contains(e.target) && !elements.btnFile.contains(e.target)) {
            hideFileMenu();
        }
        // Hide select menu
        if (!elements.selectMenu.contains(e.target) && !elements.btnSelectDropdown.contains(e.target)) {
            elements.selectMenu.classList.add('hidden');
        }
        // Hide eraser menu
        if (!elements.eraserMenu.contains(e.target) && !elements.btnEraserDropdown.contains(e.target)) {
            elements.eraserMenu.classList.add('hidden');
        }
        // Hide brush menu
        if (!elements.brushMenu.contains(e.target) && !elements.btnBrushDropdown.contains(e.target)) {
            elements.brushMenu.classList.add('hidden');
        }
    }, true);

    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyDown);

    // Window resize
    window.addEventListener('resize', handleResize);

    // Unsaved changes warning
    window.addEventListener('beforeunload', handleBeforeUnload);

    // Review button
    elements.btnReviewed.addEventListener('click', handleMarkReviewed);

    // Sidebar filters
    document.querySelectorAll('.filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            sidebarFilter = btn.dataset.filter;
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            applySidebarFilter();
        });
    });

    // Selection type dropdown
    elements.btnSelectDropdown.addEventListener('click', toggleSelectMenu);
    elements.selectMenu.addEventListener('click', handleSelectMenuAction);

    // Eraser size dropdown
    elements.btnEraserDropdown.addEventListener('click', toggleEraserMenu);
    elements.eraserMenu.addEventListener('click', handleEraserMenuAction);

    // Brush dropdown
    elements.btnBrushDropdown.addEventListener('click', toggleBrushMenu);
    elements.brushMenu.addEventListener('click', handleBrushMenuAction);

    // About modal
    elements.aboutClose.addEventListener('click', () => elements.aboutModal.classList.add('hidden'));
    elements.aboutModal.addEventListener('click', (e) => {
        if (e.target === elements.aboutModal) elements.aboutModal.classList.add('hidden');
    });

    // Export modal
    elements.exportValSplit.addEventListener('input', () => {
        elements.exportValSplitValue.textContent = Math.round(parseFloat(elements.exportValSplit.value) * 100) + '%';
    });
    elements.exportCancel.addEventListener('click', () => elements.exportModal.classList.add('hidden'));
    elements.exportConfirm.addEventListener('click', handleExportConfirm);
    elements.exportModal.addEventListener('click', (e) => {
        if (e.target === elements.exportModal) elements.exportModal.classList.add('hidden');
    });

    // Tile modal
    elements.tileBlankThreshold.addEventListener('input', () => {
        elements.tileBlankThresholdValue.textContent = Math.round(parseFloat(elements.tileBlankThreshold.value) * 100) + '%';
    });
    elements.tileCancel.addEventListener('click', () => elements.tileModal.classList.add('hidden'));
    elements.tileConfirm.addEventListener('click', handleTileConfirm);
    elements.tileModal.addEventListener('click', (e) => {
        if (e.target === elements.tileModal) elements.tileModal.classList.add('hidden');
    });

    // Detect modal
    elements.detectThreshold.addEventListener('input', () => {
        elements.detectThresholdValue.textContent = parseFloat(elements.detectThreshold.value).toFixed(2);
    });
    elements.detectCancel.addEventListener('click', () => elements.detectModal.classList.add('hidden'));
    elements.detectConfirm.addEventListener('click', handleDetectConfirm);
    elements.detectModal.addEventListener('click', (e) => {
        if (e.target === elements.detectModal) elements.detectModal.classList.add('hidden');
    });

    // Model browser
    elements.modelBrowserClose.addEventListener('click', () => elements.modelBrowserModal.classList.add('hidden'));
    elements.modelBrowserModal.addEventListener('click', (e) => {
        if (e.target === elements.modelBrowserModal) elements.modelBrowserModal.classList.add('hidden');
    });
}

function handleBeforeUnload(e) {
    if (hasUnsavedChanges && isProjectLoaded) {
        e.preventDefault();
        e.returnValue = 'You have unsaved changes. Are you sure you want to leave?';
        return e.returnValue;
    }
}

function markUnsavedChanges() { hasUnsavedChanges = true; }
function clearUnsavedChanges() { hasUnsavedChanges = false; }

// ============== Project Management ==============

async function handleNewProject() {
    const result = await eel.new_project()();
    if (result.success) {
        isProjectLoaded = true;
        clearUnsavedChanges();
        hideWelcome();
        await loadCurrentImage();
        updateImageList();
        updateSummary();
    } else if (result.message !== 'No directory selected' &&
               result.message !== 'No project file selected') {
        alert(result.message);
    }
}

async function handleLoadProject() {
    const result = await eel.load_project()();
    if (result.success) {
        isProjectLoaded = true;
        clearUnsavedChanges();
        hideWelcome();
        await loadCurrentImage();
        updateImageList();
        updateSummary();
    } else if (result.message !== 'No project file selected') {
        alert(result.message);
    }
}

async function handleImportProject() {
    const result = await eel.import_kiquant_project()();
    if (result.success) {
        isProjectLoaded = true;
        clearUnsavedChanges();
        hideWelcome();
        await loadCurrentImage();
        updateImageList();
        updateSummary();
        alert(result.message);
    } else if (result.message !== 'No file selected' &&
               result.message !== 'No save location selected') {
        alert(result.message);
    }
}

async function handleSaveProject() {
    const result = await eel.save_project()();
    if (result.success) {
        clearUnsavedChanges();
    } else {
        alert(result.message);
    }
}

function hideWelcome() {
    elements.welcomeMessage.classList.add('hidden');
    viewport.classList.add('loaded');
    elements.overviewPanel.classList.remove('hidden');
    elements.zoomControls.classList.remove('hidden');
}

// ============== File Menu ==============

function toggleFileMenu(e) {
    e.stopPropagation();
    elements.fileMenu.classList.toggle('hidden');
}

function hideFileMenu() {
    elements.fileMenu.classList.add('hidden');
}

// ============== Selection Menu ==============

function toggleSelectMenu(e) {
    e.stopPropagation();
    elements.selectMenu.classList.toggle('hidden');
    elements.eraserMenu.classList.add('hidden');
    updateSelectMenuUI();
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
    updateSelectMenuUI();
    elements.selectMenu.classList.add('hidden');

    // Switch to select mode when changing selection type
    setMode(Mode.SELECT);
}

// ============== Eraser Menu ==============

function toggleEraserMenu(e) {
    e.stopPropagation();
    elements.eraserMenu.classList.toggle('hidden');
    elements.selectMenu.classList.add('hidden');
    updateEraserMenuUI();
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
    updateEraserMenuUI();
    elements.eraserMenu.classList.add('hidden');

    // Switch to eraser mode when changing eraser size
    setMode(Mode.ERASER);
}

// ============== Brush Menu ==============

function toggleBrushMenu(e) {
    e.stopPropagation();
    elements.brushMenu.classList.toggle('hidden');
    elements.selectMenu.classList.add('hidden');
    elements.eraserMenu.classList.add('hidden');
    updateBrushMenuUI();
}

function updateBrushMenuUI() {
    // Update class selection
    document.querySelectorAll('#brush-menu .dropdown-item[data-brush-class]').forEach(item => {
        item.classList.toggle('active', parseInt(item.dataset.brushClass) === brushClass);
    });
    // Update size selection
    document.querySelectorAll('#brush-menu .dropdown-item[data-brush-size]').forEach(item => {
        item.classList.toggle('size-active', parseInt(item.dataset.brushSize) === brushRadius);
    });
}

function handleBrushMenuAction(e) {
    const item = e.target.closest('.dropdown-item');
    if (!item) return;

    if (item.dataset.brushClass !== undefined) {
        brushClass = parseInt(item.dataset.brushClass);
    }
    if (item.dataset.brushSize !== undefined) {
        brushRadius = parseInt(item.dataset.brushSize);
    }

    updateBrushMenuUI();
    elements.brushMenu.classList.add('hidden');

    // Switch to brush mode when changing brush settings
    setMode(Mode.BRUSH);
}

function getBrushColor() {
    switch (brushClass) {
        case MarkerClass.POSITIVE: return POSITIVE_COLOR;
        case MarkerClass.NEGATIVE: return NEGATIVE_COLOR;
        case MarkerClass.OTHER: return OTHER_COLOR;
        default: return '#888';
    }
}

async function handleFileMenuAction(e) {
    const item = e.target.closest('.dropdown-item');
    if (!item) return;

    const action = item.dataset.action;
    hideFileMenu();

    switch (action) {
        case 'new': handleNewProject(); break;
        case 'load': handleLoadProject(); break;
        case 'import': handleImportProject(); break;
        case 'save': handleSaveProject(); break;
        case 'tile': showTileModal(); break;
        case 'detect-current': showDetectModal('current'); break;
        case 'detect-all': showDetectModal('all'); break;
        case 'export': showExportModal(); break;
        case 'evaluate': handleEvaluate(); break;
        case 'model-browser': showModelBrowser(); break;
        case 'about': showAboutModal(); break;
    }
}

// ============== About Modal ==============

async function showAboutModal() {
    const version = await eel.get_version()();
    elements.aboutVersion.textContent = `Version ${version}`;
    elements.aboutModal.classList.remove('hidden');
}

// ============== Export Modal ==============

async function showExportModal() {
    if (!isProjectLoaded) {
        alert('No project loaded');
        return;
    }

    // Show review info
    const summary = await eel.get_review_summary()();
    const reviewedCount = summary.reviewed || 0;
    const needsReview = summary.needs_review || 0;
    const notStarted = summary.not_started || 0;
    elements.exportReviewInfo.textContent =
        `${reviewedCount} reviewed, ${needsReview} need review, ${notStarted} not started`;

    if (reviewedCount < 10 && reviewedCount > 0) {
        elements.exportReviewInfo.textContent += ' (fewer than 10 reviewed images)';
    }

    elements.exportModal.classList.remove('hidden');
}

async function handleExportConfirm() {
    elements.exportModal.classList.add('hidden');

    const valSplit = parseFloat(elements.exportValSplit.value);
    const sigma = parseFloat(elements.exportSigma.value);
    const reviewedOnly = elements.exportReviewedOnly.checked;

    showProgress('Exporting...');
    const result = await eel.export_data(valSplit, sigma, reviewedOnly)();
    hideProgress();

    if (result.success) {
        alert(result.message);
    } else {
        alert('Export failed: ' + result.message);
    }
}

// ============== Evaluate ==============

async function handleEvaluate() {
    if (!isProjectLoaded) {
        alert('No project loaded');
        return;
    }

    showProgress('Evaluating model...');
    const result = await eel.evaluate_model()();
    hideProgress();

    if (result && result.success) {
        let msg = 'Evaluation Results:\n\n';
        if (result.per_class) {
            for (const [cls, metrics] of Object.entries(result.per_class)) {
                msg += `${cls}: P=${metrics.precision.toFixed(3)} R=${metrics.recall.toFixed(3)} F1=${metrics.f1.toFixed(3)}\n`;
            }
        }
        if (result.overall) {
            msg += `\nOverall: P=${result.overall.precision.toFixed(3)} R=${result.overall.recall.toFixed(3)} F1=${result.overall.f1.toFixed(3)}`;
        }
        alert(msg);
    } else if (result) {
        alert(result.message || 'Evaluation failed');
    }
}

// ============== Progress ==============

function showProgress(text) {
    elements.progressText.textContent = text || 'Processing...';
    elements.progressBar.style.width = '0%';
    elements.progressModal.classList.remove('hidden');
}

function hideProgress() {
    elements.progressModal.classList.add('hidden');
}

// Expose progress callbacks for Python
eel.expose(onExportProgress);
function onExportProgress(message, progress) {
    elements.progressText.textContent = message;
    elements.progressBar.style.width = (progress * 100) + '%';
}

eel.expose(onEvalProgress);
function onEvalProgress(message, progress) {
    elements.progressText.textContent = message;
    elements.progressBar.style.width = (progress * 100) + '%';
}

// ============== Undo/Redo ==============

async function handleUndo() {
    if (!isProjectLoaded || isLocked()) return;
    const result = await eel.undo()();
    if (result) {
        markers = result.markers;
        updateCounts(result.positive_count, result.negative_count, result.other_count);
        updateUndoRedoButtons(result.can_undo, result.can_redo);
        markUnsavedChanges();
        render();
        renderOverview();
        updateCurrentImageListItem();
    }
}

async function handleRedo() {
    if (!isProjectLoaded || isLocked()) return;
    const result = await eel.redo()();
    if (result) {
        markers = result.markers;
        updateCounts(result.positive_count, result.negative_count, result.other_count);
        updateUndoRedoButtons(result.can_undo, result.can_redo);
        markUnsavedChanges();
        render();
        renderOverview();
        updateCurrentImageListItem();
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
        canvas.width = data.width;
        canvas.height = data.height;
        markers = data.markers;
        currentReviewStatus = data.review_status || 0;
        updateReviewButton();

        if (isFirstLoad) {
            zoomToFit();
        } else {
            constrainPan();
        }

        render();
        renderOverview();
        updateProgress(data.index + 1, data.total);
        updateCounts(data.positive_count, data.negative_count, data.other_count);
        elements.statusFilename.textContent = data.filename;

        const undoRedoState = await eel.get_undo_redo_state()();
        updateUndoRedoButtons(undoRedoState.can_undo, undoRedoState.can_redo);

        highlightImageListItem(data.index);
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
        currentReviewStatus = data.review_status || 0;
        updateReviewButton();
        constrainPan();
        render();
        renderOverview();
        updateProgress(data.index + 1, data.total);
        updateCounts(data.positive_count, data.negative_count, data.other_count);
        elements.statusFilename.textContent = data.filename;

        const undoRedoState = await eel.get_undo_redo_state()();
        updateUndoRedoButtons(undoRedoState.can_undo, undoRedoState.can_redo);

        highlightImageListItem(data.index);
    };
    currentImage.src = data.image;
}

// ============== Sidebar / Image List ==============

async function updateImageList() {
    const images = await eel.get_image_list()();
    const list = elements.imageList;
    list.innerHTML = '';

    images.forEach((img, index) => {
        const item = document.createElement('div');
        item.className = 'image-list-item';
        item.dataset.index = index;
        item.dataset.reviewStatus = img.review_status || 0;

        // Status dot
        const dot = document.createElement('span');
        dot.className = 'status-dot ' + getReviewStatusClass(img.review_status);
        item.appendChild(dot);

        // Info container
        const info = document.createElement('div');
        info.className = 'image-info';

        const name = document.createElement('div');
        name.className = 'image-name';
        name.textContent = img.filename;

        const counts = document.createElement('div');
        counts.className = 'image-counts';
        counts.innerHTML = `<span class="count-pos">${img.positive}P</span> <span class="count-neg">${img.negative}N</span> <span class="count-other">${img.other}O</span>`;

        info.appendChild(name);
        info.appendChild(counts);
        item.appendChild(info);

        item.addEventListener('click', async () => {
            const data = await eel.go_to_image(index)();
            if (data && !data.error) {
                loadImageData(data);
            }
        });

        list.appendChild(item);
    });

    updateSidebarStats();
    updateReviewSummary();
    applySidebarFilter();
}

function getReviewStatusClass(status) {
    switch (status) {
        case ReviewStatus.NEEDS_REVIEW: return 'needs-review';
        case ReviewStatus.REVIEWED: return 'reviewed';
        default: return 'not-started';
    }
}

function applySidebarFilter() {
    const items = elements.imageList.querySelectorAll('.image-list-item');
    items.forEach(item => {
        const status = parseInt(item.dataset.reviewStatus);
        let show = true;
        if (sidebarFilter === 'needs-review' && status !== ReviewStatus.NEEDS_REVIEW) show = false;
        if (sidebarFilter === 'reviewed' && status !== ReviewStatus.REVIEWED) show = false;
        item.style.display = show ? '' : 'none';
    });
}

async function updateReviewSummary() {
    const summary = await eel.get_review_summary()();
    if (summary.total > 0) {
        elements.reviewSummary.textContent =
            `${summary.reviewed} reviewed / ${summary.total} images (${summary.needs_review} need review)`;
        elements.reviewSummary.classList.add('visible');
    } else {
        elements.reviewSummary.classList.remove('visible');
    }
}

function updateCurrentImageListItem() {
    // Update counts and status dot for current image in sidebar
    const activeItem = document.querySelector('.image-list-item.active');
    if (!activeItem) return;
    const index = parseInt(activeItem.dataset.index);
    if (isNaN(index)) return;

    // Find item by index (might not match children index if filtered)
    const item = elements.imageList.querySelector(`.image-list-item[data-index="${index}"]`);
    if (!item) return;

    const pos = markers.filter(m => m.marker_class === MarkerClass.POSITIVE).length;
    const neg = markers.filter(m => m.marker_class === MarkerClass.NEGATIVE).length;
    const other = markers.filter(m => m.marker_class === MarkerClass.OTHER).length;

    const counts = item.querySelector('.image-counts');
    if (counts) {
        counts.innerHTML = `<span class="count-pos">${pos}P</span> <span class="count-neg">${neg}N</span> <span class="count-other">${other}O</span>`;
    }

    // Update status dot
    const dot = item.querySelector('.status-dot');
    if (dot) {
        dot.className = 'status-dot ' + getReviewStatusClass(currentReviewStatus);
    }
    item.dataset.reviewStatus = currentReviewStatus;
}

function highlightImageListItem(index) {
    document.querySelectorAll('.image-list-item').forEach(item => {
        item.classList.toggle('active', parseInt(item.dataset.index) === index);
    });

    // Scroll into view
    const activeItem = document.querySelector('.image-list-item.active');
    if (activeItem) {
        activeItem.scrollIntoView({ block: 'nearest' });
    }
}

async function updateSidebarStats() {
    const summary = await eel.get_summary()();
    elements.sidebarStats.textContent = `${summary.annotated_images}/${summary.total_images}`;
}

// ============== Zoom/Pan ==============

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
    const containerWidth = (container.clientWidth || elements.canvasContainer.clientWidth) - 40;
    const containerHeight = (container.clientHeight || elements.canvasContainer.clientHeight) - 40;
    const scaleX = containerWidth / currentImage.width;
    const scaleY = containerHeight / currentImage.height;
    zoom = Math.min(scaleX, scaleY, 1);
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

    if (imgWidth <= viewWidth) { panX = 0; }
    else {
        const maxPan = (imgWidth - viewWidth) / 2;
        panX = Math.max(-maxPan, Math.min(maxPan, panX));
    }

    if (imgHeight <= viewHeight) { panY = 0; }
    else {
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

    const container = elements.canvasViewport;
    const rect = container.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    const viewWidth = container.clientWidth;
    const viewHeight = container.clientHeight;

    const imgX = (mouseX - viewWidth / 2 - panX) / zoom + currentImage.width / 2;
    const imgY = (mouseY - viewHeight / 2 - panY) / zoom + currentImage.height / 2;

    const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP;
    const oldZoom = zoom;
    zoom = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, zoom + delta));

    if (zoom !== oldZoom) {
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
    overviewCtx.drawImage(currentImage, 0, 0, overviewWidth, overviewHeight);

    const scale = overviewWidth / currentImage.width;
    markers.forEach(marker => {
        const x = marker.x * scale;
        const y = marker.y * scale;
        overviewCtx.fillStyle = getMarkerColor(marker.marker_class);
        overviewCtx.beginPath();
        overviewCtx.arc(x, y, 2, 0, Math.PI * 2);
        overviewCtx.fill();
    });

    updateOverviewViewport();
}

function updateOverviewViewport() {
    if (!currentImage) return;

    const container = elements.canvasViewport;
    const viewWidth = container.clientWidth || elements.canvasContainer.clientWidth;
    const viewHeight = container.clientHeight || elements.canvasContainer.clientHeight;
    const imgWidth = currentImage.width * zoom;
    const imgHeight = currentImage.height * zoom;

    const visibleLeft = Math.max(0, (imgWidth / 2 - viewWidth / 2 - panX) / zoom);
    const visibleTop = Math.max(0, (imgHeight / 2 - viewHeight / 2 - panY) / zoom);
    const visibleWidth = Math.min(currentImage.width, viewWidth / zoom);
    const visibleHeight = Math.min(currentImage.height, viewHeight / zoom);

    const scale = overviewCanvas.width / currentImage.width;
    const vpLeft = visibleLeft * scale + 4;
    const vpTop = visibleTop * scale + 4;
    const vpWidth = visibleWidth * scale;
    const vpHeight = visibleHeight * scale;

    overviewViewport.style.left = vpLeft + 'px';
    overviewViewport.style.top = vpTop + 'px';
    overviewViewport.style.width = vpWidth + 'px';
    overviewViewport.style.height = vpHeight + 'px';

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
    if (overviewAnimationFrame) cancelAnimationFrame(overviewAnimationFrame);
    overviewAnimationFrame = requestAnimationFrame(() => panToOverviewPoint(e));
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
    const clampedX = Math.max(0, Math.min(overviewCanvas.width, e.clientX - rect.left));
    const clampedY = Math.max(0, Math.min(overviewCanvas.height, e.clientY - rect.top));
    const scale = currentImage.width / overviewCanvas.width;
    const imgX = clampedX * scale;
    const imgY = clampedY * scale;

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
    // Clear brush state when leaving brush mode
    if (currentMode === Mode.BRUSH && mode !== Mode.BRUSH) {
        brushImagePos = null;
        isBrushing = false;
    }
    currentMode = mode;
    eel.set_mode(mode);
    updateModeUI();
}

function updateModeUI() {
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', parseInt(btn.dataset.mode) === currentMode);
    });

    canvas.classList.remove('mode-positive', 'mode-negative', 'mode-other', 'mode-select', 'mode-eraser', 'mode-pan', 'mode-brush');
    const modeNames = ['Positive', 'Negative', 'Other', 'Select', 'Eraser', 'Pan', 'Brush'];
    const modeClasses = ['mode-positive', 'mode-negative', 'mode-other', 'mode-select', 'mode-eraser', 'mode-pan', 'mode-brush'];
    canvas.classList.add(modeClasses[currentMode]);
    elements.statusMode.textContent = 'Mode: ' + modeNames[currentMode];
}

// ============== Canvas Event Handlers ==============

function getImageCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const imgX = (e.clientX - rect.left) / zoom;
    const imgY = (e.clientY - rect.top) / zoom;
    return { x: Math.round(imgX), y: Math.round(imgY) };
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

    const coords = getImageCoords(e);
    const isLeftClick = e.button === 0;
    const isMiddleClick = e.button === 1;

    // Middle-click or pan mode
    if (isMiddleClick || (isLeftClick && currentMode === Mode.PAN)) {
        isPanning = true;
        panStartX = e.clientX;
        panStartY = e.clientY;
        lastPanX = panX;
        lastPanY = panY;
        canvas.classList.add('panning');
        return;
    }

    if (currentMode === Mode.POSITIVE || currentMode === Mode.NEGATIVE || currentMode === Mode.OTHER) {
        if (isLocked()) return; // Prevent editing when reviewed
        if (isLeftClick) {
            const markerClass = currentMode; // Mode values match MarkerClass values
            const result = await eel.add_marker(coords.x, coords.y, markerClass)();
            if (result) {
                markers = result.markers;
                updateCounts(result.positive_count, result.negative_count, result.other_count);
                updateUndoRedoButtons(result.can_undo, result.can_redo);
                markUnsavedChanges();
                render();
                renderOverview();
                updateCurrentImageListItem();
            }
        }
    } else if (currentMode === Mode.SELECT) {
        if (isLeftClick) {
            selectionAdditive = e.shiftKey;
            selectionClickStart = { x: coords.x, y: coords.y };
            isSelecting = true;
            selectionStartX = coords.x;
            selectionStartY = coords.y;
            selectionEndX = coords.x;
            selectionEndY = coords.y;
            if (selectionType === 'lasso') {
                lassoPoints = [coords];
            }
        }
    } else if (currentMode === Mode.ERASER) {
        if (isLocked()) return; // Prevent editing when reviewed
        if (isLeftClick) {
            isErasing = true;
            eraserHistorySaved = false;
            eraserImagePos = { x: coords.x, y: coords.y };
            eraseAtPosition(coords.x, coords.y);
        }
    } else if (currentMode === Mode.BRUSH) {
        if (isLocked()) return; // Prevent editing when reviewed
        if (isLeftClick) {
            isBrushing = true;
            brushHistorySaved = false;
            brushImagePos = { x: coords.x, y: coords.y };
            brushAtPosition(coords.x, coords.y);
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

    if (currentMode === Mode.SELECT && isSelecting) {
        const coords = getImageCoords(e);
        selectionEndX = coords.x;
        selectionEndY = coords.y;

        if (selectionType === 'lasso') {
            // Add point to lasso path (throttle to avoid too many points)
            const lastPoint = lassoPoints[lassoPoints.length - 1];
            const dist = Math.sqrt(Math.pow(coords.x - lastPoint.x, 2) +
                                   Math.pow(coords.y - lastPoint.y, 2));
            if (dist > 3) {
                lassoPoints.push(coords);
            }
        }

        render();
        return;
    }

    if (currentMode === Mode.ERASER) {
        const coords = getImageCoords(e);
        eraserImagePos = { x: coords.x, y: coords.y };
        if (isErasing && !eraserPending) {
            eraseAtPosition(coords.x, coords.y);
        }
        render();
        return;
    }

    if (currentMode === Mode.BRUSH) {
        const coords = getImageCoords(e);
        brushImagePos = { x: coords.x, y: coords.y };
        if (isBrushing && !brushPending) {
            brushAtPosition(coords.x, coords.y);
        }
        render();
        return;
    }
}

function handleCanvasMouseUp(e) {
    if (isPanning) {
        isPanning = false;
        canvas.classList.remove('panning');
        return;
    }

    if (currentMode === Mode.SELECT && isSelecting) {
        isSelecting = false;
        const coords = getImageCoords(e);

        // Check if this was a single click (minimal movement)
        const dx = Math.abs(coords.x - selectionClickStart.x);
        const dy = Math.abs(coords.y - selectionClickStart.y);
        const isSingleClick = dx < 3 && dy < 3;

        if (isSingleClick) {
            // Single click - select marker under cursor
            const markerIdx = findMarkerAtPosition(coords.x, coords.y);
            if (markerIdx >= 0) {
                selectMarkerAtIndex(markerIdx, selectionAdditive);
            } else if (!selectionAdditive) {
                // Click on empty space without shift - deselect all
                deselectAllMarkers();
            }
        } else if (selectionType === 'lasso' && lassoPoints.length > 2) {
            // Close the lasso and select markers inside polygon
            lassoPoints.push(coords);
            selectMarkersInPolygon(lassoPoints, selectionAdditive);
            lassoPoints = [];
        } else if (selectionType === 'rect') {
            const x = Math.min(selectionStartX, selectionEndX);
            const y = Math.min(selectionStartY, selectionEndY);
            const w = Math.abs(selectionEndX - selectionStartX);
            const h = Math.abs(selectionEndY - selectionStartY);
            if (w > 2 || h > 2) {
                selectMarkersInRect(x, y, w, h, selectionAdditive);
            }
        }

        selectionClickStart = null;
        render();
        return;
    }

    if (currentMode === Mode.ERASER && isErasing) {
        isErasing = false;
        eraserHistorySaved = false;
        renderOverview();
        updateCurrentImageListItem();
    }

    if (currentMode === Mode.BRUSH && isBrushing) {
        isBrushing = false;
        brushHistorySaved = false;
        renderOverview();
        updateCurrentImageListItem();
    }
}

function handleCanvasMouseLeave(e) {
    // Don't cancel selection or panning - let document handlers continue tracking
    // Only cancel eraser/brush (we don't want them to work outside canvas)
    if (isErasing) {
        isErasing = false;
        eraserHistorySaved = false;
        renderOverview();
    }
    if (currentMode === Mode.ERASER) {
        eraserImagePos = null;
        render();
    }
    if (isBrushing) {
        isBrushing = false;
        brushHistorySaved = false;
        renderOverview();
    }
    if (currentMode === Mode.BRUSH) {
        brushImagePos = null;
        render();
    }
}

function handleDocumentMouseMove(e) {
    // Continue selection/pan even when outside canvas (only if not already on canvas)
    if (e.target === canvas) return;

    if (isSelecting && currentMode === Mode.SELECT) {
        const coords = getClampedImageCoords(e);
        selectionEndX = coords.x;
        selectionEndY = coords.y;

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

function handleDocumentMouseUp(e) {
    // Finish selection/pan even when outside canvas (only if not already on canvas)
    if (e.target === canvas) return;

    if (isPanning) {
        isPanning = false;
        canvas.classList.remove('panning');
    }

    if (isSelecting && currentMode === Mode.SELECT) {
        isSelecting = false;
        const coords = getClampedImageCoords(e);

        if (selectionType === 'lasso' && lassoPoints.length > 2) {
            lassoPoints.push(coords);
            selectMarkersInPolygon(lassoPoints, selectionAdditive);
            lassoPoints = [];
        } else if (selectionType === 'rect') {
            const x = Math.min(selectionStartX, selectionEndX);
            const y = Math.min(selectionStartY, selectionEndY);
            const w = Math.abs(selectionEndX - selectionStartX);
            const h = Math.abs(selectionEndY - selectionStartY);
            if (w > 2 || h > 2) {
                selectMarkersInRect(x, y, w, h, selectionAdditive);
            }
        }
        selectionClickStart = null;
        render();
    }
}

// ============== Eraser ==============

function eraseAtPosition(imgX, imgY) {
    if (eraserPending) return;
    eraserPending = true;

    const saveHistory = !eraserHistorySaved;
    if (saveHistory) eraserHistorySaved = true;

    eel.delete_markers_in_radius(imgX, imgY, eraserRadius, saveHistory)().then(result => {
        eraserPending = false;
        if (result && result.deleted_count > 0) {
            markers = result.markers;
            updateCounts(result.positive_count, result.negative_count, result.other_count);
            updateUndoRedoButtons(result.can_undo, result.can_redo);
            markUnsavedChanges();
            render();
        }
    }).catch(() => { eraserPending = false; });
}

function drawEraserCursorOnOverlay(imgLeft, imgTop) {
    if (!eraserImagePos || currentMode !== Mode.ERASER) return;

    const screenRadius = eraserRadius * zoom;
    const x = imgLeft + eraserImagePos.x * zoom;
    const y = imgTop + eraserImagePos.y * zoom;

    overlayCtx.beginPath();
    overlayCtx.arc(x, y, screenRadius, 0, Math.PI * 2);
    overlayCtx.strokeStyle = '#ff4444';
    overlayCtx.lineWidth = 2;
    overlayCtx.stroke();

    overlayCtx.fillStyle = 'rgba(255, 68, 68, 0.15)';
    overlayCtx.fill();

    overlayCtx.beginPath();
    overlayCtx.moveTo(x - 5, y);
    overlayCtx.lineTo(x + 5, y);
    overlayCtx.moveTo(x, y - 5);
    overlayCtx.lineTo(x, y + 5);
    overlayCtx.strokeStyle = '#ff4444';
    overlayCtx.lineWidth = 1;
    overlayCtx.stroke();
}

// ============== Brush (paint class change) ==============

function brushAtPosition(imgX, imgY) {
    if (brushPending) return;
    brushPending = true;

    const saveHistory = !brushHistorySaved;
    if (saveHistory) brushHistorySaved = true;

    eel.change_markers_in_radius(imgX, imgY, brushRadius, brushClass, saveHistory)().then(result => {
        brushPending = false;
        if (result && result.changed_count > 0) {
            markers = result.markers;
            updateCounts(result.positive_count, result.negative_count, result.other_count);
            updateUndoRedoButtons(result.can_undo, result.can_redo);
            markUnsavedChanges();
            render();
        }
    }).catch(() => { brushPending = false; });
}

function drawBrushCursorOnOverlay(imgLeft, imgTop) {
    if (!brushImagePos || currentMode !== Mode.BRUSH) return;

    const screenRadius = brushRadius * zoom;
    const x = imgLeft + brushImagePos.x * zoom;
    const y = imgTop + brushImagePos.y * zoom;

    // Use the brush color based on selected class
    const color = getBrushColor();

    overlayCtx.beginPath();
    overlayCtx.arc(x, y, screenRadius, 0, Math.PI * 2);
    overlayCtx.strokeStyle = color;
    overlayCtx.lineWidth = 2;
    overlayCtx.stroke();

    // Subtle fill with brush color (convert hex to rgba)
    overlayCtx.globalAlpha = 0.15;
    overlayCtx.fillStyle = color;
    overlayCtx.fill();
    overlayCtx.globalAlpha = 1.0;

    // Crosshair at center
    overlayCtx.beginPath();
    overlayCtx.moveTo(x - 5, y);
    overlayCtx.lineTo(x + 5, y);
    overlayCtx.moveTo(x, y - 5);
    overlayCtx.lineTo(x, y + 5);
    overlayCtx.strokeStyle = color;
    overlayCtx.lineWidth = 1;
    overlayCtx.stroke();
}

// ============== Keyboard Handlers ==============

async function handleKeyDown(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    if (e.ctrlKey || e.metaKey) {
        switch (e.key.toLowerCase()) {
            case 'z':
                e.preventDefault();
                if (e.shiftKey) handleRedo(); else handleUndo();
                return;
            case 'y':
                e.preventDefault();
                handleRedo();
                return;
            case 's':
                e.preventDefault();
                handleSaveProject();
                return;
        }
    }

    switch (e.key.toLowerCase()) {
        case 'p': setMode(Mode.POSITIVE); break;
        case 'n': setMode(Mode.NEGATIVE); break;
        case 'o': setMode(Mode.OTHER); break;
        case 's': setMode(Mode.SELECT); break;
        case 'e': setMode(Mode.ERASER); break;
        case 'h': setMode(Mode.PAN); break;
        case 'b': setMode(Mode.BRUSH); break;
        case 'a':
            if (e.ctrlKey || e.metaKey) {
                e.preventDefault();
                handleSelectAll();
            }
            break;
        case 'i':
            if (e.ctrlKey || e.metaKey) {
                e.preventDefault();
                handleInvertSelection();
            }
            break;
        case 'delete':
        case 'backspace':
            if (isProjectLoaded) {
                e.preventDefault();
                handleDeleteSelected();
            }
            break;
        case 'escape':
            handleDeselectAll();
            break;
        case 'r':
            if (isProjectLoaded) {
                if (e.shiftKey) {
                    handleSetNeedsReview();
                } else {
                    handleMarkReviewed();
                }
            }
            break;
        case 'f':
            if (isProjectLoaded) zoomToFit();
            break;
        case '1':
            if (isProjectLoaded) setZoom(1);
            break;
        case '=': case '+':
            if (isProjectLoaded) setZoom(zoom + ZOOM_STEP);
            break;
        case '-':
            if (isProjectLoaded) setZoom(zoom - ZOOM_STEP);
            break;
        case 'arrowleft':
            e.preventDefault();
            handlePrevImage();
            break;
        case 'arrowright':
            e.preventDefault();
            handleNextImage();
            break;
    }
}

// ============== Rendering ==============

function getMarkerColor(markerClass) {
    switch (markerClass) {
        case MarkerClass.POSITIVE: return POSITIVE_COLOR;
        case MarkerClass.NEGATIVE: return NEGATIVE_COLOR;
        case MarkerClass.OTHER: return OTHER_COLOR;
        default: return '#888';
    }
}

function render() {
    if (!currentImage) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(currentImage, 0, 0);

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

    renderOverlayMarkers(viewWidth, viewHeight, left, top);
}

function renderOverlayMarkers(viewWidth, viewHeight, imgLeft, imgTop) {
    const dpr = window.devicePixelRatio || 1;
    overlayCanvas.width = viewWidth * dpr;
    overlayCanvas.height = viewHeight * dpr;
    overlayCanvas.style.width = viewWidth + 'px';
    overlayCanvas.style.height = viewHeight + 'px';
    overlayCtx.scale(dpr, dpr);

    overlayCtx.clearRect(0, 0, viewWidth, viewHeight);

    markers.forEach(marker => {
        const screenX = imgLeft + marker.x * zoom;
        const screenY = imgTop + marker.y * zoom;
        drawMarkerOnOverlay(marker, screenX, screenY);
    });

    drawSelectionRect(imgLeft, imgTop);
    drawEraserCursorOnOverlay(imgLeft, imgTop);
    drawBrushCursorOnOverlay(imgLeft, imgTop);
}

function drawMarkerOnOverlay(marker, screenX, screenY) {
    const r = MARKER_RADIUS;
    const t = MARKER_THICKNESS;

    if (marker.selected) {
        overlayCtx.fillStyle = '#ffd700';
        drawCrossOnOverlay(screenX, screenY, r + 3, t + 3);
    }

    overlayCtx.fillStyle = getMarkerColor(marker.marker_class);
    drawCrossOnOverlay(screenX, screenY, r, t);
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

function updateCounts(positive, negative, other) {
    // Counts are tracked but not shown in toolbar (removed per user request)
    // The sidebar still shows counts per image
}

async function updateSummary() {
    const summary = await eel.get_summary()();
    if (summary.total > 0) {
        elements.statusSummary.textContent =
            `${summary.annotated_images}/${summary.total_images} annotated | ${summary.total} markers`;
    } else {
        elements.statusSummary.textContent = '';
    }
    updateSidebarStats();
}

// ============== Tiling ==============

function showTileModal() {
    elements.tileModal.classList.remove('hidden');
}

async function handleTileConfirm() {
    elements.tileModal.classList.add('hidden');

    const tileSize = parseInt(elements.tileSize.value);
    const skipBlank = elements.tileSkipBlank.checked;
    const blankThreshold = parseFloat(elements.tileBlankThreshold.value);

    showProgress('Tiling images...');
    const result = await eel.tile_source_images(tileSize, skipBlank, blankThreshold)();
    hideProgress();

    if (result.success) {
        const createProject = confirm(
            result.message + '\n\nCreate a new project from the tiled images?'
        );
        if (createProject) {
            // Auto-create project from tile output
            const projResult = await eel.new_project_from_dir(result.output_dir)();
            if (projResult && projResult.success) {
                isProjectLoaded = true;
                clearUnsavedChanges();
                hideWelcome();
                await loadCurrentImage();
                updateImageList();
                updateSummary();
            } else if (projResult) {
                alert(projResult.message);
            }
        }
    } else {
        alert('Tiling failed: ' + result.message);
    }
}

// Expose tiling progress callback
eel.expose(onTileProgress);
function onTileProgress(message, progress) {
    elements.progressText.textContent = message;
    elements.progressBar.style.width = (progress * 100) + '%';
}

// ============== Detection ==============

async function showDetectModal(mode) {
    if (!isProjectLoaded) {
        alert('No project loaded');
        return;
    }

    // Prevent detection on locked (reviewed) images
    if (mode === 'current' && isLocked()) {
        alert('This image is locked (reviewed). Unlock it first to run detection.');
        return;
    }

    // Warn if current image has existing markers
    if (mode === 'current' && markers.length > 0) {
        if (!confirm(`This image has ${markers.length} existing markers.\n\nDetection will replace all markers. Continue?`)) {
            return;
        }
    }

    detectMode = mode;
    elements.detectModalTitle.textContent = mode === 'all' ? 'Detect All Images' : 'Detect Current Image';

    // Check for existing annotations when detecting all
    if (mode === 'all') {
        const stats = await eel.get_annotation_stats()();
        if (stats.images_with_markers > 0 || stats.images_reviewed > 0) {
            let warning = 'Warning: This will overwrite existing annotations!\n\n';
            if (stats.images_with_markers > 0) {
                warning += `• ${stats.images_with_markers} images have markers (${stats.total_markers} total)\n`;
            }
            if (stats.images_reviewed > 0) {
                warning += `• ${stats.images_reviewed} images are marked as reviewed\n`;
            }
            warning += '\nAll markers will be replaced and review status will be reset.\n\nContinue?';

            if (!confirm(warning)) {
                return;
            }
        }
    }

    // Populate model dropdown
    showProgress('Loading models...');
    const modelsResult = await eel.get_available_models()();
    hideProgress();

    elements.detectModel.innerHTML = '';
    if (modelsResult.models && modelsResult.models.length > 0) {
        modelsResult.models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m.id;
            opt.textContent = `${m.name} (${m.id})`;
            if (m.id === modelsResult.default_model) opt.selected = true;
            elements.detectModel.appendChild(opt);
        });
    } else {
        // No models registered yet — offer to download base
        const opt = document.createElement('option');
        opt.value = 'base';
        opt.textContent = 'KiNet Base (will download)';
        elements.detectModel.appendChild(opt);
    }

    elements.detectModal.classList.remove('hidden');
}

async function handleDetectConfirm() {
    elements.detectModal.classList.add('hidden');

    const modelId = elements.detectModel.value;
    const threshold = parseFloat(elements.detectThreshold.value);
    const minDistance = parseInt(elements.detectMinDistance.value);

    showProgress('Preparing detection...');

    // Ensure model is available (downloads if needed)
    const ensureResult = await eel.ensure_model_available()();
    if (!ensureResult.success) {
        hideProgress();
        alert('Could not prepare model: ' + ensureResult.message);
        return;
    }

    let result;
    if (detectMode === 'all') {
        result = await eel.detect_all_images(modelId, threshold, minDistance)();
    } else {
        result = await eel.detect_current_image(modelId, threshold, minDistance)();
    }

    hideProgress();

    if (result && result.success) {
        if (detectMode === 'current') {
            // Update current view
            markers = result.markers;
            currentReviewStatus = result.review_status || ReviewStatus.NEEDS_REVIEW;
            updateCounts(result.positive_count, result.negative_count, result.other_count);
            updateUndoRedoButtons(result.can_undo, result.can_redo);
            markUnsavedChanges();
            render();
            renderOverview();
            updateCurrentImageListItem();
        }
        // Refresh sidebar for all cases
        await updateImageList();
        updateReviewSummary();
    } else if (result) {
        alert(result.message || 'Detection failed');
    }
}

// Expose detection progress callback
eel.expose(onDetectProgress);
function onDetectProgress(message, progress) {
    elements.progressText.textContent = message;
    elements.progressBar.style.width = (progress * 100) + '%';
}

// ============== Model Browser ==============

async function showModelBrowser() {
    showProgress('Loading models...');
    const modelsResult = await eel.get_available_models()();
    hideProgress();

    const list = elements.modelList;
    list.innerHTML = '';

    if (!modelsResult.models || modelsResult.models.length === 0) {
        list.innerHTML = '<p style="text-align:center;color:#888;padding:20px;">No models registered. Run detection to download the base model.</p>';
        elements.modelBrowserModal.classList.remove('hidden');
        return;
    }

    for (const model of modelsResult.models) {
        const entry = document.createElement('div');
        entry.className = 'model-entry' + (model.id === modelsResult.default_model ? ' default-model' : '');

        // Header
        const header = document.createElement('div');
        header.className = 'model-entry-header';
        header.innerHTML = `
            <span class="model-entry-name">${model.name}</span>
            <span class="model-entry-id">${model.id}</span>
        `;
        entry.appendChild(header);

        // Date
        if (model.created) {
            const meta = document.createElement('div');
            meta.className = 'model-entry-meta';
            meta.textContent = 'Created: ' + model.created;
            entry.appendChild(meta);
        }

        // Lineage
        if (model.parent_model) {
            const lineage = await eel.get_model_lineage(model.id)();
            if (lineage && lineage.length > 1) {
                const lineageDiv = document.createElement('div');
                lineageDiv.className = 'model-entry-lineage';
                lineageDiv.textContent = lineage.map(m => m.id).join(' -> ');
                entry.appendChild(lineageDiv);
            }
        }

        // Metrics
        if (model.metrics) {
            const metricsDiv = document.createElement('div');
            metricsDiv.className = 'model-entry-metrics';
            const parts = [];
            if (model.metrics.best_val_loss != null) parts.push(`Val loss: ${model.metrics.best_val_loss.toFixed(5)}`);
            if (model.metrics.epochs_trained != null) parts.push(`Epochs: ${model.metrics.epochs_trained}`);
            if (model.metrics.positive_f1 != null) parts.push(`Pos F1: ${model.metrics.positive_f1.toFixed(3)}`);
            if (model.metrics.negative_f1 != null) parts.push(`Neg F1: ${model.metrics.negative_f1.toFixed(3)}`);
            metricsDiv.textContent = parts.join(' | ');
            entry.appendChild(metricsDiv);
        }

        // Training data
        if (model.training_data) {
            const dataDiv = document.createElement('div');
            dataDiv.className = 'model-entry-meta';
            const td = model.training_data;
            const parts = [];
            if (td.train_images != null) parts.push(`${td.train_images} train`);
            if (td.val_images != null) parts.push(`${td.val_images} val`);
            dataDiv.textContent = 'Data: ' + parts.join(', ');
            entry.appendChild(dataDiv);
        }

        // Actions
        if (model.id !== modelsResult.default_model) {
            const actions = document.createElement('div');
            actions.className = 'model-entry-actions';
            const setDefaultBtn = document.createElement('button');
            setDefaultBtn.textContent = 'Set as Default';
            setDefaultBtn.addEventListener('click', async () => {
                await eel.set_default_model(model.id)();
                showModelBrowser(); // Refresh
            });
            actions.appendChild(setDefaultBtn);
            entry.appendChild(actions);
        } else {
            const badge = document.createElement('div');
            badge.className = 'model-entry-meta';
            badge.innerHTML = '<strong style="color:#0078d4;">Default model</strong>';
            entry.appendChild(badge);
        }

        list.appendChild(entry);
    }

    elements.modelBrowserModal.classList.remove('hidden');
}

// ============== Review Status ==============

function isLocked() {
    return currentReviewStatus === ReviewStatus.REVIEWED;
}

function updateReviewButton() {
    if (isLocked()) {
        elements.btnReviewed.textContent = 'Unlock';
        elements.btnReviewed.title = 'Unlock for Editing (R)';
        elements.btnReviewed.classList.add('locked');
    } else {
        elements.btnReviewed.textContent = 'Mark Reviewed';
        elements.btnReviewed.title = 'Mark Reviewed & Next (R)';
        elements.btnReviewed.classList.remove('locked');
    }
    // Update canvas locked state
    canvas.classList.toggle('locked', isLocked());
}

async function handleMarkReviewed() {
    if (!isProjectLoaded) return;

    const result = await eel.toggle_reviewed()();
    if (result && result.data && !result.data.error) {
        loadImageData(result.data);
        await updateImageList();
        updateReviewSummary();
    }
}

async function handleSetNeedsReview() {
    if (!isProjectLoaded) return;

    await eel.set_review_status(ReviewStatus.NEEDS_REVIEW)();
    currentReviewStatus = ReviewStatus.NEEDS_REVIEW;
    updateReviewButton();
    updateCurrentImageListItem();
    updateReviewSummary();
}

// ============== Context Menu ==============

function handleContextMenu(e) {
    e.preventDefault();
    if (!isProjectLoaded) return;

    const menu = elements.contextMenu;
    menu.style.left = e.clientX + 'px';
    menu.style.top = e.clientY + 'px';
    menu.classList.remove('hidden');
}

function hideContextMenu(e) {
    if (!elements.contextMenu.contains(e.target)) {
        elements.contextMenu.classList.add('hidden');
    }
}

async function handleContextMenuAction(e) {
    const item = e.target.closest('.context-menu-item');
    if (!item) return;

    elements.contextMenu.classList.add('hidden');
    const action = item.dataset.action;

    switch (action) {
        case 'mode-positive': setMode(Mode.POSITIVE); break;
        case 'mode-negative': setMode(Mode.NEGATIVE); break;
        case 'mode-other': setMode(Mode.OTHER); break;
        case 'mode-pan': setMode(Mode.PAN); break;
        case 'select-all': await handleSelectAll(); break;
        case 'deselect-all': await handleDeselectAll(); break;
        case 'invert-selection': await handleInvertSelection(); break;
        case 'change-positive': await handleChangeClass(MarkerClass.POSITIVE); break;
        case 'change-negative': await handleChangeClass(MarkerClass.NEGATIVE); break;
        case 'change-other': await handleChangeClass(MarkerClass.OTHER); break;
        case 'delete-selected': await handleDeleteSelected(); break;
    }
}

// ============== Selection Operations ==============

async function handleSelectAll() {
    if (!isProjectLoaded) return;
    const result = await eel.select_all()();
    if (result) {
        markers = result.markers;
        render();
    }
}

async function handleDeselectAll() {
    if (!isProjectLoaded) return;
    const result = await eel.deselect_all()();
    if (result) {
        markers = result.markers;
        render();
    }
}

async function handleInvertSelection() {
    if (!isProjectLoaded) return;
    const result = await eel.invert_selection()();
    if (result) {
        markers = result.markers;
        render();
    }
}

async function handleChangeClass(newClass) {
    if (!isProjectLoaded || isLocked()) return;
    const result = await eel.change_selected_class(newClass)();
    if (result) {
        markers = result.markers;
        updateUndoRedoButtons(result.can_undo, result.can_redo);
        markUnsavedChanges();
        render();
        renderOverview();
        updateCurrentImageListItem();
    }
}

async function handleDeleteSelected() {
    if (!isProjectLoaded || isLocked()) return;
    const result = await eel.delete_selected()();
    if (result) {
        markers = result.markers;
        updateUndoRedoButtons(result.can_undo, result.can_redo);
        markUnsavedChanges();
        render();
        renderOverview();
        updateCurrentImageListItem();
    }
}

async function selectMarkersInRect(x, y, width, height, additive = false) {
    if (!isProjectLoaded) return;
    const result = await eel.select_markers_in_rect(x, y, width, height, additive)();
    if (result) {
        markers = result.markers;
        render();
    }
}

async function selectMarkersInPolygon(points, additive = false) {
    if (!isProjectLoaded || points.length < 3) return;
    const result = await eel.select_markers_in_polygon(points, additive)();
    if (result) {
        markers = result.markers;
        render();
    }
}

async function selectMarkerAtIndex(index, additive = false) {
    if (!isProjectLoaded) return;
    const result = await eel.select_marker_at_index(index, additive)();
    if (result) {
        markers = result.markers;
        render();
    }
}

async function deselectAllMarkers() {
    if (!isProjectLoaded) return;
    const result = await eel.deselect_all()();
    if (result) {
        markers = result.markers;
        render();
    }
}

function drawSelectionRect(imgLeft, imgTop) {
    if (!isSelecting) return;

    if (selectionType === 'lasso' && lassoPoints.length > 1) {
        // Draw lasso path
        overlayCtx.strokeStyle = '#0078d4';
        overlayCtx.lineWidth = 2;
        overlayCtx.setLineDash([5, 5]);

        overlayCtx.beginPath();
        const first = lassoPoints[0];
        overlayCtx.moveTo(imgLeft + first.x * zoom, imgTop + first.y * zoom);

        for (let i = 1; i < lassoPoints.length; i++) {
            const pt = lassoPoints[i];
            overlayCtx.lineTo(imgLeft + pt.x * zoom, imgTop + pt.y * zoom);
        }

        // Draw line to current mouse position
        overlayCtx.lineTo(imgLeft + selectionEndX * zoom, imgTop + selectionEndY * zoom);
        overlayCtx.stroke();
        overlayCtx.setLineDash([]);

        // Fill the lasso area
        overlayCtx.fillStyle = 'rgba(0, 120, 212, 0.1)';
        overlayCtx.beginPath();
        overlayCtx.moveTo(imgLeft + first.x * zoom, imgTop + first.y * zoom);
        for (let i = 1; i < lassoPoints.length; i++) {
            const pt = lassoPoints[i];
            overlayCtx.lineTo(imgLeft + pt.x * zoom, imgTop + pt.y * zoom);
        }
        overlayCtx.lineTo(imgLeft + selectionEndX * zoom, imgTop + selectionEndY * zoom);
        overlayCtx.closePath();
        overlayCtx.fill();
    } else if (selectionType === 'rect') {
        const x1 = imgLeft + Math.min(selectionStartX, selectionEndX) * zoom;
        const y1 = imgTop + Math.min(selectionStartY, selectionEndY) * zoom;
        const w = Math.abs(selectionEndX - selectionStartX) * zoom;
        const h = Math.abs(selectionEndY - selectionStartY) * zoom;

        overlayCtx.strokeStyle = '#0078d4';
        overlayCtx.lineWidth = 2;
        overlayCtx.setLineDash([5, 5]);
        overlayCtx.strokeRect(x1, y1, w, h);
        overlayCtx.setLineDash([]);

        overlayCtx.fillStyle = 'rgba(0, 120, 212, 0.1)';
        overlayCtx.fillRect(x1, y1, w, h);
    }
}
