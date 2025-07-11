// Функции для модального окна информации
function openInfo(tab) {
    const infoTexts = {
        main: `<b>Главная</b><br>Загрузите XML-файлы с коллизиями и архивы изображений для инференса. После загрузки откроется подробная статистика и станет доступна загрузка результатов разметки.`,
        settings: `<b>Настройки</b><br>Выберите модель для инференса, настройте пороги уверенности и просмотрите метрики. Здесь также отображаются обученные пары категорий.`,
        train: `<b>Обучение</b><br>Загрузите XML-файлы и архивы изображений для обучения новой модели. После выбора файлов появится статистика по данным. Укажите параметры обучения и запустите процесс.`,
        progress: `<b>Прогресс обучения</b><br>Следите за статусом обучения в реальном времени: графики метрик, итоговые показатели и подробная статистика по каждому XML-файлу доступны здесь.`
    };
    let text = infoTexts[tab] || infoTexts['main'];
    document.getElementById('infoText').innerHTML = text;
    document.getElementById('infoModal').style.display = 'flex';
}
function closeInfo() {
    document.getElementById('infoModal').style.display = 'none';
}
window.onclick = function(event) {
    let modal = document.getElementById('infoModal');
    if (event.target === modal) closeInfo();
}

// --- Графики и метрики обучения ---
let lossChart, accuracyChart;
function renderTrainCharts(metrics) {
    const ctxLoss = document.getElementById('lossChart').getContext('2d');
    const ctxAcc = document.getElementById('accuracyChart').getContext('2d');
    if (lossChart) lossChart.destroy();
    if (accuracyChart) accuracyChart.destroy();
    lossChart = new Chart(ctxLoss, {
        type: 'line',
        data: {
            labels: metrics.epochs,
            datasets: [
                { label: 'Train Loss', data: metrics.train_losses, borderColor: '#23408e', fill: false },
                { label: 'Val Loss', data: metrics.val_losses, borderColor: '#e67e22', fill: false }
            ]
        },
        options: { responsive: true, plugins: { legend: { position: 'top' } } }
    });
    accuracyChart = new Chart(ctxAcc, {
        type: 'line',
        data: {
            labels: metrics.epochs,
            datasets: [
                { label: 'Val Accuracy', data: metrics.val_accuracies, borderColor: '#1a7f37', fill: false }
            ]
        },
        options: { responsive: true, plugins: { legend: { position: 'top' } } }
    });
}
function renderMetricsList(metrics) {
    const ul = document.getElementById('metrics-list');
    if (!ul) return;
    ul.innerHTML = '';
    const items = [
        `Точность (accuracy): ${metrics.final_accuracy?.toFixed(4)}`,
        `F1-score: ${metrics.final_f1?.toFixed(4)}`,
        `Recall: ${metrics.final_recall?.toFixed(4)}`,
        `Precision: ${metrics.final_precision?.toFixed(4)}`
    ];
    for (const item of items) {
        const li = document.createElement('li');
        li.textContent = item;
        ul.appendChild(li);
    }
}
function renderConfusionMatrix(matrix) {
    const table = document.getElementById('confusion-matrix');
    if (!table) return;
    table.innerHTML = '';
    if (!matrix || !matrix.length) return;
    // Заголовок
    const header = document.createElement('tr');
    header.innerHTML = '<th></th>' + matrix[0].map((_, j) => `<th>Прогноз ${j}</th>`).join('');
    table.appendChild(header);
    // Строки
    for (let i = 0; i < matrix.length; i++) {
        const row = document.createElement('tr');
        row.innerHTML = `<th>Истинный ${i}</th>` + matrix[i].map(val => `<td>${val}</td>`).join('');
        table.appendChild(row);
    }
}
function renderCategoryPairsTable(pairs, sortCol = 2, sortDir = 'desc') {
    const container = document.getElementById('categoryPairsList');
    container.innerHTML = '';
    if (!pairs || pairs.length === 0) {
        container.innerHTML = '<div class="empty">Нет пар</div>';
        return;
    }
    const search = (document.getElementById('categoryPairsSearch')?.value || '').toLowerCase();
    let filtered = pairs.filter(pair =>
        pair[0].toLowerCase().includes(search) ||
        pair[1].toLowerCase().includes(search)
    );
    filtered.sort((a, b) => {
        let v1 = a[sortCol], v2 = b[sortCol];
        if (sortCol === 2) { v1 = +v1; v2 = +v2; }
        if (v1 < v2) return sortDir === 'asc' ? -1 : 1;
        if (v1 > v2) return sortDir === 'asc' ? 1 : -1;
        return 0;
    });
    let html = '<table class="category-pairs-table"><thead><tr>' +
        '<th data-col="0">Категория 1</th>' +
        '<th data-col="1">Категория 2</th>' +
        '<th data-col="2">Кол-во</th>' +
        '</tr></thead><tbody>';
    filtered.forEach(pair => {
        html += `<tr><td>${pair[0]}</td><td>${pair[1]}</td><td>${pair[2]}</td></tr>`;
    });
    html += '</tbody></table>';
    container.innerHTML = html;
    // Сортировка по клику
    const ths = container.querySelectorAll('th');
    ths.forEach(th => {
        th.style.cursor = 'pointer';
        th.onclick = function() {
            let col = +th.getAttribute('data-col');
            let dir = (window.categoryPairsSortCol === col && window.categoryPairsSortDir === 'asc') ? 'desc' : 'asc';
            window.categoryPairsSortCol = col;
            window.categoryPairsSortDir = dir;
            renderCategoryPairsTable(window.categoryPairsData, col, dir);
        };
    });
}
// Пример функции для обновления прогресса (можно вызывать через fetch)
function updateTrainProgress(metrics) {
    renderTrainCharts(metrics);
    renderMetricsList(metrics);
    renderConfusionMatrix(metrics.confusion_matrix);
}

// Live-обновление прогресса обучения
function pollTrainProgress() {
    fetch('/train_progress')
        .then(r => r.json())
        .then(data => {
            if (data.status === 'not_started') return;
            updateTrainProgress(data);
        });
}
if (window.location.pathname.includes('train_progress')) {
    setInterval(pollTrainProgress, 2000);
    pollTrainProgress();
}

// --- Динамический анализ файлов для обучения (AJAX) ---
function analyzeTrainFiles() {
    const xmlInput = document.getElementById('trainXmlInput');
    const zipInput = document.getElementById('trainZipInput');
    const statsDiv = document.getElementById('trainStats');
    if (!xmlInput || !zipInput || !statsDiv) return;
    const formData = new FormData();
    for (let f of xmlInput.files) formData.append('xml_file', f);
    for (let f of zipInput.files) formData.append('zip_file', f);
    statsDiv.innerHTML = '<span style="color:#888">Анализ данных...</span>';
    fetch('/analyze_train_files', { method: 'POST', body: formData })
        .then(r => r.json())
        .then(data => {
            if (data.success && data.stats) {
                let html = `<div class='stats-block'>` +
                    `<b>Коллизий:</b> ${data.stats.total_collisions}<br>` +
                    `<b>Найдено изображений:</b> ${data.stats.found_images}<br>` +
                    `<b>Распределение по классам:</b><ul>`;
                for (const [k, v] of Object.entries(data.stats.class_counts)) {
                    html += `<li>Класс ${k}: ${v}</li>`;
                }
                html += `</ul></div>`;
                statsDiv.innerHTML = html;
            } else {
                statsDiv.innerHTML = `<span style='color:#b71c1c'>Ошибка анализа: ${data.error || 'Не удалось получить статистику'}</span>`;
            }
        })
        .catch(() => {
            statsDiv.innerHTML = `<span style='color:#b71c1c'>Ошибка анализа: не удалось связаться с сервером</span>`;
        });
}
// Автоматически анализировать при выборе файлов
const trainXmlInput = document.getElementById('trainXmlInput');
const trainZipInput = document.getElementById('trainZipInput');
if (trainXmlInput && trainZipInput) {
    trainXmlInput.addEventListener('change', analyzeTrainFiles);
    trainZipInput.addEventListener('change', analyzeTrainFiles);
}

// --- Динамический анализ файлов для инференса (главная) ---
function analyzeMainFiles() {
    const xmlInput = document.getElementById('xmlInput');
    const zipInput = document.getElementById('zipInput');
    const statsDiv = document.getElementById('analyzeStats');
    if (!xmlInput || !zipInput || !statsDiv) return;
    const formData = new FormData();
    for (let f of xmlInput.files) formData.append('xml_file', f);
    for (let f of zipInput.files) formData.append('zip_file', f);
    statsDiv.innerHTML = '<span style="color:#888">Анализ данных...</span>';
    fetch('/analyze_main_files', { method: 'POST', body: formData })
        .then(r => r.json())
        .then(data => {
            if (data.success && data.stats) {
                const stats = data.stats;
                let html = `<div class='stats-grid'>` +
                    `<div class='stat-item'><div class='stat-label'>XML файлов</div><div class='stat-value'>${stats.xml_file_count ?? '—'}</div></div>` +
                    `<div class='stat-item'><div class='stat-label'>ZIP архивов</div><div class='stat-value'>${stats.zip_file_count ?? '—'}</div></div>` +
                    `<div class='stat-item'><div class='stat-label'>Всего коллизий</div><div class='stat-value'>${stats.total_collisions ?? '—'}</div></div>` +
                    `<div class='stat-item'><div class='stat-label'>Найдено изображений</div><div class='stat-value'>${stats.image_count ?? stats.found_images ?? '—'}</div></div>` +
                    `</div>`;
                // Распределение по классам
                if (stats.class_counts && typeof stats.class_counts === 'object') {
                    html += `<div style='margin:12px 0 16px 0;'><b>Распределение по классам:</b><ul style='margin:0 0 0 16px;'>`;
                    for (const [k, v] of Object.entries(stats.class_counts)) {
                        html += `<li>Класс ${k}: ${v}</li>`;
                    }
                    html += `</ul></div>`;
                }
                // Таблица по каждому XML-файлу (только 2 столбца)
                if (stats.per_file && Array.isArray(stats.per_file) && stats.per_file.length > 0) {
                    html += `<div style='margin-top:16px;'>`;
                    html += `<table class="xml-stats-table" style="width:100%;margin-bottom:16px;">
                        <thead>
                            <tr>
                                <th>XML файл</th>
                                <th>Коллизий всего</th>
                            </tr>
                        </thead>
                        <tbody>`;
                    stats.per_file.forEach(fileStat => {
                        html += `<tr>
                            <td>${fileStat.file}</td>
                            <td><b>${fileStat.total_collisions}</b></td>
                        </tr>`;
                    });
                    html += `</tbody></table></div>`;
                }
                statsDiv.innerHTML = html;
                // Выводим таблицу пар категорий
                window.categoryPairsSortCol = 2;
                window.categoryPairsSortDir = 'desc';
                window.categoryPairsData = stats.category_pairs || [];
                showCategoryPairsTable(window.categoryPairsData);
            } else {
                statsDiv.innerHTML = `<span style='color:#b71c1c'>Ошибка анализа: ${data.error || 'Не удалось получить статистику'}</span>`;
                showCategoryPairsTable([]);
            }
        })
        .catch(() => {
            statsDiv.innerHTML = `<span style='color:#b71c1c'>Ошибка анализа: не удалось связаться с сервером</span>`;
            showCategoryPairsTable([]);
        });
}

function showCategoryPairsTable(pairs) {
    const searchInput = document.getElementById('categoryPairsSearch');
    window.categoryPairsSortCol = window.categoryPairsSortCol ?? 2;
    window.categoryPairsSortDir = window.categoryPairsSortDir ?? 'desc';
    window.categoryPairsData = pairs;
    if (pairs && pairs.length) {
        if (searchInput) searchInput.style.display = '';
        renderCategoryPairsTable(pairs, window.categoryPairsSortCol, window.categoryPairsSortDir);
    } else {
        if (searchInput) searchInput.style.display = 'none';
        const list = document.getElementById('categoryPairsList');
        if (list) list.innerHTML = '';
    }
}
// Автоматически анализировать при выборе файлов
const mainXmlInput = document.getElementById('xmlInput');
const mainZipInput = document.getElementById('zipInput');
if (mainXmlInput && mainZipInput) {
    mainXmlInput.addEventListener('change', analyzeMainFiles);
    mainZipInput.addEventListener('change', analyzeMainFiles);
}

// В функции updatePreviewStats (train.html):
async function updatePreviewStatsTrainPage() {
    const xmlInput = document.getElementById('trainXmlInput');
    const zipInput = document.getElementById('trainZipInput');
    const previewStats = document.getElementById('previewStats');
    if (!xmlInput || !zipInput || !previewStats) return;

    const formData = new FormData();
    for (let f of xmlInput.files) formData.append('xml_file', f);
    for (let f of zipInput.files) formData.append('zip_file', f);

    previewStats.innerHTML = '<span style="color:#888">Анализ данных...</span>';

    try {
        const resp = await fetch('/api/train_preview', { method: 'POST', body: formData });
        const data = await resp.json();
        if (data.error) {
            previewStats.innerHTML = `<span style="color:#c62828;">${data.error}</span>`;
            window.categoryPairsData = [];
            renderCategoryPairsTable(window.categoryPairsData);
        } else {
            // Итоговая строка по всем файлам (используем stats_total)
            const statsContainer = document.getElementById('statsContainer');
            const stats = data.stats_total || {};
            let statsHtml = `<div class="stats-grid">
                <div class="stat-item"><div class="stat-label">Файлов</div><div class="stat-value">${stats.total_files ?? '-'}</div></div>
                <div class="stat-item"><div class="stat-label">Всего коллизий</div><div class="stat-value">${stats.total_collisions ?? '-'}</div></div>
                <div class="stat-item"><div class="stat-label">Approved</div><div class="stat-value">${stats.total_approved ?? '-'}</div></div>
                <div class="stat-item"><div class="stat-label">Active</div><div class="stat-value">${stats.total_active ?? '-'}</div></div>
                <div class="stat-item"><div class="stat-label">Reviewed</div><div class="stat-value">${stats.total_reviewed ?? '-'}</div></div>
            </div>`;
            statsContainer.innerHTML = statsHtml;
            // Ссылки для скачивания
            const downloadContainer = document.getElementById('downloadContainer');
            let downloadHtml = '<h4>📥 Скачать результаты:</h4><div class="download-links">';
            data.download_links.forEach(link => {
                downloadHtml += `
                    <a href="${link.url}" class="download-link">
                        <span class="file-name">${link.name}</span>
                        <span class="download-icon">⬇️</span>
                    </a>
                `;
            });
            downloadHtml += '</div>';
            downloadContainer.innerHTML = downloadHtml;
            results.style.display = 'block';
        }
    } catch (e) {
        previewStats.innerHTML = `<span style="color:#c62828;">Ошибка предпросмотра: ${e.message}</span>`;
        window.categoryPairsData = [];
        renderCategoryPairsTable(window.categoryPairsData);
    }
}

// Безопасно добавляем обработчик input для поиска по категориям
const searchInput = document.getElementById('categoryPairsSearch');
if (searchInput) {
    searchInput.addEventListener('input', function() {
        showCategoryPairsTable(window.categoryPairsData);
    });
}

// Логика разметки visual/Reviewed реализована на backend, frontend только отображает результат

// --- Загрузка настроек при загрузке страницы для analysisInfo ---
function updateAnalysisInfoFromSettings(settings) {
    const analysisInfo = document.getElementById('analysisInfo');
    if (!analysisInfo) return;
    let mode = settings.inference_mode === 'model' ? 'модель' : 'алгоритм';
    let manual = settings.manual_review_enabled ? 'с ручной разметкой' : 'без ручной разметки';
    let format = settings.export_format === 'bimstep' ? 'BIM Step' : 'стандартный';
    let model = settings.model_file || '';
    analysisInfo.textContent = `В режиме ${mode} ${manual}. Формат экспорта: ${format}${mode === 'модель' ? `, модель: ${model}` : ''}`;
}

document.addEventListener('DOMContentLoaded', function() {
    fetch('/api/settings').then(r => r.json()).then(settings => {
        updateAnalysisInfoFromSettings(settings);
    });
    // ... существующий код ...
});

// --- Модальное окно для ручной разметки ---
let manualReviewQueue = [];
let manualReviewIndex = 0;
let manualReviewResults = [];

// --- Горячие клавиши для ручной разметки ---
document.addEventListener('keydown', function(e) {
    const modal = document.getElementById('manualReviewModal');
    if (!modal || modal.style.display !== 'flex') return;
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
    if (e.key === '1') {
        markManualReview('Approved');
        e.preventDefault();
    } else if (e.key === '2') {
        markManualReview('Active');
        e.preventDefault();
    } else if (e.key === '3') {
        markManualReview('Reviewed');
        e.preventDefault();
    }
});

function showManualReviewModal(collisions) {
    manualReviewQueue = collisions;
    manualReviewIndex = 0;
    manualReviewResults = [];
    if (manualReviewQueue.length > 0) {
        renderManualReviewItem();
        document.getElementById('manualReviewModal').style.display = 'flex';
    }
}

function renderManualReviewItem() {
    const item = manualReviewQueue[manualReviewIndex];
    if (!item) return closeManualReview();
    // Картинка
    const img = document.getElementById('manualReviewImage');
    if (item.image_file) {
        if (item.session_id && item.image_file) {
            img.src = `/download/${item.session_id}/${encodeURIComponent(item.image_file)}`;
        } else if (item.image_file.startsWith('http')) {
            img.src = item.image_file;
        } else {
            img.src = item.image_file;
        }
        img.style.display = '';
    } else {
        img.src = '';
        img.style.display = 'none';
    }
    // Счётчик (например, 4/35)
    const counter = document.getElementById('manualReviewCounter');
    if (counter) {
        counter.textContent = `${manualReviewIndex + 1} / ${manualReviewQueue.length}`;
    }
    // Кнопки навигации
    const prevBtn = document.getElementById('manualReviewPrev');
    const nextBtn = document.getElementById('manualReviewNext');
    if (prevBtn) prevBtn.disabled = manualReviewIndex === 0;
    if (nextBtn) nextBtn.disabled = manualReviewIndex === manualReviewQueue.length - 1;
    // Инфо
    document.getElementById('manualReviewInfo').innerHTML =
        `<b>Категории:</b><br><div>${item.element1_category}</div><div>${item.element2_category}</div>` +
        (item.description ? `<br><b>Описание:</b> ${item.description}` : '');
}

function manualReviewPrev() {
    if (manualReviewIndex > 0) {
        manualReviewIndex--;
        renderManualReviewItem();
    }
}

function manualReviewNext() {
    if (manualReviewIndex < manualReviewQueue.length - 1) {
        manualReviewIndex++;
        renderManualReviewItem();
    }
}

function markManualReview(status) {
    const item = manualReviewQueue[manualReviewIndex];
    manualReviewResults.push({
        clash_id: item.clash_id,
        status: status,
        source_file: item.source_file
    });
    manualReviewIndex++;
    if (manualReviewIndex < manualReviewQueue.length) {
        renderManualReviewItem();
    } else {
        closeManualReview();
        // Отправляем результаты на backend
        const sessionId = window.lastSessionId || null;
        if (sessionId) {
            fetch('/api/manual_review', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: sessionId, reviews: manualReviewResults })
            })
            .then(r => r.json())
            .then(data => {
                if (data.success) {
                    alert('Ручная разметка успешно сохранена!');
                } else {
                    alert('Ошибка сохранения разметки: ' + (data.error || 'Неизвестная ошибка'));
                }
            })
            .catch(e => {
                alert('Ошибка сети при сохранении разметки: ' + e.message);
            });
        } else {
            alert('Ручная разметка завершена! (session_id не найден)');
        }
    }
}

function closeManualReview() {
    document.getElementById('manualReviewModal').style.display = 'none';
}

// --- Встраиваем запуск ручной разметки после анализа ---
document.getElementById('analyzeForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const results = document.getElementById('results');
    const errorContainer = document.getElementById('errorContainer');
    let analysisInfo = document.getElementById('analysisInfo');
    // Показываем индикатор загрузки
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML = 'Анализ выполняется...';
    loadingIndicator.style.display = 'block';
    results.style.display = 'none';
    errorContainer.style.display = 'none';

    // analysisInfo обновляется из настроек до анализа не требуется, т.к. уже обновлено при загрузке
    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        // --- Новый блок: отображаем настройки анализа ---
        if (data.analysis_settings && analysisInfo) {
            updateAnalysisInfoFromSettings(data.analysis_settings);
        }
        if (data.error) {
            errorContainer.innerHTML = `<span class="icon">⚠️</span> ${data.error}`;
            errorContainer.style.display = 'block';
        } else {
            // ... остальной код ...
            if (data.session_id) window.lastSessionId = data.session_id;
            if (data.manual_review_collisions && Array.isArray(data.manual_review_collisions) && data.manual_review_collisions.length > 0) {
                showManualReviewModal(data.manual_review_collisions);
            }
        }
    } catch (error) {
        errorContainer.innerHTML = `<span class="icon">⚠️</span> Ошибка сети: ${error.message}`;
        errorContainer.style.display = 'block';
    } finally {
        // Скрываем индикатор загрузки
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '🔍 Начать анализ';
        loadingIndicator.style.display = 'none';
    }
});
