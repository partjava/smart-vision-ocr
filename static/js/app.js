// 选项卡切换逻辑
function switchTab(tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

    if (tabName === 'plate') {
        event.currentTarget.classList.add('active');
        document.getElementById('plate-tab').classList.add('active');
    } else {
        event.currentTarget.classList.add('active');
        document.getElementById('document-tab').classList.add('active');
    }
}

// 通用剪贴板复制功能
function copyText(elementId) {
    const text = document.getElementById(elementId).innerText;
    navigator.clipboard.writeText(text).then(() => {
        const copyBtn = event.currentTarget;
        copyBtn.innerHTML = '<i class="fa-solid fa-check"></i> 已复制';
        setTimeout(() => {
            copyBtn.innerHTML = '<i class="fa-regular fa-copy"></i> 复制文本';
        }, 2000);
    });
}

// ----------------- 车牌识别静态图上传处理 -----------------
const plateDropzone = document.getElementById('plate-dropzone');
const plateFileInput = document.getElementById('plate-file-input');
const platePreviewBox = document.getElementById('plate-preview-box');
const plateResultCard = document.getElementById('plate-result-card');
const plateResultText = document.getElementById('plate-result-text');
const plateTime = document.getElementById('plate-time');
const plateStepsSection = document.getElementById('plate-steps-section');

// 上传交互事件
if (plateDropzone) {
    plateDropzone.addEventListener('dragover', (e) => { e.preventDefault(); plateDropzone.classList.add('dragover'); });
    plateDropzone.addEventListener('dragleave', () => { plateDropzone.classList.remove('dragover'); });
    plateDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        plateDropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handlePlateUpload(e.dataTransfer.files[0]);
        }
    });
}

if (plateFileInput) {
    plateFileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handlePlateUpload(e.target.files[0]);
        }
    });
}

function handlePlateUpload(file) {
    // 1. 在上传区域显示原图预览
    const reader = new FileReader();
    reader.onload = (e) => {
        plateDropzone.innerHTML = `<img src="${e.target.result}" alt="预览" style="max-height:80%; max-width:80%; object-fit:contain; border-radius:8px;">`;
    };
    reader.readAsDataURL(file);

    plateResultCard.style.display = 'none';
    plateStepsSection.style.display = 'none';

    const customPlateCard = document.getElementById('custom-plate-result-card');
    if (customPlateCard) customPlateCard.style.display = 'none';

    // 车牌识别路线: YOLO 定位 + 投影分割 + 三模型 CNN
    const dlFormData = new FormData();
    dlFormData.append('image', file);

    fetch('/api/segment_and_predict_plate', {
        method: 'POST',
        body: dlFormData
    })
    .then(response => response.json())
    .then(data => {
        if (data.plate_found && customPlateCard) {
            customPlateCard.style.display = 'block';

            // 显示 YOLO 裁剪车牌到预览区
            if (data.plate_crop) {
                platePreviewBox.innerHTML = `
                    <div style="text-align: center; width: 100%;">
                        <p style="color:var(--text-secondary); font-size:0.8rem; margin-bottom:0.5rem;">YOLO 定位裁剪车牌</p>
                        <img src="data:image/jpeg;base64,${data.plate_crop}" style="max-height: 80px; border: 2px solid var(--success); border-radius: 4px;">
                    </div>
                `;
            }

            // 渲染字符切片
            const slicesContainer = document.getElementById('custom-plate-slices');
            if (slicesContainer && data.char_slices) {
                slicesContainer.innerHTML = data.char_slices.map(slice => 
                    `<img src="data:image/png;base64,${slice}" style="height: 36px; border: 1px solid rgba(255,255,255,0.1); border-radius: 4px;">`
                ).join('');
            }
            
            // 渲染三模型结果 + 置信度
            const CONF_THRESHOLD = 0.6;
            function renderPlateConf(modelData) {
                if (modelData.status !== 'success') return "未加载权重";
                const text = modelData.text || '未识别';
                const conf = modelData.confidence || 0;
                const confColor = conf < CONF_THRESHOLD ? 'var(--danger)' : 'var(--success)';
                const warn = conf < CONF_THRESHOLD ? ' ⚠️' : '';
                return `${text}<br><span style="font-size:0.7rem;color:${confColor}">${(conf*100).toFixed(1)}%${warn}</span>`;
            }
            document.getElementById('custom-plate-resnet').innerHTML = renderPlateConf(data.predictions.resnet);
            document.getElementById('custom-plate-mobilenet').innerHTML = renderPlateConf(data.predictions.mobilenet);
            document.getElementById('custom-plate-custom').innerHTML = renderPlateConf(data.predictions.custom_cnn);
            document.getElementById('custom-plate-ensemble').innerHTML = renderPlateConf(data.predictions.ensemble);

            // 渲染逐字符对比表
            renderPlateComparisonTable(data.predictions);
        }
    })
    .catch(err => {
        console.error("模型车牌切分与预测失败:", err);
    });
}

// ----------------- 文档扫描与通用 OCR 上传处理 -----------------
const docDropzone = document.getElementById('doc-dropzone');
const docFileInput = document.getElementById('doc-file-input');
const docPreviewBox = document.getElementById('doc-preview-box');
const docResultCard = document.getElementById('doc-result-card');
const docResultText = document.getElementById('doc-result-text');
const docTime = document.getElementById('doc-time');
const docStepsSection = document.getElementById('doc-steps-section');

if (docDropzone) {
    docDropzone.addEventListener('dragover', (e) => { e.preventDefault(); docDropzone.classList.add('dragover'); });
    docDropzone.addEventListener('dragleave', () => { docDropzone.classList.remove('dragover'); });
    docDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        docDropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            handleDocUpload(e.dataTransfer.files[0]);
        }
    });
}

if (docFileInput) {
    docFileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleDocUpload(e.target.files[0]);
        }
    });
}

function handleDocUpload(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        docPreviewBox.innerHTML = `<img src="${e.target.result}" alt="预览">`;
    };
    reader.readAsDataURL(file);

    const formData = new FormData();
    formData.append('image', file);

    docResultCard.style.display = 'none';
    docStepsSection.style.display = 'none';

    fetch('/api/scan_document', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => { throw new Error(data.error || '服务器错误'); });
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }

        // 显示结果
        docResultCard.style.display = 'block';
        docResultText.innerText = data.recognized_text;
        docTime.innerText = `OCR 耗时: ${Math.round(data.inference_time_ms)} ms`;

        // 渲染增强预览图
        docPreviewBox.innerHTML = `
            <div style="display: flex; gap: 10px; width: 100%; height: 100%;">
                <div style="flex:1; text-align:center;">
                    <span style="font-size:0.75rem; color:var(--text-secondary)">原始图片</span>
                    <img src="${data.original_url}" style="height:210px; object-fit:contain; margin-top:5px;">
                </div>
                <div style="flex:1; text-align:center;">
                    <span style="font-size:0.75rem; color:var(--success)">去阴影二值化增强文档</span>
                    <img src="${data.enhanced_url}" style="height:210px; object-fit:contain; margin-top:5px; border:1px solid rgba(16,185,129,0.3)">
                </div>
            </div>
        `;

        // 渲染步骤图
        document.getElementById('doc-step-canny').src = data.steps.canny;
        document.getElementById('doc-step-contours').src = data.steps.contours;
        document.getElementById('doc-step-warped').src = data.steps.warped_gray;
        document.getElementById('doc-step-enhanced').src = data.enhanced_url;
        docStepsSection.style.display = 'block';
    })
    .catch(err => {
        console.error(err);
        alert("文档扫描失败: " + err.message);
    });
}

// ----------------- 实时视频车牌识别与日志轮询 -----------------
const videoDropzone = document.getElementById('video-dropzone');
const videoFileInput = document.getElementById('video-file-input');
const btnStartVideo = document.getElementById('btn-start-video');
const videoShowcase = document.getElementById('video-showcase');
const videoStreamImg = document.getElementById('video-stream-img');
const logsContainer = document.getElementById('logs-container');
const logCountBadge = document.getElementById('log-count');

let logPoller = null;
let videoPaused = false;

if (videoDropzone) {
    videoDropzone.addEventListener('dragover', (e) => { e.preventDefault(); videoDropzone.classList.add('dragover'); });
    videoDropzone.addEventListener('dragleave', () => { videoDropzone.classList.remove('dragover'); });
    videoDropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        videoDropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            uploadVideoFile(e.dataTransfer.files[0]);
        }
    });
}

if (videoFileInput) {
    videoFileInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            uploadVideoFile(e.target.files[0]);
        }
    });
}

function uploadVideoFile(file) {
    const formData = new FormData();
    formData.append('video', file);

    videoDropzone.innerHTML = `<i class="fa-solid fa-spinner fa-spin" style="font-size: 2rem; color: var(--primary);"></i><p>正在上传视频并进行数据初始化...</p>`;
    btnStartVideo.style.display = 'none';

    fetch('/api/upload_video', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            videoDropzone.innerHTML = `<i class="fa-solid fa-circle-check" style="color:var(--success); font-size:2rem; margin-bottom: 0.5rem;"></i><p>视频已就绪: ${file.name}</p>`;
            btnStartVideo.style.display = 'inline-block';
        } else {
            alert(data.error);
        }
    })
    .catch(err => {
        console.error(err);
        alert("视频上传初始化失败。");
    });
}

function startVideoProcessing() {
    // 启动视频展示框
    videoShowcase.style.display = 'grid';
    // 显示暂停按钮
    document.getElementById('btn-pause-video').style.display = 'inline-block';
    videoPaused = false;
    document.getElementById('pause-icon').className = 'fa-solid fa-pause';
    document.getElementById('pause-text').innerText = '暂停';
    document.getElementById('pause-overlay').style.display = 'none';

    // 设置流路径
    videoStreamImg.src = "/video_feed?t=" + new Date().getTime();

    // 清空日志区域
    logsContainer.innerHTML = '';

    // 启动日志轮询 (每秒轮询一次通行记录)
    if (logPoller) clearInterval(logPoller);

    logPoller = setInterval(() => {
        if (videoPaused) return;  // 暂停时不轮询日志
        fetch('/api/video_logs')
        .then(res => res.json())
        .then(logs => {
            if (logs.length === 0) {
                logsContainer.innerHTML = `<div class="preview-placeholder" style="padding-top: 5rem; text-align: center;">正在实时分析视频车牌通行情况...</div>`;
                return;
            }

            logCountBadge.innerText = `${logs.length} 条记录`;

            // 倒序排列，最新记录显示在最上方
            let html = '';
            for (let i = logs.length - 1; i >= 0; i--) {
                const log = logs[i];
                html += `
                    <div class="log-item">
                        <img src="${log.image_url}" alt="车牌截取">
                        <div class="log-info">
                            <span class="log-plate">${log.plate}</span>
                            <span class="log-time">${log.time} 通行</span>
                            ${log.model ? `<span class="log-model" style="font-size:0.7rem;color:var(--text-secondary);">模型: ${log.model}</span>` : ''}
                        </div>
                    </div>
                `;
            }
            logsContainer.innerHTML = html;
        });
    }, 1000);
}

function togglePauseVideo() {
    videoPaused = !videoPaused;
    const overlay = document.getElementById('pause-overlay');
    const icon = document.getElementById('pause-icon');
    const text = document.getElementById('pause-text');

    if (videoPaused) {
        // 暂停：只显示遮罩，不清空 img（保留最后一帧画面）
        overlay.style.display = 'flex';
        icon.className = 'fa-solid fa-play';
        text.innerText = '继续';
    } else {
        // 继续：隐藏遮罩
        overlay.style.display = 'none';
        icon.className = 'fa-solid fa-pause';
        text.innerText = '暂停';
    }
}

// ----------------- HTML5 Canvas 手写画板交互 -----------------
const canvas = document.getElementById('canvas');
let ctx = null;

if (canvas) {
    ctx = canvas.getContext('2d');
    setupCanvas();
}

function setupCanvas() {
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 14;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    
    // 清空为全黑背景 (EMNIST训练为黑底白字)
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    let drawing = false;

    // 鼠标事件
    canvas.addEventListener('mousedown', () => { drawing = true; ctx.beginPath(); });
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', () => { drawing = false; });
    canvas.addEventListener('mouseleave', () => { drawing = false; });

    // 触摸事件 (适配移动端)
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        drawing = true;
        ctx.beginPath();
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        ctx.moveTo(touch.clientX - rect.left, touch.clientY - rect.top);
    });
    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        if (!drawing) return;
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        ctx.lineTo(touch.clientX - rect.left, touch.clientY - rect.top);
        ctx.stroke();
    });
    canvas.addEventListener('touchend', () => { drawing = false; });
}

function draw(e) {
    if (!e.buttons) return; // 必须按下鼠标
    const rect = canvas.getBoundingClientRect();
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
}

function clearCanvas() {
    if (!ctx) return;
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // 重置预测文本
    const resChar = document.getElementById('res-char');
    const mobChar = document.getElementById('mob-char');
    const customChar = document.getElementById('custom-char');
    const resLatency = document.getElementById('res-latency');
    const mobLatency = document.getElementById('mob-latency');
    const customLatency = document.getElementById('custom-latency');
    const resTop5 = document.getElementById('res-top5');
    const mobTop5 = document.getElementById('mob-top5');
    const customTop5 = document.getElementById('custom-top5');
    const speedConclusion = document.getElementById('speed-conclusion');

    if (resChar) resChar.innerText = '-';
    if (mobChar) mobChar.innerText = '-';
    if (customChar) customChar.innerText = '-';
    if (resLatency) resLatency.innerText = '推理耗时: -- ms';
    if (mobLatency) mobLatency.innerText = '推理耗时: -- ms';
    if (customLatency) customLatency.innerText = '推理耗时: -- ms';
    if (resTop5) resTop5.innerHTML = '';
    if (mobTop5) mobTop5.innerHTML = '';
    if (customTop5) customTop5.innerHTML = '';
    if (speedConclusion) speedConclusion.innerText = '等待画板写入并完成识别，系统将自动对比三个模型的推理延迟效能与轻量化表现...';
}

function predictDrawing() {
    if (!canvas) return;
    const base64Data = canvas.toDataURL('image/png');

    // 发起预测 API 请求
    fetch('/api/predict_canvas', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: base64Data })
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }

        // 置信度阈值
        const CONF_THRESHOLD = 0.6;

        // 1. 渲染 ResNet 结果
        if (data.resnet) {
            const conf = data.resnet.confidence || 0;
            const confColor = conf < CONF_THRESHOLD ? 'var(--danger)' : 'var(--success)';
            const confWarn = conf < CONF_THRESHOLD ? ' ⚠️ 低置信度' : '';
            document.getElementById('res-char').innerHTML = `${data.resnet.prediction}<span style="font-size:0.7rem;color:${confColor};margin-left:4px;">${(conf*100).toFixed(1)}%${confWarn}</span>`;
            document.getElementById('res-latency').innerText = `推理耗时: ${data.resnet.time_ms.toFixed(3)} ms`;
            renderTop5List('res-top5', data.resnet.top5, 'resnet');
        }

        // 2. 渲染 MobileNet 结果
        if (data.mobilenet) {
            const conf = data.mobilenet.confidence || 0;
            const confColor = conf < CONF_THRESHOLD ? 'var(--danger)' : 'var(--success)';
            const confWarn = conf < CONF_THRESHOLD ? ' ⚠️ 低置信度' : '';
            document.getElementById('mob-char').innerHTML = `${data.mobilenet.prediction}<span style="font-size:0.7rem;color:${confColor};margin-left:4px;">${(conf*100).toFixed(1)}%${confWarn}</span>`;
            document.getElementById('mob-latency').innerText = `推理耗时: ${data.mobilenet.time_ms.toFixed(3)} ms`;
            renderTop5List('mob-top5', data.mobilenet.top5, 'mobilenet');
        }

        // 3. 渲染 CustomCNN 结果
        if (data.custom_cnn) {
            const conf = data.custom_cnn.confidence || 0;
            const confColor = conf < CONF_THRESHOLD ? 'var(--danger)' : 'var(--success)';
            const confWarn = conf < CONF_THRESHOLD ? ' ⚠️ 低置信度' : '';
            const customCharEl = document.getElementById('custom-char');
            const customLatencyEl = document.getElementById('custom-latency');
            if (customCharEl) customCharEl.innerHTML = `${data.custom_cnn.prediction}<span style="font-size:0.7rem;color:${confColor};margin-left:4px;">${(conf*100).toFixed(1)}%${confWarn}</span>`;
            if (customLatencyEl) customLatencyEl.innerText = `推理耗时: ${data.custom_cnn.time_ms.toFixed(3)} ms`;
            renderTop5List('custom-top5', data.custom_cnn.top5, 'custom-cnn');
        }

        // 4. 计算比较结论
        const resTime = data.resnet ? data.resnet.time_ms : 999;
        const mobTime = data.mobilenet ? data.mobilenet.time_ms : 999;
        const customTime = data.custom_cnn ? data.custom_cnn.time_ms : 999;
        
        let fastestName = "ResNet18";
        let fastestTime = resTime;
        if (mobTime < fastestTime) {
            fastestName = "MobileNetV3-Small";
            fastestTime = mobTime;
        }
        if (customTime < fastestTime) {
            fastestName = "CustomCharCNN";
            fastestTime = customTime;
        }
        
        let conclusion = `三模型比对完成。其中，<b>${fastestName}</b> 表现出最佳的推理速度，平均延迟仅为 <b>${fastestTime.toFixed(3)} ms</b>。`;
        if (data.custom_cnn) {
            conclusion += ` 从零搭建的 CustomCharCNN 展现了极简模型的速度优势（耗时 ${customTime.toFixed(3)} ms），适合在边缘计算或算力有限的嵌入式场景中进行部署；而 ResNet18（${resTime.toFixed(3)} ms）与 MobileNetV3-Small（${mobTime.toFixed(3)} ms）则展示了迁移学习模型在复杂特征下的预测能力。`;
        }
        const speedConclusionEl = document.getElementById('speed-conclusion');
        if (speedConclusionEl) speedConclusionEl.innerHTML = conclusion;

        // 5. 请求 GradCAM 热力图
        fetchGradCAM(base64Data);
    })
    .catch(err => {
        console.error(err);
        alert("模型预测失败，请确保您已完成本地训练并生成了权重文件。");
    });
}

function fetchGradCAM(base64Data) {
    fetch('/api/gradcam', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Data })
    })
    .then(res => res.json())
    .then(data => {
        const section = document.getElementById('gradcam-section');
        if (section) section.style.display = 'block';

        function renderGradCAM(containerId, modelData) {
            const container = document.getElementById(containerId);
            if (!container) return;
            if (!modelData || !modelData.heatmap) {
                container.innerHTML = '<span style="color:var(--text-secondary);font-size:0.8rem;">未加载权重</span>';
                return;
            }
            container.innerHTML = `
                <img src="data:image/png;base64,${modelData.heatmap}" style="width:100%;border-radius:8px;border:1px solid rgba(255,255,255,0.1);">
                <div style="font-size:0.75rem;color:var(--text-secondary);margin-top:0.3rem;">预测: ${modelData.predicted_class} (${(modelData.confidence*100).toFixed(1)}%)</div>
            `;
        }

        renderGradCAM('gradcam-resnet', data.resnet);
        renderGradCAM('gradcam-mobilenet', data.mobilenet);
        renderGradCAM('gradcam-custom', data.custom_cnn);
    })
    .catch(err => console.error("GradCAM 请求失败:", err));
}

function renderTop5List(containerId, top5Data, type) {
    const container = document.getElementById(containerId);
    let html = '';
    top5Data.forEach(item => {
        const percentage = (item.prob * 100).toFixed(1);
        html += `
            <div class="prob-item">
                <div class="prob-label">
                    <span>字符: ${item.char}</span>
                    <span>${percentage}%</span>
                </div>
                <div class="prob-bar ${type}">
                    <div class="prob-fill" style="width: ${percentage}%;"></div>
                </div>
            </div>
        `;
    });
    container.innerHTML = html;
}

// ----------------- 车牌识别三模型逐字符对比表 -----------------
function renderPlateComparisonTable(predictions) {
    const tableSection = document.getElementById('plate-comparison-table');
    if (!tableSection) return;

    const resnet = predictions.resnet;
    const mobilenet = predictions.mobilenet;
    const custom = predictions.custom_cnn;
    const ensemble = predictions.ensemble;

    // 至少一个模型成功才显示
    if (resnet.status !== 'success' && mobilenet.status !== 'success' && custom.status !== 'success') {
        tableSection.style.display = 'none';
        return;
    }

    const resText = resnet.text || '';
    const mobText = mobilenet.text || '';
    const custText = custom.text || '';
    const ensText = ensemble ? (ensemble.text || '') : '';
    const maxLen = Math.max(resText.length, mobText.length, custText.length, ensText.length);

    if (maxLen === 0) {
        tableSection.style.display = 'none';
        return;
    }

    tableSection.style.display = 'block';

    // 表头
    const thead = document.getElementById('plate-comparison-thead');
    thead.innerHTML = `
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
            <th style="padding: 6px 10px; text-align: center; color: var(--text-secondary);">位置</th>
            <th style="padding: 6px 10px; text-align: center; color: var(--primary);">ResNet18</th>
            <th style="padding: 6px 10px; text-align: center; color: var(--warning);">MobileNetV3</th>
            <th style="padding: 6px 10px; text-align: center; color: var(--success);">CustomCNN</th>
            <th style="padding: 6px 10px; text-align: center; color: #c084fc;">集成融合</th>
            <th style="padding: 6px 10px; text-align: center; color: var(--text-secondary);">一致性</th>
        </tr>
    `;

    // 逐字符对比
    const tbody = document.getElementById('plate-comparison-tbody');
    let rows = '';
    let agreeCount = 0;

    for (let i = 0; i < maxLen; i++) {
        const r = resText[i] || '-';
        const m = mobText[i] || '-';
        const c = custText[i] || '-';
        const e = ensText[i] || '-';

        // 判断模型是否一致（忽略缺失的）
        const chars = [r, m, c, e].filter(ch => ch !== '-');
        const allAgree = chars.length >= 2 && chars.every(ch => ch === chars[0]);
        if (allAgree) agreeCount++;

        const agreeStyle = allAgree
            ? 'color: var(--success);'
            : 'color: var(--danger); font-weight: bold;';
        const agreeText = allAgree ? '✓ 一致' : '✗ 分歧';
        const rowBg = allAgree ? '' : 'background: rgba(239,68,68,0.05);';

        // 每个字符单元格：如果跟共识不同则高亮
        function cellStyle(char, consensus) {
            if (char === '-') return 'opacity: 0.4;';
            if (!allAgree && char !== consensus) return 'color: var(--danger); font-weight: bold;';
            return '';
        }
        const consensus = allAgree ? chars[0] : '';

        rows += `
            <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); ${rowBg}">
                <td style="padding: 4px 10px; text-align: center; color: var(--text-secondary);">${i + 1}</td>
                <td style="padding: 4px 10px; text-align: center; font-weight: 600; ${cellStyle(r, consensus)}">${r}</td>
                <td style="padding: 4px 10px; text-align: center; font-weight: 600; ${cellStyle(m, consensus)}">${m}</td>
                <td style="padding: 4px 10px; text-align: center; font-weight: 600; ${cellStyle(c, consensus)}">${c}</td>
                <td style="padding: 4px 10px; text-align: center; font-weight: 600; color: #c084fc; ${cellStyle(e, consensus)}">${e}</td>
                <td style="padding: 4px 10px; text-align: center; ${agreeStyle}">${agreeText}</td>
            </tr>
        `;
    }
    tbody.innerHTML = rows;

    // 共识结论
    const consensusDiv = document.getElementById('plate-consensus');
    if (agreeCount === maxLen) {
        consensusDiv.innerHTML = `<span style="color: var(--success);"><i class="fa-solid fa-check-circle"></i> 三模型完全一致 (${agreeCount}/${maxLen})，识别结果可信度高</span>`;
    } else {
        consensusDiv.innerHTML = `<span style="color: var(--warning);"><i class="fa-solid fa-triangle-exclamation"></i> 三模型存在分歧 (${agreeCount}/${maxLen} 一致)，建议人工复核</span>`;
    }
}
